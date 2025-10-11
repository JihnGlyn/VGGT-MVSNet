import torch
import torch.nn as nn
import torch.nn.functional as F


def resize_img(image):
    # image: [B, 3, H, W]
    _, _, h, w = image.shape
    max_edge = max(h, w)
    if max_edge <= 2072:
        return image

    scale = 2072 / max_edge
    if h >= w:
        new_h = 2072
        new_w = int(w * scale)
    else:
        new_w = 2072
        new_h = int(h * scale)

    image = F.interpolate(image, size=(new_h, new_w), mode='bilinear', align_corners=False)
    return image


def create_stage_images(image: torch.Tensor):
    return [
        image,
        F.interpolate(image, scale_factor=0.5, mode="nearest"),
        F.interpolate(image, scale_factor=0.25, mode="nearest"),
    ]


def depth_regression(p, depth_values):
    if depth_values.dim() <= 2:
        # print("regression dim <= 2")
        depth_values = depth_values.view(*depth_values.shape, 1, 1)
    depth = torch.sum(p * depth_values, 1)

    return depth


def depth_wta(p, depth_values):
    """Winner take all."""
    wta_index_map = torch.argmax(p, dim=1, keepdim=True).type(torch.long)
    wta_depth_map = torch.gather(depth_values, 1, wta_index_map).squeeze(1)
    photometric_confidence = torch.max(p, dim=1)[0]
    return wta_depth_map, photometric_confidence


def conf_regression(p, n=4):
    ndepths = p.size(1)
    # photometric confidence
    prob_volume_sum4 = n * F.avg_pool3d(F.pad(p.unsqueeze(1), pad=[0, 0, 0, 0, n // 2 - 1, n // 2]),
                                        (n, 1, 1), stride=1, padding=0).squeeze(1)
    depth_index = depth_regression(p.detach(),
                                   depth_values=torch.arange(ndepths, device=p.device, dtype=torch.float)).long()
    depth_index = depth_index.clamp(min=0, max=ndepths - 1)
    conf = torch.gather(prob_volume_sum4, 1, depth_index)
    return conf.squeeze(1)


def get_cur_depth_range_samples(depth, num_depths, depth_range_scale, min_depth, max_depth, init=False):
    device = min_depth.device
    b, h, w = depth.shape
    depth = depth.unsqueeze(1)
    inverse_min_depth = 1.0 / min_depth
    inverse_max_depth = 1.0 / max_depth
    if init is True:
        depth_sample = torch.arange(start=0, end=num_depths, step=1, device=device).view(
            1, num_depths, 1, 1).repeat(b, 1, h, w).float()

        depth_sample = inverse_max_depth.view(b, 1, 1, 1) + depth_sample / num_depths * (
                inverse_min_depth.view(b, 1, 1, 1) - inverse_max_depth.view(b, 1, 1, 1))

        return 1.0 / depth_sample
    else:
        depth_sample = (
            torch.arange(-num_depths // 2, num_depths // 2, 1, device=device)
            .view(1, num_depths, 1, 1).repeat(b, 1, h, w).float()
        )
        inverse_depth_interval = (inverse_min_depth - inverse_max_depth) * depth_range_scale
        inverse_depth_interval = inverse_depth_interval.view(b, 1, 1, 1)

        depth_sample = 1.0 / depth.detach() + inverse_depth_interval * depth_sample

        depth_clamped = []
        del depth
        for k in range(b):
            depth_clamped.append(
                torch.clamp(depth_sample[k], min=inverse_max_depth[k], max=inverse_min_depth[k]).unsqueeze(0)
            )

        return 1.0 / torch.cat(depth_clamped, dim=0)


def homo_warping(src_fea, src_proj, ref_proj, depth_samples):
    # src_fea: [B, C, Hin, Win] source features, for each source view in batch
    # src_proj: [B, 4, 4] source camera projection matrix, for each source view in batch
    # ref_proj: [B, 4, 4] reference camera projection matrix, for each ref view in batch
    # depth_samples: [B, Ndepth, Hout, Wout] virtual depth layers
    # out: [B, C, Ndepth, H, W]
    batch, num_depth, height, width = depth_samples.shape
    channels = src_fea.shape[1]

    with torch.no_grad():
        proj = torch.matmul(src_proj, torch.inverse(ref_proj))
        rot = proj[:, :3, :3]  # [B,3,3]
        trans = proj[:, :3, 3:4]  # [B,3,1]

        y, x = torch.meshgrid(
            [
                torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                torch.arange(0, width, dtype=torch.float32, device=src_fea.device),
            ]
        )
        y, x = y.contiguous().view(height * width), x.contiguous().view(height * width)
        xyz = torch.unsqueeze(torch.stack((x, y, torch.ones_like(x))), 0).repeat(batch, 1, 1)  # [B, 3, H*W]

        rot_depth_xyz = torch.matmul(rot, xyz).unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(
            batch, 1, num_depth, height * width
        )  # [B, 3, Ndepth, H*W]
        proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
        # avoid negative depth
        negative_depth_mask = proj_xyz[:, 2:] <= 1e-3
        proj_xyz[:, 0:1][negative_depth_mask] = float(width)
        proj_xyz[:, 1:2][negative_depth_mask] = float(height)
        proj_xyz[:, 2:3][negative_depth_mask] = 1.0
        grid = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
        proj_x_normalized = grid[:, 0, :, :] / ((width - 1) / 2) - 1  # [B, Ndepth, H*W]
        proj_y_normalized = grid[:, 1, :, :] / ((height - 1) / 2) - 1
        grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]

    return F.grid_sample(
        src_fea,
        grid.view(batch, num_depth * height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(batch, channels, num_depth, height, width)


def homo_warping_grad(src_fea, src_proj, ref_proj, depth_samples):
    # src_fea: [B, C, Hin, Win] source features, for each source view in batch
    # src_proj: [B, 4, 4] source camera projection matrix, for each source view in batch
    # ref_proj: [B, 4, 4] reference camera projection matrix, for each ref view in batch
    # depth_samples: [B, Ndepth, Hout, Wout] virtual depth layers
    # out: [B, C, Ndepth, H, W]
    batch, num_depth, height, width = depth_samples.shape
    channels = src_fea.shape[1]

    proj = torch.matmul(src_proj, torch.inverse(ref_proj))
    rot = proj[:, :3, :3]  # [B,3,3]
    trans = proj[:, :3, 3:4]  # [B,3,1]

    y, x = torch.meshgrid(
        [
            torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
            torch.arange(0, width, dtype=torch.float32, device=src_fea.device),
        ]
    )
    y, x = y.contiguous().view(height * width), x.contiguous().view(height * width)
    xyz = torch.unsqueeze(torch.stack((x, y, torch.ones_like(x))), 0).repeat(batch, 1, 1)  # [B, 3, H*W]

    rot_depth_xyz = torch.matmul(rot, xyz).unsqueeze(2).repeat(1, 1, num_depth, 1) * depth_samples.view(
        batch, 1, num_depth, height * width
    )  # [B, 3, Ndepth, H*W]
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1, 1)  # [B, 3, Ndepth, H*W]
    # avoid negative depth
    negative_depth_mask = proj_xyz[:, 2:] <= 1e-3
    proj_xyz[:, 0:1][negative_depth_mask] = float(width)
    proj_xyz[:, 1:2][negative_depth_mask] = float(height)
    proj_xyz[:, 2:3][negative_depth_mask] = 1.0
    grid = proj_xyz[:, :2, :, :] / proj_xyz[:, 2:3, :, :]  # [B, 2, Ndepth, H*W]
    proj_x_normalized = grid[:, 0, :, :] / ((width - 1) / 2) - 1  # [B, Ndepth, H*W]
    proj_y_normalized = grid[:, 1, :, :] / ((height - 1) / 2) - 1
    grid = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, Ndepth, H*W, 2]

    return F.grid_sample(
        src_fea,
        grid.view(batch, num_depth * height, width, 2),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).view(batch, channels, num_depth, height, width)


def init_bn(module):
    if module.weight is not None:
        nn.init.ones_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)
    return


def init_uniform(module, init_method):
    if module.weight is not None:
        if init_method == "kaiming":
            nn.init.kaiming_uniform_(module.weight)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(module.weight)
    return


class Conv3d(nn.Module):
    """Applies a 3D convolution (optionally with batch normalization and relu activation)
    over an input signal composed of several input planes.

    Attributes:
        conv (nn.Module): convolution module
        bn (nn.Module): batch normalization module
        relu (bool): whether to activate by relu

    Notes:
        Default momentum for batch normalization is set to be 0.01,

    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Conv3d, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride,
                              bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Deconv3d(nn.Module):
    """Applies a 3D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv3d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm3d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

        # assert init_method in ["kaiming", "xavier"]
        # self.init_weights(init_method)

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)


class Conv2d(nn.Module):
    """Implements 2d Convolution + batch normalization + ReLU"""

    def __init__(
            self,
            in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, pad: int = 1, dilation: int = 1,
    ) -> None:
        super(Conv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, dilation=dilation, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """forward method"""
        return F.relu(self.bn(self.conv(x)), inplace=True)


class Deconv2d(nn.Module):
    """Applies a 2D deconvolution (optionally with batch normalization and relu activation)
       over an input signal composed of several input planes.

       Attributes:
           conv (nn.Module): convolution module
           bn (nn.Module): batch normalization module
           relu (bool): whether to activate by relu

       Notes:
           Default momentum for batch normalization is set to be 0.01,

       """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 relu=True, bn=True, bn_momentum=0.1, init_method="xavier", **kwargs):
        super(Deconv2d, self).__init__()
        self.out_channels = out_channels
        assert stride in [1, 2]
        self.stride = stride

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride,
                                       bias=(not bn), **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, momentum=bn_momentum) if bn else None
        # self.bn = nn.GroupNorm(8, out_channels) if bn else None
        self.relu = relu

    def forward(self, x):
        y = self.conv(x)
        if self.bn is not None:
            x = self.bn(y)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x

    def init_weights(self, init_method):
        """default initialization"""
        init_uniform(self.conv, init_method)
        if self.bn is not None:
            init_bn(self.bn)
