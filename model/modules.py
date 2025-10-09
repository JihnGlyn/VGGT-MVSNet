import torch
import torch.nn as nn
import torch.nn.functional as F
from model.submodules import *


class FeatureFuse(nn.Module):
    def __init__(self, base_channels=16):
        super(FeatureFuse, self).__init__()
        self.deconv1 = Deconv2d(in_channels=16, out_channels=base_channels, stride=2, padding=1, output_padding=1)
        self.conv1 = Conv2d(3, base_channels, kernel_size=3, pad=1)
        self.fuse1 = Conv2d(base_channels * 2, base_channels, kernel_size=3, pad=1)

    def forward(self, image, vggt_feature):
        # image:[B,3,H/2,W/2] vggt_feature:[B,16,H/4,W/4]
        conv = self.conv1(image)
        deconv = self.deconv1(vggt_feature)
        fuse = self.fuse1(torch.cat((conv, deconv), dim=1))
        # fuse:[B,16,H/2,W/2]
        return fuse


class RefineNet(nn.Module):
    def __init__(self, base_channels=8):
        super(RefineNet, self).__init__()
        # img: [B,3,H,W]
        self.conv0 = Conv2d(in_channels=3, out_channels=base_channels)
        # depth map:[B,1,H/2,W/2]
        self.conv1 = Conv2d(in_channels=1, out_channels=base_channels)
        self.conv2 = Conv2d(in_channels=base_channels, out_channels=base_channels)
        self.deconv = nn.ConvTranspose2d(
            in_channels=base_channels, out_channels=base_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False
        )
        self.bn = nn.BatchNorm2d(base_channels)
        self.conv3 = Conv2d(in_channels=base_channels * 2, out_channels=base_channels)
        self.res = nn.Conv2d(in_channels=base_channels, out_channels=1, kernel_size=3, padding=1, bias=False)

    def forward(self, img, depth_0, depth_min, depth_max):
        batch_size = depth_min.size()[0]
        # pre-scale the depth map into [0,1]
        depth = (depth_0 - depth_min.view(batch_size, 1, 1, 1)) / (depth_max - depth_min).view(batch_size, 1, 1, 1)

        conv0 = self.conv0(img)
        deconv = F.relu(self.bn(self.deconv(self.conv2(self.conv1(depth)))), inplace=True)
        # depth residual
        res = self.res(self.conv3(torch.cat((deconv, conv0), dim=1)))
        del conv0
        del deconv

        depth = F.interpolate(depth, scale_factor=2.0, mode="nearest") + res
        # convert the normalized depth back
        return depth * (depth_max - depth_min).view(batch_size, 1, 1, 1) + depth_min.view(batch_size, 1, 1, 1)


class CostRegNetBig(nn.Module):
    def __init__(self, in_channels, base_channels=8):
        super(CostRegNetBig, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)

        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Conv3d(base_channels * 4, base_channels * 8, stride=2, padding=1)
        self.conv6 = Conv3d(base_channels * 8, base_channels * 8, padding=1)

        self.conv7 = Deconv3d(base_channels * 8, base_channels * 4, stride=2, padding=1, output_padding=1)

        self.conv9 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)

        self.conv11 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)

        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        conv4 = self.conv4(self.conv3(conv2))
        x = self.conv6(self.conv5(conv4))
        x = conv4 + self.conv7(x)
        x = conv2 + self.conv9(x)
        x = conv0 + self.conv11(x)
        x = self.prob(x)
        return x


class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels=16):
        super(CostRegNet, self).__init__()
        self.conv0 = Conv3d(in_channels, base_channels, padding=1)

        self.conv1 = Conv3d(base_channels, base_channels * 2, stride=2, padding=1)
        self.conv2 = Conv3d(base_channels * 2, base_channels * 2, padding=1)
        self.conv3 = Conv3d(base_channels * 2, base_channels * 4, stride=2, padding=1)
        self.conv4 = Conv3d(base_channels * 4, base_channels * 4, padding=1)

        self.conv5 = Deconv3d(base_channels * 4, base_channels * 2, stride=2, padding=1, output_padding=1)
        self.conv6 = Deconv3d(base_channels * 2, base_channels * 1, stride=2, padding=1, output_padding=1)
        self.prob = nn.Conv3d(base_channels, 1, 3, stride=1, padding=1, bias=False)

    def forward(self, x):
        conv0 = self.conv0(x)
        conv2 = self.conv2(self.conv1(conv0))
        x = self.conv4(self.conv3(conv2))
        x = conv2 + self.conv5(x)
        x = conv0 + self.conv6(x)
        x = self.prob(x)
        return x


class SimilarityNet(nn.Module):
    """Similarity Net, used in Evaluation module (adaptive evaluation step)
    1. Do 1x1x1 convolution on aggregated cost [B, G, Ndepth, H, W] among all the source views,
        where G is the number of groups
    2. Perform adaptive spatial cost aggregation to get final cost (scores)
    """

    def __init__(self, G: int) -> None:
        """Initialize method

        Args:
            G: the feature channels of input will be divided evenly into G groups
        """
        super(SimilarityNet, self).__init__()

        self.conv0 = Conv3d(in_channels=G, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = Conv3d(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.similarity = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)

    def forward(self, x1: torch.Tensor, grid: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        """Forward method for SimilarityNet

        Args:
            x1: [B, G, Ndepth, H, W], where G is the number of groups, aggregated cost among all the source views with
                pixel-wise view weight
            grid: position of sampling points in adaptive spatial cost aggregation, (B, evaluate_neighbors*H, W, 2)
            weight: weight of sampling points in adaptive spatial cost aggregation, combination of
                feature weight and depth weight, [B,Ndepth,1,H,W]

        Returns:
            final cost: in the shape of [B,Ndepth,H,W]
        """

        batch, G, num_depth, height, width = x1.size()
        num_neighbors = grid.size()[1] // height

        # [B,Ndepth,num_neighbors,H,W]
        x1 = F.grid_sample(
            input=self.similarity(self.conv1(self.conv0(x1))).squeeze(1),
            grid=grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        ).view(batch, num_depth, num_neighbors, height, width)

        return torch.sum(x1 * weight, dim=2)


def Propagation(depth_sample: torch.Tensor, grid: torch.Tensor):
    batch, num_depth, height, width = depth_sample.size()
    num_neighbors = grid.size()[1] // height
    propagate_depth_sample = F.grid_sample(
        depth_sample[:, num_depth // 2, :, :].unsqueeze(1),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False
    ).view(batch, num_neighbors, height, width)
    return torch.sort(torch.cat((depth_sample, propagate_depth_sample), dim=1), dim=1)[0]


def get_grid(
        grid_type: str, batch: int, height: int, width: int,
        offset: torch.Tensor, device: torch.device, neighbors: int, dilation: int,
) -> torch.Tensor:
    """Compute the offset for adaptive propagation or spatial cost aggregation in adaptive evaluation

        Args:
            grid_type: type of grid - propagation (1) or evaluation (2)
            batch: batch size
            height: grid height
            width: grid width
            neighbors:
            dilation: grid dilation
            offset: grid offset
            device: device on which to place tensor

        Returns:
            generated grid: in the shape of [batch, propagate_neighbors*H, W, 2]
        """
    if grid_type == "propagation":
        if neighbors == 4:  # if 4 neighbors to be sampled in propagation
            original_offset = [[-dilation, 0], [0, -dilation], [0, dilation], [dilation, 0]]
        elif neighbors == 8:  # if 8 neighbors to be sampled in propagation
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
        elif neighbors == 16:  # if 16 neighbors to be sampled in propagation
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
            for i in range(len(original_offset)):
                offset_x, offset_y = original_offset[i]
                original_offset.append([2 * offset_x, 2 * offset_y])
        else:
            raise NotImplementedError
    elif grid_type == "evaluation":
        dilation = dilation - 1  # dilation of evaluation is a little smaller than propagation
        if neighbors == 9:  # if 9 neighbors to be sampled in evaluation
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, 0],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
        elif neighbors == 17:  # if 17 neighbors to be sampled in evaluation
            original_offset = [
                [-dilation, -dilation],
                [-dilation, 0],
                [-dilation, dilation],
                [0, -dilation],
                [0, 0],
                [0, dilation],
                [dilation, -dilation],
                [dilation, 0],
                [dilation, dilation],
            ]
            for i in range(len(original_offset)):
                offset_x, offset_y = original_offset[i]
                if offset_x != 0 or offset_y != 0:
                    original_offset.append([2 * offset_x, 2 * offset_y])
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

    with torch.no_grad():
        y_grid, x_grid = torch.meshgrid(
            [
                torch.arange(0, height, dtype=torch.float32, device=device),
                torch.arange(0, width, dtype=torch.float32, device=device),
            ]
        )
        y_grid, x_grid = y_grid.contiguous().view(height * width), x_grid.contiguous().view(height * width)
        xy = torch.stack((x_grid, y_grid))  # [2, H*W]
        xy = torch.unsqueeze(xy, 0).repeat(batch, 1, 1)  # [B, 2, H*W]

    xy_list = []
    for i in range(len(original_offset)):
        original_offset_y, original_offset_x = original_offset[i]
        offset_x = original_offset_x + offset[:, 2 * i, :].unsqueeze(1)
        offset_y = original_offset_y + offset[:, 2 * i + 1, :].unsqueeze(1)
        xy_list.append((xy + torch.cat((offset_x, offset_y), dim=1)).unsqueeze(2))

    xy = torch.cat(xy_list, dim=2)  # [B, 2, 9, H*W]

    del xy_list
    del x_grid
    del y_grid

    x_normalized = xy[:, 0, :, :] / ((width - 1) / 2) - 1
    y_normalized = xy[:, 1, :, :] / ((height - 1) / 2) - 1
    del xy
    grid = torch.stack((x_normalized, y_normalized), dim=3)  # [B, 9, H*W, 2]
    del x_normalized
    del y_normalized
    return grid.view(batch, len(original_offset) * height, width, 2)


class PropaNet(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int) -> None:
        super(PropaNet, self).__init__()
        self.rgb = Conv2d(in_channels=in_channels, out_channels=16)
        self.output = nn.Conv2d(in_channels=16, out_channels=out_channels, kernel_size=3,
                                padding=dilation, dilation=dilation, bias=True)
        nn.init.constant_(self.output.weight, 0)
        nn.init.constant_(self.output.bias, 0)

    # @make_nograd_func
    def forward(self, ref_fea: torch.Tensor) -> torch.Tensor:
        x1 = self.rgb(ref_fea)
        output = self.output(x1)

        return output


class PixelwiseNet(nn.Module):
    """Pixelwise Net: A simple pixel-wise view weight network, composed of 1x1x1 convolution layers
    and sigmoid nonlinearities, takes the initial set of similarities to output a number between 0 and 1 per
    pixel as estimated pixel-wise view weight.

    1. The Pixelwise Net is used in adaptive evaluation step
    2. The similarity is calculated by ref_feature and other source_features warped by differentiable_warping
    3. The learned pixel-wise view weight is estimated in the first iteration of Patchmatch and kept fixed in the
    matching cost computation.
    """

    def __init__(self, G: int) -> None:
        """Initialize method

        Args:
            G: the feature channels of input will be divided evenly into G groups
        """
        super(PixelwiseNet, self).__init__()
        self.conv0 = Conv3d(in_channels=G, out_channels=16, kernel_size=1, stride=1, padding=0)
        self.conv1 = Conv3d(in_channels=16, out_channels=8, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1: torch.Tensor) -> torch.Tensor:
        """Forward method for PixelwiseNet

        Args:
            x1: pixel-wise view weight, [B, G, Ndepth, H, W], where G is the number of groups
        """
        # [B,1,H,W]
        return torch.max(self.output(self.conv2(self.conv1(self.conv0(x1))).squeeze(1)), dim=1)[0].unsqueeze(1)


def depth_weight(depth_sample, grid, neighbors):
    """Calculate depth weight
    1. Adaptive spatial cost aggregation
    2. Weight based on depth difference of sampling points and center pixel

    Args:
        depth_sample: sample depth map, (B,Ndepth,H,W)
        grid: position of sampling points in adaptive spatial cost aggregation, (B, neighbors*H, W, 2)
        neighbors: number of neighbors to be sampled in evaluation

    Returns:
        depth weight
    """
    batch, num_depth, height, width = depth_sample.size()
    depth_min, depth_max = depth_sample.min(), depth_sample.max()
    inverse_depth_min = 1.0 / depth_min
    inverse_depth_max = 1.0 / depth_max
    interval_scale = 0.25
    # normalization
    x = 1.0 / depth_sample
    del depth_sample
    x = (x - inverse_depth_max.view(batch, 1, 1, 1)) / (inverse_depth_min - inverse_depth_max).view(batch, 1, 1, 1)

    x1 = F.grid_sample(
        x, grid, mode="bilinear", padding_mode="border", align_corners=False
    ).view(batch, num_depth, neighbors, height, width)
    del grid

    # [B,Ndepth,N_neighbors,H,W]
    x1 = torch.abs(x1 - x.unsqueeze(2)) / interval_scale
    del x

    # sigmoid output approximate to 1 when x=4
    return torch.sigmoid(4.0 - 2.0 * x1.clamp(min=0, max=4)).detach()
