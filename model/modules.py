import torch
import torch.nn as nn
import torch.nn.functional as F
from model.submodules import *


class FeatureNet(nn.Module):
    def __init__(self, base_channels=8):
        super(FeatureNet, self).__init__()
        self.base_channels = base_channels

        self.conv0 = nn.Sequential(
                Conv2d(3, base_channels, 3, 1),
                Conv2d(base_channels, base_channels, 3, 1))

        self.conv1 = nn.Sequential(
                Conv2d(base_channels, base_channels * 2, 5, stride=2, pad=2),
                Conv2d(base_channels * 2, base_channels * 2, 3, 1),
                Conv2d(base_channels * 2, base_channels * 2, 3, 1))

        self.conv2 = nn.Sequential(
                Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, pad=2),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1),
                Conv2d(base_channels * 4, base_channels * 4, 3, 1))

        self.out1 = Conv2d(base_channels * 4, base_channels * 4, 1)

        final_chs = base_channels * 4
        self.inner1 = nn.Conv2d(base_channels * 2, final_chs, 1, bias=True)
        self.inner2 = nn.Conv2d(base_channels * 1, final_chs, 1, bias=True)

        self.out2 = Conv2d(final_chs, final_chs, 3, 1)
        self.out3 = Conv2d(final_chs, final_chs, 3, 1)

        self.out_channels = [4 * base_channels, base_channels * 2, base_channels]

    def forward(self, x):
        """forward.

        :param x: [B, C, H, W]
        :return outputs: stage1 [B, 32, 128, 160], stage2 [B, 16, 256, 320], stage3 [B, 8, 512, 640]
        """
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = {}
        out = self.out1(intra_feat)
        outputs["level_l"] = out
        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        out = self.out2(intra_feat)
        outputs["level_m"] = out

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        out = self.out3(intra_feat)
        outputs["level_h"] = out

        return outputs


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

