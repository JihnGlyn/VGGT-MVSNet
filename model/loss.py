import torch
import torch.nn as nn
import torch.nn.functional as F
from model.submodules import *

device = torch.device("cuda")


class unsup_loss(nn.Module):
    def __init__(self):
        super(unsup_loss, self).__init__()
        self.ssim = SSIM()
        self.mse = nn.MSELoss()
        self.s_w = [10, 10, 10]
        self.l_w = [1, 1]

    def forward(self, imgs, feas, projs, depths, masks):

        ssim_loss = 0.0
        mse_loss = 0.0

        ssim_l = 0.0
        ssim_m = 0.0
        ssim_h = 0.0
        mse_l = 0.0
        mse_m = 0.0
        mse_h = 0.0
        # LOW-RES
        ref_i, src_is = imgs["level_2"][:, 0], imgs["level_2"][:, 1:2]  # USING ONLY 2 SRCS FOR LOSS
        ref_f, src_fs = feas["level_l"][:, 0], feas["level_l"][:, 1:2]
        ref_pr, src_prs = projs["level_l"][0], projs["level_l"][1:2]
        mask = masks["level_l"]
        d = depths[0]   # [B,1,H,W]
        for i, (src_i, src_f, src_pr) in enumerate(zip(src_is, src_fs, src_prs)):
            ref_i, ref_f = ref_i * mask, src_f * mask
            warped_img = homo_warping_grad(src_is[:, i], src_pr, ref_pr, d).squeeze(2) * mask  # [B,C,1,H,W] -> [B,C,H,W]
            warped_fea = homo_warping_grad(src_fs[:, i], src_pr, ref_pr, d).squeeze(2) * mask  # [B,C,1,H,W] -> [B,C,H,W]
            ssim_l = torch.mean((self.ssim(ref_i, warped_img)) + torch.mean(self.ssim(ref_f, warped_fea))) + ssim_l
            mse_l = (self.mse(ref_i, warped_img) + self.mse(ref_f, warped_fea)) + mse_l
        ssim_loss = ssim_l + ssim_loss
        mse_loss = mse_l + mse_loss
        lr = ssim_l + mse_l

        # MID-RES
        ref_i, src_is = imgs["level_1"][:, 0], imgs["level_1"][:, 1:2]  # USING ONLY 2 SRCS FOR LOSS
        ref_f, src_fs = feas["level_m"][:, 0], feas["level_m"][:, 1:2]
        ref_pr, src_prs = projs["level_m"][0], projs["level_m"][1:2]
        mask = masks["level_m"]
        d = depths[1]
        for i, (src_i, src_f, src_pr) in enumerate(zip(src_is, src_fs, src_prs)):
            ref_i, ref_f = ref_i * mask, src_f * mask
            warped_img = homo_warping_grad(src_is[:, i], src_pr, ref_pr, d).squeeze(2) * mask  # [B,C,1,H,W] -> [B,C,H,W]
            warped_fea = homo_warping_grad(src_fs[:, i], src_pr, ref_pr, d).squeeze(2) * mask  # [B,C,1,H,W] -> [B,C,H,W]
            ssim_m = torch.mean((self.ssim(ref_i, warped_img)) + torch.mean(self.ssim(ref_f, warped_fea))) + ssim_m
            mse_m = (self.mse(ref_i, warped_img) + self.mse(ref_f, warped_fea)) + mse_m
        ssim_loss = ssim_m + ssim_loss
        mse_loss = mse_m + mse_loss
        mr = ssim_m + mse_m

        # HIGH-RES
        ref_i, src_is = imgs["level_0"][:, 0], imgs["level_0"][:, 1:2]  # USING ONLY 2 SRCS FOR LOSS
        ref_f, src_fs = feas["level_h"][:, 0], feas["level_h"][:, 1:2]
        ref_pr, src_prs = projs["level_h"][0], projs["level_h"][1:2]
        mask = masks["level_h"]
        d = depths[2]
        for i, (src_i, src_f, src_pr) in enumerate(zip(src_is, src_fs, src_prs)):
            ref_i, ref_f = ref_i * mask, src_f * mask
            warped_img = homo_warping_grad(src_is[:, i], src_pr, ref_pr, d).squeeze(2) * mask  # [B,C,1,H,W] -> [B,C,H,W]
            warped_fea = homo_warping_grad(src_fs[:, i], src_pr, ref_pr, d).squeeze(2) * mask  # [B,C,1,H,W] -> [B,C,H,W]
            ssim_h = torch.mean((self.ssim(ref_i, warped_img)) + torch.mean(self.ssim(ref_f, warped_fea))) + ssim_h
            mse_h = (self.mse(ref_i, warped_img) + self.mse(ref_f, warped_fea)) + mse_h
        ssim_loss = ssim_h + ssim_loss
        mse_loss = mse_h + mse_loss
        hr = ssim_h + mse_h

        # total_loss = self.l_w[0] * ssim_loss + self.l_w[1] * mse_loss
        total_loss = self.s_w[0] * lr + self.s_w[1] * mr + self.s_w[2] * hr

        return total_loss, ssim_loss, mse_loss, lr, mr, hr


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.mask_pool = nn.AvgPool2d(3, 1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        # [B,C,H,W]
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        output = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        return output
