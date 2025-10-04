from modules import *
from submodules import *


class PatchmatchNet(nn.Module):
    def __init__(self, G):
        super(PatchmatchNet, self).__init__()
        self.G = G
        self.propa_conv = PropaNet()
        self.eval_conv = PropaNet()
        self.pixel_wise_net = PixelwiseNet(self.G)
        self.softmax = nn.LogSoftmax(dim=1)
        self.reg_net = SimilarityNet(G)

    def forward(self, ref_fea, src_feas, ref_proj, src_projs, depth_hypo, view_weights, num_depth=8, wta=False):
        B, D, H, W = depth_hypo.shape
        C = ref_fea.shape[1]
        device = ref_fea.device

        # propa_offset = torch.zeros(B, 2 * self.grab_neighbors, H * W).to(device)
        propa_offset = self.propa_conv(ref_fea).view(B, 2 * self.propagate_neighbors, H * W)
        propa_grid = get_grid(grid_type="propagation", batch=B, height=H, width=W, offset=propa_offset, device=device,
                              neighbors=num_depth, dilation=4)
        depth_sample = Propagation(depth_sample=depth_hypo, grid=propa_grid)

        ref_volume = ref_fea.unsqueeze(2).repeat(1, 1, D, 1, 1)
        cor_weight_sum = 1e-8
        cor_feats = 0
        view_weights_list = []
        pixel_wise_weight_sum = 1e-5 * torch.ones((B, 1, 1, H, W), dtype=torch.float32, device=device)
        for i, (src_fea, src_proj) in enumerate(zip(src_feas, src_projs)):
            # [B, C, D, H, W] -> [B, G, C/G, D, H, W]
            warped_src = homo_warping(src_fea, src_proj, ref_proj, depth_hypo).view(B, self.G, C // self.G, D, H, W)
            cor_feat = (warped_src * ref_volume).mean(2)  # B G D H W
            if view_weights is None:
                view_weight = self.pixel_wise_net(cor_feat)
                view_weights_list.append(view_weight)
            else:
                view_weight = view_weights[:, i].unsqueeze(1)

            cor_weight_sum += cor_feat * view_weight.unsqueeze(1)  # B D H W
            pixel_wise_weight_sum += view_weight.unsqueeze(1)

        similarity = cor_weight_sum.div_(pixel_wise_weight_sum)
        score = self.reg_net(similarity, eval_grid, weight)
        score = torch.exp(self.softmax(score))

        if view_weights is None:
            view_weights = torch.cat(view_weights_list, dim=1)  # [B,4,H,W], 4 is the number of source views

        if not wta:
            depth = depth_regression(score, depth_hypo)
            conf = conf_regression(score, 4)
        else:
            depth, conf = depth_wta(score, depth_hypo)

        output_stage = {
            "depth": depth,
            "photometric_confidence": conf,
            "view_weights": view_weights,
        }
        return output_stage


class MVSNet(nn.Module):
    def __init__(self, G):
        super(MVSNet, self).__init__()
        self.G = G
        self.reg_net = CostRegNet(G)
        self.pixel_wise_net = PixelwiseNet(self.G)

    def forward(self, ref_fea, src_feas, ref_proj, src_projs, depth_hypo, view_weights, wta=False):
        """forward

        :param ref_fea: [B, C, H, W]
        :param src_feas: (N-1) [B, C, H, W]
        :param ref_proj: [B, 4, 4]
        :param src_projs: (N-1) [B, 4, 4]
        :param depth_hypo: [B, D, H, W]
        :param view_weights: [B, N-1, H, W]
        :param wta: bool
        :return: {depth: [B, H, W], conf: [B, H, W], view_weights: [B, N-1, H, W]}
        """
        # TODO: step 1. feature map homographic warping and cost volume construction
        B, D, H, W = depth_hypo.shape
        C = ref_fea.shape[1]
        view_weights_list = []

        similarity_sum = 0
        pixel_wise_weight_sum = 1e-8

        ref_volume = ref_fea.view(B, self.G, C // self.G, 1, H, W)
        for i, (src_fea, src_proj) in enumerate(zip(src_feas, src_projs)):
            # [B, C, D, H, W] -> [B, G, C/G, D, H, W]
            warped_volume = homo_warping(src_fea, src_proj, ref_proj, depth_hypo).view(B, self.G, C // self.G, D, H, W)
            similarity = (warped_volume * ref_volume).mean(2)  # B G D H W
            if view_weights is None:
                view_weight = self.pixel_wise_net(similarity)
                view_weights_list.append(view_weight)
            else:
                view_weight = view_weights[:, i].unsqueeze(1)

            if self.training:
                similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1)  # [B, 1, D, H, W]
                pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1)  # [B, 1, 1, H, W]
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                similarity_sum += similarity * view_weight.unsqueeze(1)
                pixel_wise_weight_sum += view_weight.unsqueeze(1)

            del warped_volume

        similarity = similarity_sum.div_(pixel_wise_weight_sum)
        prob_volume_pre = self.reg_net(similarity).squeeze(1)
        prob_volume = torch.exp(F.log_softmax(prob_volume_pre, dim=1))

        if view_weights is None:
            view_weights = torch.cat(view_weights_list, dim=1)  # [B,N-1,H,W]

        if not wta:
            depth = depth_regression(prob_volume, depth_hypo)
            # conf = conf_regression(prob_volume, 4)
            conf = None
        else:
            depth, conf = depth_wta(prob_volume, depth_hypo)

        output_stage = {
            "depth": depth,
            "photometric_confidence": conf,
            "view_weights": view_weights,
        }
        return output_stage
