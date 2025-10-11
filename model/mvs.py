from model.modules import *
from model.submodules import *


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
        _, D, H, W = depth_hypo.shape
        B, C = ref_fea.shape[0], ref_fea.shape[1]
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
            conf = conf_regression(prob_volume, 4)
            conf = None
        else:
            depth, conf = depth_wta(prob_volume, depth_hypo)

        output_stage = {
            "depth": depth,
            "photometric_confidence": conf,
            "view_weights": view_weights,
        }
        return output_stage
