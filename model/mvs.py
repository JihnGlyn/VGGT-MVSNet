from model.modules import *
from model.submodules import *


class MVSNet(nn.Module):
    def __init__(self, G, big=True, pwn=True):
        super(MVSNet, self).__init__()
        self.G = G
        self.reg_net = CostRegNet(G, deep=True) if big else CostRegNet(G)
        self.softmax = nn.LogSoftmax(dim=1)
        self.pixel_wise_net = PixelwiseNet(self.G) if pwn else None

    def forward(self, ref_feature, src_features, ref_proj, src_projs, depth_sample, view_weights, wta=False):
        _, num_depth, height, width = depth_sample.shape
        batch, feature_channel, _, _ = ref_feature.size()
        ref_feature = F.interpolate(ref_feature, size=(height, width), mode='bilinear', align_corners=False)

        device = ref_feature.device

        # num_depth = depth_sample.size()[1]
        assert (
                len(src_features) == len(src_projs)
        ), "Evaluation: Different number of images and projection matrices"

        # Change to a tensor with value 1e-5
        pixel_wise_weight_sum = 1e-5 * torch.ones((batch, 1, 1, height, width), dtype=torch.float32, device=device)
        ref_feature = ref_feature.view(batch, self.G, feature_channel // self.G, 1, height, width)
        similarity_sum = torch.zeros((batch, self.G, num_depth, height, width), dtype=torch.float32, device=device)

        i = 0
        view_weights_list = []
        for src_feature, src_proj in zip(src_features, src_projs):
            warped_feature = homo_warping(
                src_feature, src_proj, ref_proj, depth_sample
            ).view(batch, self.G, feature_channel // self.G, num_depth, height, width)
            # group-wise correlation
            similarity = (warped_feature * ref_feature).mean(2)
            # pixel-wise view weight
            if is_empty(view_weights):
                view_weight = self.pixel_wise_net(similarity)
                view_weights_list.append(view_weight)
            else:
                # reuse the pixel-wise view weight from first iteration of Patchmatch on stage 3
                view_weight = view_weights[:, i].unsqueeze(1)  # [B,1,H,W]
                i = i + 1

            if self.training:
                similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1)
                pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1)
            else:
                similarity_sum += similarity * view_weight.unsqueeze(1)
                pixel_wise_weight_sum += view_weight.unsqueeze(1)

        # aggregated matching cost across all the source views
        similarity = similarity_sum.div_(pixel_wise_weight_sum)  # [B, G, Ndepth, H, W]
        score = self.reg_net(similarity).squeeze(1)     # [B,G,Ndepth,H,W] -> [B,D,H,W]
        # apply softmax to get probability
        score = torch.exp(self.softmax(score))

        if is_empty(view_weights):
            view_weights = torch.cat(view_weights_list, dim=1)  # [B,4,H,W], 4 is the number of source views

        depth = depth_regression(score, depth_sample)
        conf = conf_regression(score, 4)

        output_stage = {
            "depth": depth,
            "photometric_confidence": conf,
            "prob_volume": score,
            "view_weights": view_weights.detach(),
        }
        return output_stage
