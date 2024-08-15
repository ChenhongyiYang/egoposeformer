# Author: Chenhongyi Yang

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from pose_estimation.models.backbones.resnet import ResnetBackbone


class EgoPoseFormerHeatmap(nn.Module):
    def __init__(
        self,
        encoder_cfg,
        num_heatmap,
        train_cfg=dict(w_heatmap=1.0),
        **kwargs
    ):
        super(EgoPoseFormerHeatmap, self).__init__()

        self.num_heatmap = num_heatmap
        self.train_cfg = train_cfg

        self.encoder = ResnetBackbone(**encoder_cfg)
        self.conv_heatmap = nn.Conv2d(self.encoder.get_output_channel(), num_heatmap, 1)

        self.criteria = nn.MSELoss(reduction="mean")

    def get_loss(self, pred_heatmap, gt_heatmap):
        with autocast(False):
            loss_heatmap = self.criteria(pred_heatmap, gt_heatmap)
            loss_heatmap *= self.train_cfg.get("w_heatmap", 1.0)
        return loss_heatmap

    def forward_backbone(self, img):
        B, V, img_c, img_h, img_w = img.shape
        img = img.reshape(B, V, img_c, img_h, img_w)
        feats = self.encoder(img)  # [B, V, C, h, w]
        return feats

    def forward(self, img, heatmap_gt):
        if self.training:
            return self.forward_train(img, heatmap_gt)
        else:
            return self.forward_test(img, heatmap_gt)

    def forward_train(self, img, heatmap_gt, **kwargs):
        """
        img: [B, V, C, img_h, img_w]
        heatmap_gt: [B, V, c_heatmap * 2, heat_h, heat_w]
        """

        B, V, img_c, img_h, img_w = img.shape
        heatmap_gt_list = [heatmap_gt[:, i] for i in range(V)]

        feats = self.forward_backbone(img)  # [B, V, C, h, w]
        heatmaps = self.conv_heatmap(feats.view(B * V, *feats.shape[2:]))
        heatmaps = heatmaps.view(B, V, *heatmaps.shape[1:])
        heatmap_list = [heatmaps[:, i] for i in range(V)]

        loss = sum([self.get_loss(x, y) for x, y in zip(heatmap_list, heatmap_gt_list)])
        losses = dict(heatmap_loss=loss)
        return losses

    def forward_test(self, img, heatmap_gt, **kwargs):
        with torch.no_grad():
            B, V, img_c, img_h, img_w = img.shape

            heatmap_gt_list = [heatmap_gt[:, i] for i in range(V)]

            feats = self.forward_backbone(img)  # [B, V, C, h, w]
            heatmaps = self.conv_heatmap(feats.view(B * V, *feats.shape[2:]))
            heatmaps = heatmaps.view(B, V, *heatmaps.shape[1:])
            heatmap_list = [heatmaps[:, i] for i in range(V)]

            heatmap_list = [x.reshape(B, -1) for x in heatmap_list]
            heatmap_gt_list = [y.reshape(B, -1) for y in heatmap_gt_list]

            error = sum(
                torch.abs(x-y)
                for x, y in zip(heatmap_list, heatmap_gt_list)
            )
            error = error.sum(dim=1).reshape(B, ).detach().cpu().numpy()

            pos_inds = [y > 0 for y in heatmap_gt_list]

            pos_error = [
                sum(
                    torch.abs(x[i][ind[i]] - y[i][ind[i]]).sum().detach().cpu().numpy()
                    for x, y, ind in zip(heatmap_list, heatmap_gt_list, pos_inds)
                )
                for i in range(B)
            ]
            return dict(l1_error=error, pos_l1_error=torch.tensor(pos_error, device=img.device))
