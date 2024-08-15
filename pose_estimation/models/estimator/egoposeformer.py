# Author: Chenhongyi Yang

import copy
from collections import OrderedDict

import torch
import torch.nn as nn

from pose_estimation.models.backbones.resnet import ResnetBackbone
from pose_estimation.models.utils.deform_attn import MSDeformAttn
from pose_estimation.models.utils.transformer import CustomMultiheadAttention, FFN
from pose_estimation.models.utils.camera_models import projection_funcs
from pose_estimation.models.utils.pose_metric import (
    MpjpeLoss,
    batch_compute_similarity_transform_numpy
)

INF = 1e10
EPS = 1e-6


class EgoPoseFormer(nn.Module):
    def __init__(
        self,
        input_dims,
        embed_dims,
        encoder_cfg,
        mlp_dims,
        mlp_dropout,
        num_mlp_layers,
        transformer_cfg,
        num_former_layers,
        num_pred_mlp_layers,
        image_size,
        camera_model,
        feat_down_stride,
        coor_norm_max,
        coor_norm_min,
        norm_mlp_pred=False,
        num_joints=16,
        num_views=2,
        to_mm=10.0,
        encoder_pretrained=None,
        train_cfg=dict(w_mpjpe=1.0),
        **kwargs
    ):
        super(EgoPoseFormer, self).__init__(**kwargs)

        self.invalid_pad = INF

        self.num_joints = num_joints
        self.num_views = num_views
        self.embed_dims = embed_dims
        self.to_mm = to_mm

        self.feat_down_stride = feat_down_stride
        self.feat_shape = (
            image_size[0] // feat_down_stride,
            image_size[1] // feat_down_stride,
        )
        self.image_size = image_size
        # ------------------------------------
        self.encoder = ResnetBackbone(**encoder_cfg)

        # Transform channel number
        self.feat_proj = nn.Conv2d(input_dims, embed_dims, 1, 1, 0)

        # Transformer Layer
        self.layers = nn.ModuleList()
        for idx in range(num_former_layers):
            _cfg = copy.deepcopy(transformer_cfg)
            _cfg.update({
                "num_views": num_views,
                "embed_dims": embed_dims,
                "feat_shape": self.feat_shape,
            })
            self.layers.append(EgoPoseFormerTransformerLayer(**_cfg))

        self.query_gen_mlp = nn.Sequential(
            nn.Linear(4, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims)
        )

        mlp_layers = []
        in_dims = embed_dims * num_views
        for _ in range(num_mlp_layers):
            mlp_layers.append(
                nn.Sequential(
                    nn.Linear(in_dims, mlp_dims),
                    nn.GELU(),
                    nn.Dropout(mlp_dropout),
                )
            )
            in_dims = mlp_dims
        mlp_layers.append(nn.Linear(in_dims, 3 * self.num_joints))
        self.mlp_pred = nn.Sequential(*mlp_layers)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.reg_mlp = torch.nn.ModuleList()
        for _ in range(num_former_layers):
            reg_mlp = []
            for i in range(num_pred_mlp_layers - 1):
                reg_mlp.append(nn.Linear(embed_dims, embed_dims))
                reg_mlp.append(nn.GELU())
            reg_mlp.append(nn.Linear(embed_dims, 3))
            self.reg_mlp.append(nn.Sequential(*reg_mlp))
        self.post_norm = torch.nn.ModuleList(
            [nn.LayerNorm(embed_dims) for _ in range(num_former_layers)]
        )

        self.norm_mlp_pred = norm_mlp_pred
        if norm_mlp_pred:
            self.register_buffer("coor_min", torch.tensor(coor_norm_min))
            self.register_buffer("coor_max", torch.tensor(coor_norm_max))

        self.camera_model = camera_model
        self._local_to_image = projection_funcs.get(camera_model)

        self.train_cfg = train_cfg
        self.load_pretrain(encoder_pretrained)
        self.criteria = MpjpeLoss()

    def _unnorm_coor(self, coor, norm_range=(-1.0, 1.0)):
        norm_gap = (norm_range[1] - norm_range[0])
        unnormed_coor = (self.coor_max - self.coor_min) * \
                        (coor - norm_range[0]) / norm_gap + self.coor_min
        return unnormed_coor

    def _norm_coor(self, coor, norm_range=(-1.0, 1.0)):
        normed_coor = (coor - self.coor_min) / (
            self.coor_max - self.coor_min
        )
        norm_gap = (norm_range[1] - norm_range[0])
        normed_coor = norm_gap * normed_coor - norm_gap * 0.5
        return normed_coor

    def _forward_mlp(self, frame_feats):
        B, V, C = frame_feats.shape[:3]

        x = frame_feats.flatten(start_dim=0, end_dim=1)
        x = self.avg_pool(x).reshape(B, V * C)
        mlp_pred = self.mlp_pred(x).reshape(B, self.num_joints, 3)
        if self.norm_mlp_pred:
            self._unnorm_coor(mlp_pred)
        return mlp_pred

    def _forward_transformer(
        self,
        image_feats,
        origin_3d,
        init_anchors_3d,
    ):
        # image_feats [B, V, C, H, W]
        # image_pe [B, V, H * W, C]
        # init_anchors_3d [B, J, 3]

        B, V, C, H, W = image_feats.shape
        J = self.num_joints

        img_feats = image_feats.permute(0, 1, 3, 4, 2).reshape(B, V, H * W, C)
        anchors_2d, anchors_valid = self._local_to_image(init_anchors_3d, origin_3d)
        anchors_2d = anchors_2d.to(dtype=img_feats.dtype)

        joint_inds = (
            torch.arange(1, J+1)
            .to(dtype=img_feats.dtype, device=img_feats.device)
            .reshape(1, J, 1)
            .repeat(B, 1, 1)
        ) / float(J)
        x = self.query_gen_mlp(torch.cat((joint_inds, init_anchors_3d), dim=-1))

        preds = []
        for idx in range(len(self.layers)):
            x = self.layers[idx](
                x,
                img_feats,
                anchors_2d,
                anchors_valid,
            )
            _x = self.post_norm[idx](x)
            offset_pred = self.reg_mlp[idx](_x)
            pred = offset_pred + init_anchors_3d.detach()
            preds.append(pred)  # [B, J, C]
        return preds

    def _forward(self, img, origin_3d, gt):
        B, V, img_c, img_h, img_w = img.shape

        img = img.reshape(B, V, img_c, img_h, img_w)
        frame_feats = self.encoder(img)  # [B, V, C, h, w]
        frame_feats = self.feat_proj(frame_feats.reshape(B * V, *frame_feats.shape[-3:]))
        frame_feats = frame_feats.reshape(B, V, *frame_feats.shape[-3:])

        mlp_pred_3d = self._forward_mlp(frame_feats)  # [B, J, 3]

        init_anchors_3d = mlp_pred_3d.clone().detach()  # [B, J, 3]
        attn_pred_coor3ds = self._forward_transformer(
            frame_feats,
            origin_3d,
            init_anchors_3d,
        )
        pred_3d_all = []
        pred_3d_all.append(mlp_pred_3d)
        pred_3d_all = pred_3d_all + attn_pred_coor3ds
        return pred_3d_all

    def load_pretrain(self, pretrain):
        if pretrain is None:
            return
        ckpt = torch.load(pretrain, map_location="cpu")
        self.load_state_dict(ckpt["state_dict"], strict=False)

    def get_loss(self, pred_pose, gt_pose):
        mpjpe_loss = self.criteria(pred_pose, gt_pose) * self.train_cfg.get("w_mpjpe", 1.0)
        return dict(mpjpe_loss=mpjpe_loss)

    def forward(self, img, origin_3d, gt_pose, **kwargs):
        if self.training:
            return self.forward_train(img, origin_3d, gt_pose, **kwargs)
        else:
            return self.forward_test(img, origin_3d, gt_pose, **kwargs)

    def forward_train(self, img, origin_3d, gt_pose, **kwargs):
        pred_pose_all = self._forward(img, origin_3d, gt=gt_pose)

        losses = OrderedDict()
        for i, pred_pose in enumerate(pred_pose_all):
            losses_i = self.get_loss(pred_pose, gt_pose)
            for k, v in losses_i.items():
                losses["%s_%d"%(k, i)] = v
        return losses

    def forward_test(self, img, origin_3d, gt_pose=None, **kwargs):
        B = img.shape[0]
        pred_pose_all = self._forward(img, origin_3d, gt=gt_pose)
        pose_proposal = pred_pose_all[0]
        pred_final = pred_pose_all[-1]
        pose_proposal_output = pose_proposal.detach().cpu().numpy()
        pred_pose_output = pred_final.detach().cpu().numpy()

        output_dict = OrderedDict()
        output_dict["pred_pose"] = pred_pose_output
        output_dict["propose_pose"] = pose_proposal_output

        if gt_pose is not None:
            metrics_final = self.evaluate(pred_final, gt_pose, "final")
            metrics_proposal = self.evaluate(pose_proposal, gt_pose, "proposal")

            output_dict.update(metrics_final)
            output_dict.update(metrics_proposal)
        return output_dict

    def evaluate(self, pred_pose, pose_gt, prefix):
        B = pred_pose.shape[0]

        S1_hat = batch_compute_similarity_transform_numpy(pred_pose, pose_gt.to(dtype=torch.float))

        error = torch.linalg.norm(pred_pose - pose_gt, dim=-1, ord=2) * self.to_mm
        pa_error = torch.linalg.norm(S1_hat - pose_gt, dim=-1, ord=2) * self.to_mm

        mpjpe = error.mean(dim=1).reshape(B,).detach().cpu().numpy()
        pa_mpjpe = pa_error.mean(dim=1).reshape(B,).detach().cpu().numpy()

        metrics = OrderedDict()
        metrics[prefix+"_mpjpe"] = mpjpe
        metrics[prefix+"_pa_mpjpe"] = pa_mpjpe
        return metrics


class DeformStereoAttn(MSDeformAttn):
    def __init__(self, feat_shape, **kwargs):
        _init_cfg = {
            "d_model": kwargs.pop("embed_dim"),
            "n_heads": kwargs.pop("num_heads"),
            "n_points": 16
        }
        super(DeformStereoAttn, self).__init__(**_init_cfg)

        self.register_buffer("spatial_shapes", torch.tensor([[feat_shape[0], feat_shape[1]]], dtype=torch.long))
        self.register_buffer("start_index", torch.tensor([0,], dtype=torch.long))

    def forward(self, query, img_feat, anchors_2d):
        # query: [B, J, C]
        # img_feat: [B, H * W, C]
        # img_feat: [B, J, 2]

        B, J, C = query.shape

        anchors_2d = anchors_2d.detach()

        _q = query.reshape(B, J, C)
        _kv = img_feat.reshape(B, -1, C)
        _ref_pts = anchors_2d.reshape(B, J, 1, 2)

        out = super(DeformStereoAttn, self).forward(_q, _ref_pts, _kv, self.spatial_shapes, self.start_index)
        out = out.reshape(B, J, C)
        return out

class EgoformerSpatialMHA(CustomMultiheadAttention):
    def forward(self, q, k, v, bias):
        # q, k, v: [B, J, C]

        B, J, C = q.shape

        _q = self.q_proj(q).reshape(B, J, self.num_heads, self.head_dims).permute(0, 2, 1, 3)  # [B, H, J, c]
        _k = self.k_proj(k).reshape(B, J, self.num_heads, self.head_dims).permute(0, 2, 1, 3)
        _v = self.v_proj(v).reshape(B, J, self.num_heads, self.head_dims).permute(0, 2, 1, 3)

        attn = (_q @ _k.transpose(-2, -1)) * self.scale  # [B, H, J, J]
        if bias is not None:
            attn = attn + bias

        attn = attn.softmax(dim=-1)

        x = (attn @ _v).permute(0, 2, 1, 3).reshape(B, J, C)
        if self.out_proj is not None:
            x = self.out_proj(x)
        return x

class EgoPoseFormerTransformerLayer(nn.Module):
    def __init__(
        self,
        num_views,
        embed_dims,
        cross_attn_cfg,
        spatial_attn_cfg,
        ffn_cfg,
        feat_shape,
    ):
        super().__init__()
        self.embed_dims = embed_dims

        _cross_attn_cfg = copy.deepcopy(cross_attn_cfg)
        _cross_attn_cfg.update({
            "embed_dim": embed_dims,
            "feat_shape": feat_shape
        })
        self.cross_attn = DeformStereoAttn(**_cross_attn_cfg)
        self.fuse_mlp = nn.Linear(num_views * embed_dims, embed_dims)
        self.norm_cross = nn.LayerNorm(embed_dims)

        _spatial_attn_cfg = copy.deepcopy(spatial_attn_cfg)
        _spatial_attn_cfg.update({"embed_dim": embed_dims})
        self.spatial_attn = EgoformerSpatialMHA(**_spatial_attn_cfg)
        self.norm_spatial = nn.LayerNorm(embed_dims)

        _ffn_cfg = copy.deepcopy(ffn_cfg)
        _ffn_cfg.update({"embed_dims": embed_dims})
        self.ffn = FFN(**_ffn_cfg)
        self.norm_ffn = nn.LayerNorm(embed_dims)

    def _run_spatial_attn(self, joint_query):
        identity = joint_query

        q = joint_query
        k = q
        v = joint_query

        attn_res = self.spatial_attn(q, k, v, bias=None)
        x = identity + attn_res
        x = self.norm_spatial(x)
        return x


    def _run_cross_attn(
        self,
        joint_query,
        image_feats,
        anchors_2d,
        anchors_valid,
    ):
        B, V, J = image_feats.shape[:3]
        identity = joint_query

        q = joint_query
        feats_per_view = []
        for i in range(V):
            qi = q
            feat_i = image_feats[:, i]
            anchors_i = anchors_2d[:, i]
            attn_res = self.cross_attn(qi, feat_i, anchors_i)
            attn_res = attn_res.masked_fill(~anchors_valid[:, i][..., None].expand_as(attn_res), 0.0)
            feats_per_view.append(attn_res)  # [B, J, C]
        feats_all = self.fuse_mlp(torch.cat(feats_per_view, dim=-1))

        x = identity + feats_all
        x = self.norm_cross(x)
        return x

    def _forward_ffn(self, x):
        x = x + self.ffn(x)
        x = self.norm_ffn(x)
        return x

    def forward(
        self,
        joint_query,
        image_feats,
        anchors_2d,
        anchors_valid,
    ):
        x = joint_query

        x = self._run_cross_attn(x, image_feats, anchors_2d, anchors_valid)
        x = self._run_spatial_attn(x)
        x = self._forward_ffn(x)
        return x


