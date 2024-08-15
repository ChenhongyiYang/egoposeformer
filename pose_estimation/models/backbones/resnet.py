# Author: Chenhongyi Yang
# Reference: https://github.com/hiroyasuakada/UnrealEgo

import torch
import torch.nn as nn
import torchvision


class ResNetTorchvision(nn.Module):
    def __init__(
        self,
        model_name,
        use_imagenet_pretrain,
        out_stride,
    ):
        super().__init__()
        backbone = self.build_resnet(model_name, use_imagenet_pretrain)

        base_layers = list(backbone.children())
        self.layer_s2 = nn.Sequential(*base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer_s4 = nn.Sequential(*base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer_s8 = base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer_s16 = base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer_s32 = base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

        self.out_stride = out_stride

    def get_backbone_out_channel(self, model_name):
        if model_name == "resnet18":
            return 512
        elif model_name == "resnet34":
            return 512
        elif model_name == "resnet50":
            return 2048
        elif model_name == "resnet101":
            return 2048
        else:
            raise NotImplementedError("model type [%s] is invalid", model_name)

    def build_resnet(self, model_name, use_pretrain):
        if model_name == "resnet18":
            return torchvision.models.resnet18(pretrained=use_pretrain)
        elif model_name == "resnet34":
            return torchvision.models.resnet34(pretrained=use_pretrain)
        elif model_name == "resnet50":
            return torchvision.models.resnet50(pretrained=use_pretrain)
        elif model_name == "resnet101":
            return torchvision.models.resnet101(pretrained=use_pretrain)
        else:
            raise NotImplementedError("model type [%s] is invalid", model_name)

    def forward(self, x):
        if len(x.shape) == 4:
            B, V, H, W = x.shape
            x = x.reshape(B * V, 1, H, W).repeat(1, 3, 1, 1)
        elif len(x.shape) == 5:
            B, V, C, H, W = x.shape
            x = x.reshape(B * V, C, H, W)

        out_s2 = self.layer_s2(x)
        out_s4 = self.layer_s4(out_s2)
        out_s8 = self.layer_s8(out_s4)
        out_s16 = self.layer_s16(out_s8)
        out_s32 = self.layer_s32(out_s16)

        out_s2 = out_s2.reshape(B, V, *out_s2.shape[1:])
        out_s4 = out_s4.reshape(B, V, *out_s4.shape[1:])
        out_s8 = out_s8.reshape(B, V, *out_s8.shape[1:])
        out_s16 = out_s16.reshape(B, V, *out_s16.shape[1:])
        out_s32 = out_s32.reshape(B, V, *out_s32.shape[1:])

        if self.out_stride == 4:
            output = [out_s4, out_s8, out_s16, out_s32]
        elif self.out_stride == 8:
            output = [out_s8, out_s16, out_s32]
        elif self.out_stride == 16:
            output = [out_s16, out_s32]
        elif self.out_stride == 32:
            output = [out_s32]
        else:
            raise NotImplementedError

        return output


class EfficientFPN(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        with_relu=True,
    ):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.with_relu = with_relu

        self.updample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.lateral_convs = nn.ModuleList()
        self.fuse_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(len(in_channels)):
            l_conv = []
            l_conv.append(nn.Conv2d(in_channels[i], out_channels, 1))
            if self.with_relu:
                l_conv.append(nn.ReLU(inplace=False))
            self.lateral_convs.append(nn.Sequential(*l_conv))

            if i != 0:
                fuse_conv = []
                fuse_conv.append(
                    nn.Conv2d(out_channels * 2, out_channels, 1, padding=0, stride=1)
                )
                if self.with_relu:
                    fuse_conv.append(nn.ReLU(inplace=False))
                self.fuse_convs.append(nn.Sequential(*fuse_conv))

                fpn_conv = []
                fpn_conv.append(
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, stride=1)
                )
                if self.with_relu:
                    fpn_conv.append(nn.ReLU(inplace=False))
                self.fpn_convs.append(nn.Sequential(*fpn_conv))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        B, V = inputs[0].shape[:2]
        inputs = [x.flatten(start_dim=0, end_dim=1) for x in inputs]

        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = self.fpn_convs[i-1](
                self.fuse_convs[i - 1](
                    torch.cat((laterals[i - 1], self.updample(laterals[i])), dim=1)
                )
            )
        out = laterals[0].reshape(B, V, *laterals[0].shape[1:])
        return out


class ResnetBackbone(nn.Module):
    def __init__(self, resnet_cfg, neck_cfg):
        super().__init__()
        self.backbone = ResNetTorchvision(**resnet_cfg)
        self.neck = EfficientFPN(**neck_cfg)

    def get_output_channel(self):
        return self.neck.out_channels

    def forward(self, image):
        backbone_feats = self.backbone(image)
        x = self.neck(backbone_feats)
        return x


