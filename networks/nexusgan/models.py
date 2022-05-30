import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from math import log2, ceil
from torch.nn.utils import spectral_norm
from networks.nexusgan.blocks import *

__all__ = ["EMA", "Discriminator", "Generator"]


class EMA(nn.Module):
    def __init__(self, model: nn.Module, weight_decay: float) -> None:
        super(EMA, self).__init__()
        self.model = model
        self.weight_decay = weight_decay
        self.shadow = {}
        self.backup = {}

    def register(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    1.0 - self.weight_decay
                ) * param.data + self.weight_decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self) -> None:
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]


class Generator(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=20):
        super(Generator, self).__init__()
        self.scale = scale
        self.conv1 = SeparableConv2d(num_in_ch, num_feat, 3, 1, 1)
        # residual pixel-attention blocks
        self.rpa = nn.Sequential(
            OrderedDict(
                [
                    ("rpa{}".format(i), ResidualNexusBlock(num_feat=num_feat))
                    for i in range(num_block)
                ]
            )
        )
        # up-sampling blocks with pixel-attention
        num_usblock = ceil(log2(scale))
        self.us = nn.Sequential(
            OrderedDict(
                [
                    ("us{}".format(i), UpSamplingNexusBlock(num_feat=num_feat, scale=2))
                    for i in range(num_usblock)
                ]
            )
        )
        self.conv2 = SeparableConv2d(num_feat, num_feat // 2, 3, 1, 1)
        self.conv3 = SeparableConv2d(num_feat // 2, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        z = self.conv1(x)
        z = self.lrelu(z)
        z_ = self.rpa(z)
        z = z + z_
        z = self.us(z)
        z = self.conv2(z)
        z = self.lrelu(z)
        out = self.conv3(z)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64):
        super(Discriminator, self).__init__()

        self.conv0 = nn.Conv2d(
            num_in_ch, num_feat, kernel_size=3, stride=1, padding=1
        )

        self.conv1 = spectral_norm(
            nn.Conv2d(num_feat, num_feat * 2, 3, 2, 1, bias=False)
        )
        self.conv2 = spectral_norm(
            nn.Conv2d(num_feat * 2, num_feat * 4, 3, 2, 1, bias=False)
        )

        # Center
        self.conv3 = spectral_norm(
            nn.Conv2d(num_feat * 4, num_feat * 8, 3, 2, 1, bias=False)
        )

        self.gating = spectral_norm(
            nn.Conv2d(num_feat * 8, num_feat * 4, 1, 1, 1, bias=False)
        )

        # attention Blocks
        self.attn_1 = AttentionNexusBlock(
            x_channels=num_feat * 4, g_channels=num_feat * 4
        )
        self.attn_2 = AttentionNexusBlock(
            x_channels=num_feat * 2, g_channels=num_feat * 4
        )
        self.attn_3 = AttentionNexusBlock(x_channels=num_feat, g_channels=num_feat * 4)

        # Cat
        self.cat_1 = ConcatenationNexusBlock(dim_in=num_feat * 8, dim_out=num_feat * 4)
        self.cat_2 = ConcatenationNexusBlock(dim_in=num_feat * 4, dim_out=num_feat * 2)
        self.cat_3 = ConcatenationNexusBlock(dim_in=num_feat * 2, dim_out=num_feat)

        # upsample
        self.conv4 = spectral_norm(
            nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False)
        )
        self.conv5 = spectral_norm(
            nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False)
        )
        self.conv6 = spectral_norm(
            nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False)
        )

        # extra
        self.conv7 = spectral_norm(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)
        )
        self.conv8 = spectral_norm(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False)
        )
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

        self._initialize_weights()


    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="leaky_relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2)

        gated = F.leaky_relu(self.gating(x3), negative_slope=0.2)

        # Attention
        attn1 = self.attn_1(x2, gated)
        attn2 = self.attn_2(x1, gated)
        attn3 = self.attn_3(x0, gated)

        # upsample
        x3 = self.cat_1(attn1, x3)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2)
        x4 = self.cat_2(attn2, x4)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2)
        x5 = self.cat_3(attn3, x5)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2)

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2)
        out = self.conv9(out)

        return out
