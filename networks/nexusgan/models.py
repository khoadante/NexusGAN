from tokenize import group
from torch import nn as nn
from math import log2, ceil
from collections import OrderedDict
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
from networks.nexusgan.blocks import RPA, US, add_attn, unetCat


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
        self.sep_conv1_ = nn.Conv2d(num_in_ch, num_in_ch, 3, 1, 1, groups=num_in_ch)
        self.sep_conv1 = nn.Conv2d(num_in_ch, num_feat, 1)
        # residual pixel-attention blocks
        self.rpa = nn.Sequential(
            OrderedDict(
                [("rpa{}".format(i), RPA(num_feat=num_feat)) for i in range(num_block)]
            )
        )
        # up-sampling blocks with pixel-attention
        num_usblock = ceil(log2(scale))
        self.us = nn.Sequential(
            OrderedDict(
                [
                    ("us{}".format(i), US(num_feat=num_feat, scale=2))
                    for i in range(num_usblock)
                ]
            )
        )
        self.sep_conv2_ = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=num_feat)
        self.sep_conv2 = nn.Conv2d(num_feat, num_feat // 2, 1)
        self.sep_conv3_ = nn.Conv2d(
            num_feat // 2, num_feat // 2, 3, 1, 1, groups=num_feat // 2
        )
        self.sep_conv3 = nn.Conv2d(num_feat // 2, num_out_ch, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x):
        z = self.sep_conv1(self.sep_conv1_(x))
        z = self.lrelu(z)
        z_ = self.rpa(z)
        z = z + z_
        z = self.us(z)
        z = self.sep_conv2(self.sep_conv2_(z))
        z = self.lrelu(z)
        out = self.sep_conv3(self.sep_conv3_(z))
        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


class Discriminator(nn.Module):
    def __init__(self, num_in_ch=3, num_feat=64):
        super(Discriminator, self).__init__()
        norm = spectral_norm

        self.sep_conv0_ = nn.Conv2d(num_in_ch, num_in_ch, 3, 1, 1, groups=num_in_ch)
        self.sep_conv0 = nn.Conv2d(num_in_ch, num_feat, 1)

        self.sep_conv1_ = nn.Conv2d(
            num_feat, num_feat, 3, 2, 1, groups=num_feat, bias=False
        )
        self.sep_conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 1, bias=False))

        self.sep_conv2_ = nn.Conv2d(
            num_feat * 2, num_feat * 2, 3, 2, 1, groups=num_feat * 2, bias=False
        )
        self.sep_conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 1, bias=False))

        # Center
        self.sep_conv3_ = nn.Conv2d(
            num_feat * 4, num_feat * 4, 3, 2, 1, groups=num_feat * 4, bias=False
        )
        self.sep_conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 1, bias=False))

        self.gating = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 1, 1, 1, bias=False))

        # attention Blocks
        self.attn_1 = add_attn(x_channels=num_feat * 4, g_channels=num_feat * 4)
        self.attn_2 = add_attn(x_channels=num_feat * 2, g_channels=num_feat * 4)
        self.attn_3 = add_attn(x_channels=num_feat, g_channels=num_feat * 4)

        # Cat
        self.cat_1 = unetCat(dim_in=num_feat * 8, dim_out=num_feat * 4)
        self.cat_2 = unetCat(dim_in=num_feat * 4, dim_out=num_feat * 2)
        self.cat_3 = unetCat(dim_in=num_feat * 2, dim_out=num_feat)

        # upsample
        self.sep_conv4_ = nn.Conv2d(
            num_feat * 8, num_feat * 8, 3, 1, 1, groups=num_feat * 8, bias=False
        )
        self.sep_conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 1, bias=False))
        self.sep_conv5_ = nn.Conv2d(
            num_feat * 4, num_feat * 4, 3, 1, 1, groups=num_feat * 4, bias=False
        )
        self.sep_conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 1, bias=False))
        self.sep_conv6_ = nn.Conv2d(
            num_feat * 2, num_feat * 2, 3, 1, 1, groups=num_feat * 2, bias=False
        )
        self.sep_conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 1, bias=False))

        # extra
        self.sep_conv7_ = nn.Conv2d(
            num_feat, num_feat, 3, 1, 1, groups=num_feat, bias=False
        )
        self.sep_conv7 = norm(nn.Conv2d(num_feat, num_feat, 1, bias=False))
        self.sep_conv8_ = nn.Conv2d(
            num_feat, num_feat, 3, 1, 1, groups=num_feat, bias=False
        )
        self.sep_conv8 = norm(nn.Conv2d(num_feat, num_feat, 1, bias=False))
        self.sep_conv9_ = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=num_feat)
        self.sep_conv9 = nn.Conv2d(num_feat, 1, 1)

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x):
        x0 = F.leaky_relu(self.sep_conv0(self.sep_conv0_(x)), negative_slope=0.2)
        x1 = F.leaky_relu(self.sep_conv1(self.sep_conv1_(x0)), negative_slope=0.2)
        x2 = F.leaky_relu(self.sep_conv2(self.sep_conv2_(x1)), negative_slope=0.2)
        x3 = F.leaky_relu(self.sep_conv3(self.sep_conv3_(x2)), negative_slope=0.2)

        gated = F.leaky_relu(self.gating(x3), negative_slope=0.2)

        # Attention
        attn1 = self.attn_1(x2, gated)
        attn2 = self.attn_2(x1, gated)
        attn3 = self.attn_3(x0, gated)

        # upsample
        x3 = self.cat_1(attn1, x3)
        x4 = F.leaky_relu(self.sep_conv4(self.sep_conv4_(x3)), negative_slope=0.2)
        x4 = self.cat_2(attn2, x4)
        x5 = F.leaky_relu(self.sep_conv5(self.sep_conv5_(x4)), negative_slope=0.2)
        x5 = self.cat_3(attn3, x5)
        x6 = F.leaky_relu(self.sep_conv6(self.sep_conv6_(x5)), negative_slope=0.2)

        # extra
        out = F.leaky_relu(self.sep_conv7(self.sep_conv7_(x6)), negative_slope=0.2)
        out = F.leaky_relu(self.sep_conv8(self.sep_conv8_(out)), negative_slope=0.2)
        out = self.sep_conv9(self.sep_conv9_(out))

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="leaky_relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
