from tokenize import group
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.utils import spectral_norm


class US(nn.Module):
    def __init__(self, num_feat, scale):
        super(US, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 1)
        # plugin pixel attention
        self.pa_conv = nn.Conv2d(num_feat, num_feat, 1)
        self.pa_sigmoid = nn.Sigmoid()
        # separable conv
        self.sep_conv1_ = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=num_feat)
        self.sep_conv1 = nn.Conv2d(num_feat, num_feat, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        x_ = self.conv1(F.interpolate(x, scale_factor=self.scale, mode="nearest"))
        x_ = self.lrelu(x_)
        z = self.pa_conv(x_)
        z = self.pa_sigmoid(z)
        z = torch.mul(x_, z) + x_
        z = self.sep_conv1_(self.sep_conv1(z))
        out = self.lrelu(z)
        return out


class RPA(nn.Module):
    def __init__(self, num_feat):
        super(RPA, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat * 2, 1)
        self.conv2 = nn.Conv2d(num_feat * 2, num_feat * 4, 1)
        self.sep_conv1_ = nn.Conv2d(
            num_feat * 4, num_feat * 4, 3, 1, 1, groups=num_feat * 4
        )
        self.sep_conv1 = nn.Conv2d(num_feat * 4, num_feat, 1)
        self.sep_conv2_ = nn.Conv2d(num_feat, num_feat, 3, 1, 1, groups=num_feat)
        self.sep_conv2 = nn.Conv2d(num_feat, num_feat, 1)
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):
        z = self.conv1(x)
        z = self.lrelu(z)
        z = self.conv2(z)
        z = self.lrelu(z)
        z = self.sep_conv1(self.sep_conv1_(z))
        z = self.sigmoid(z)
        z = x * z + x
        z = self.sep_conv2(self.sep_conv2_(z))
        out = self.lrelu(z)
        return out


class add_attn(nn.Module):
    def __init__(self, x_channels, g_channels=256):
        super(add_attn, self).__init__()
        self.W = nn.Sequential(
            nn.Conv2d(x_channels, x_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(x_channels),
        )
        self.theta_ = nn.Conv2d(
            x_channels, x_channels, 2, 2, 0, groups=x_channels, bias=False
        )
        self.theta = nn.Conv2d(x_channels, x_channels, 1, bias=False)

        self.phi = nn.Conv2d(g_channels, x_channels, 1, 1, 0)
        self.psi = nn.Conv2d(x_channels, 1, 1, 1, 0)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(self.theta_(x))
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(
            self.phi(g), size=theta_x_size[2:], mode="bilinear", align_corners=False
        )
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(
            sigm_psi_f, size=input_size[2:], mode="bilinear", align_corners=False
        )

        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y


class unetCat(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(unetCat, self).__init__()
        norm = spectral_norm
        self.sep_conv_ = nn.Conv2d(dim_in, dim_in, 3, 1, 1, groups=dim_in, bias=False)
        self.sep_conv = norm(nn.Conv2d(dim_in, dim_out, 1, bias=False))

    def forward(self, input_1, input_2):
        # Upsampling
        input_2 = F.interpolate(
            input_2, scale_factor=2, mode="bilinear", align_corners=False
        )

        output_2 = F.leaky_relu(
            self.sep_conv(self.sep_conv_(input_2)), negative_slope=0.2
        )

        offset = output_2.size()[2] - input_1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        output_1 = F.pad(input_1, padding)
        y = torch.cat([output_1, output_2], 1)
        return y
