import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

__all__ = [
    "SeparableConv2d",
    "ResidualNexusBlock",
    "UpSamplingNexusBlock",
    "AttentionNexusBlock",
    "ConcatenationNexusBlock",
]


def _initialize_weights(layers):
    for layer in layers:
        nn.init.kaiming_normal_(layer.weight)
        layer.weight.data *= 0.1
        if layer.bias is not None:
            nn.init.constant_(layer.bias, 0)


class SeparableConv2d(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride=1, padding=1, bias=True
    ):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            groups=in_channels,
            bias=bias,
            padding=padding,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)

        _initialize_weights([self.depthwise, self.pointwise])

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

    


class ResidualNexusBlock(nn.Module):
    def __init__(self, num_feat):
        super(ResidualNexusBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat * 2, kernel_size=1)
        self.conv2 = nn.Conv2d(num_feat * 2, num_feat * 4, kernel_size=1)
        self.conv3 = SeparableConv2d(
            num_feat * 4, num_feat, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = SeparableConv2d(
            num_feat, num_feat, kernel_size=3, stride=1, padding=1
        )
        self.sigmoid = nn.Sigmoid()
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        _initialize_weights([self.conv1, self.conv2])

    def forward(self, x):
        z = self.conv1(x)
        z = self.lrelu(z)
        z = self.conv2(z)
        z = self.lrelu(z)
        z = self.conv3(z)
        z = self.sigmoid(z)
        z = x * z + x
        z = self.conv4(z)
        out = self.lrelu(z)
        return out


class UpSamplingNexusBlock(nn.Module):
    def __init__(self, num_feat, scale):
        super(UpSamplingNexusBlock, self).__init__()
        self.scale = scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, kernel_size=1)
        self.pa_conv = nn.Conv2d(num_feat, num_feat, kernel_size=1)
        self.pa_sigmoid = nn.Sigmoid()
        self.conv2 = SeparableConv2d(
            num_feat, num_feat, kernel_size=3, stride=1, padding=1
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.2)

        _initialize_weights([self.conv1, self.pa_conv])

    def forward(self, x):
        x_ = self.conv1(F.interpolate(x, scale_factor=self.scale, mode="nearest"))
        x_ = self.lrelu(x_)
        z = self.pa_conv(x_)
        z = self.pa_sigmoid(z)
        z = torch.mul(x_, z) + x_
        z = self.conv2(z)
        out = self.lrelu(z)
        return out


class AttentionNexusBlock(nn.Module):
    def __init__(self, x_channels, g_channels=256):
        super(AttentionNexusBlock, self).__init__()
        self.conv = nn.Conv2d(
            x_channels, x_channels, kernel_size=1, stride=1, padding=0, bias=False
        )

        self.W = nn.Sequential(
            self.conv,
            nn.BatchNorm2d(x_channels),
        )
        self.theta = nn.Conv2d(
            x_channels, x_channels, kernel_size=2, stride=2, padding=0, bias=False
        )

        self.phi = nn.Conv2d(
            g_channels, x_channels, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.psi = nn.Conv2d(
            x_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True
        )

        _initialize_weights([self.conv, self.theta, self.phi, self.psi])

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(
            self.phi(g), size=theta_x_size[2:], mode="bilinear", align_corners=False
        )
        f = F.relu(theta_x + phi_g)

        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(
            sigm_psi_f, size=input_size[2:], mode="bilinear", align_corners=False
        )

        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y


class ConcatenationNexusBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatenationNexusBlock, self).__init__()
        self.convU = spectral_norm(
            nn.Conv2d(dim_in, dim_out, 3, 1, 1, bias=False)
        )

    def forward(self, input_1, input_2):
        # Upsampling
        input_2 = F.interpolate(
            input_2, scale_factor=2, mode="bilinear", align_corners=False
        )

        output_2 = F.leaky_relu(self.convU(input_2), negative_slope=0.2)

        offset = output_2.size()[2] - input_1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        output_1 = F.pad(input_1, padding)
        y = torch.cat([output_1, output_2], 1)
        return y
