import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

__all__ = ["ResidualDenseBlock", "ResidualResidualDenseBlock", "SeparableConv2d"]


class ResidualDenseBlock(nn.Module):
    """Achieves densely connected convolutional layers.
    `Densely Connected Convolutional Networks <https://arxiv.org/pdf/1608.06993v5.pdf>` paper.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualDenseBlock, self).__init__()
        self.conv1x1 = SeparableConv2d(
            channels + growth_channels * 0, growth_channels, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=False
        )

        self.conv1 = SeparableConv2d(
            channels + growth_channels * 0, growth_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv2 = SeparableConv2d(
            channels + growth_channels * 1, growth_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv3 = SeparableConv2d(
            channels + growth_channels * 2, growth_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv4 = SeparableConv2d(
            channels + growth_channels * 3, growth_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )
        self.conv5 = SeparableConv2d(
            channels + growth_channels * 4, channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)
        )

        self.leaky_relu = nn.LeakyReLU(0.2, True)
        self.identity = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out1 = self.leaky_relu(self.conv1(x))
        out2 = self.leaky_relu(self.conv2(torch.cat([x, out1], 1)))
        out2 = out2 + self.conv1x1(x)
        out3 = self.leaky_relu(self.conv3(torch.cat([x, out1, out2], 1)))
        out4 = self.leaky_relu(self.conv4(torch.cat([x, out1, out2, out3], 1)))
        out4 = out4 + out2
        out5 = self.identity(self.conv5(torch.cat([x, out1, out2, out3, out4], 1)))
        out = torch.mul(out5, 0.2)
        out = torch.add(out, identity)

        return out


class ResidualResidualDenseBlock(nn.Module):
    """Multi-layer residual dense convolution block.

    Args:
        channels (int): The number of channels in the input image.
        growth_channels (int): The number of channels that increase in each layer of convolution.
    """

    def __init__(self, channels: int, growth_channels: int) -> None:
        super(ResidualResidualDenseBlock, self).__init__()
        self.rdb1 = ResidualDenseBlock(channels, growth_channels)
        self.rdb2 = ResidualDenseBlock(channels, growth_channels)
        self.rdb3 = ResidualDenseBlock(channels, growth_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        out = torch.mul(out, 0.2)
        out = torch.add(out, identity)

        return out


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

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class AttentionBlock(nn.Module):
    def __init__(self, x_channels, g_channels=256):
        super(AttentionBlock, self).__init__()
        self.W = nn.Sequential(
            nn.Conv2d(x_channels,
                      x_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
            nn.BatchNorm2d(x_channels))
        self.theta = nn.Conv2d(x_channels,
                               x_channels,
                               kernel_size=2,
                               stride=2,
                               padding=0,
                               bias=False)

        self.phi = nn.Conv2d(g_channels,
                             x_channels,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)
        self.psi = nn.Conv2d(x_channels,
                             out_channels=1,
                             kernel_size=1,
                             stride=1,
                             padding=0,
                             bias=True)

    def forward(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        phi_g = F.interpolate(self.phi(g),
                              size=theta_x_size[2:],
                              mode='bilinear', align_corners=False)
        f = F.relu(theta_x + phi_g, inplace=True)

        sigm_psi_f = torch.sigmoid(self.psi(f))
        sigm_psi_f = F.interpolate(
            sigm_psi_f, size=input_size[2:], mode='bilinear', align_corners=False)

        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)
        return W_y


class ConcatenationBlock(nn.Module):
    def __init__(self, dim_in, dim_out):
        super(ConcatenationBlock, self).__init__()
        self.convU = spectral_norm(
            nn.Conv2d(dim_in, dim_out, 3, 1, 1, bias=False))

    def forward(self, input_1, input_2):
        # Upsampling
        input_2 = F.interpolate(input_2, scale_factor=2,
                                mode='bilinear', align_corners=False)

        output_2 = F.leaky_relu(self.convU(
            input_2), negative_slope=0.2, inplace=True)

        offset = output_2.size()[2] - input_1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        output_1 = F.pad(input_1, padding)
        y = torch.cat([output_1, output_2], 1)
        return y