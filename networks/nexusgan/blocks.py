import torch
from torch import nn

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
