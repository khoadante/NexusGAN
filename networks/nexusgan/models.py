import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from networks.nexusgan.blocks import AttentionBlock, ConcatenationBlock, SeparableConv2d, ResidualResidualDenseBlock

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
    def __init__(
        self, in_channels: int = 3, out_channels: int = 3, upscale_factor: int = 4
    ) -> None:
        super(Generator, self).__init__()
        if upscale_factor == 2:
            in_channels *= 4
            downscale_factor = 2
        elif upscale_factor == 1:
            in_channels *= 16
            downscale_factor = 4
        else:
            in_channels *= 1
            downscale_factor = 1

        # Down-sampling layer
        self.downsampling = nn.PixelUnshuffle(downscale_factor)

        # The first layer of convolutional layer
        self.conv1 = SeparableConv2d(in_channels, 64, (3, 3), (1, 1), (1, 1))

        # Feature extraction backbone network
        trunk = []
        for _ in range(5):
            trunk.append(ResidualResidualDenseBlock(64, 32))
        self.trunk = nn.Sequential(*trunk)

        # After the feature extraction network, reconnect a layer of convolutional blocks
        self.conv2 = SeparableConv2d(64, 64, (3, 3), (1, 1), (1, 1))

        # Upsampling convolutional layer
        self.upsampling1 = nn.Sequential(
            SeparableConv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        self.upsampling2 = nn.Sequential(
            SeparableConv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        # Output layer
        self.conv3 = nn.Sequential(
            SeparableConv2d(64, 64, (3, 3), (1, 1), (1, 1)),
            nn.LeakyReLU(0.2, True),
        )

        # Output layer
        self.conv4 = SeparableConv2d(64, out_channels, (3, 3), (1, 1), (1, 1))

        # Initialize neural network weights
        self._initialize_weights()

    # The model should be defined in the Torch.script method.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # If upscale_factor not equal 4, must use nn.PixelUnshuffle() ops
        out = self.downsampling(x)

        out1 = self.conv1(out)
        out = self.trunk(out1)
        out2 = self.conv2(out)
        out = torch.add(out1, out2)
        out = self.upsampling1(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.upsampling2(F.interpolate(out, scale_factor=2, mode="nearest"))
        out = self.conv3(out)
        out = self.conv4(out)

        out = torch.clamp_(out, 0.0, 1.0)

        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight)
                module.weight.data *= 0.1
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)


class Discriminator(nn.Module):
    def __init__(self, in_channels: int = 3,  features=64) -> None:
        super(Discriminator, self).__init__()
        self.conv0 = SeparableConv2d(in_channels, features,
                               kernel_size=3, stride=1, padding=1)

        self.conv1 = SeparableConv2d(features, features * 2, 3, 2, 1, bias=False)
        self.conv2 = SeparableConv2d(features * 2, features * 4, 3, 2, 1, bias=False)

        # Center
        self.conv3 = SeparableConv2d(features * 4, features * 8, 3, 2, 1, bias=False)

        self.gating = SeparableConv2d(features * 8, features * 4, 1, 1, 1, bias=False)

        # attention Blocks
        self.attn_1 = AttentionBlock(x_channels=features * 4, g_channels=features * 4)
        self.attn_2 = AttentionBlock(x_channels=features * 2, g_channels=features * 4)
        self.attn_3 = AttentionBlock(x_channels=features, g_channels=features * 4)

        # Cat
        self.cat_1 = ConcatenationBlock(dim_in=features * 8, dim_out=features * 4)
        self.cat_2 = ConcatenationBlock(dim_in=features * 4, dim_out=features * 2)
        self.cat_3 = ConcatenationBlock(dim_in=features * 2, dim_out=features)

        # upsample
        self.conv4 = SeparableConv2d(features * 8, features * 4, 3, 1, 1, bias=False)
        self.conv5 = SeparableConv2d(features * 4, features * 2, 3, 1, 1, bias=False)
        self.conv6 = SeparableConv2d(features * 2, features, 3, 1, 1, bias=False)

        # extra
        self.conv7 = SeparableConv2d(features, features, 3, 1, 1, bias=False)
        self.conv8 = SeparableConv2d(features, features, 3, 1, 1, bias=False)
        self.conv9 = SeparableConv2d(features, 1, 3, 1, 1)

        # Initialize model weights.
        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

    # Support torch.script function
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        gated = F.leaky_relu(self.gating(
            x3), negative_slope=0.2, inplace=True)

        # Attention
        attn1 = self.attn_1(x2, gated)
        attn2 = self.attn_2(x1, gated)
        attn3 = self.attn_3(x0, gated)

        # upsample
        x3 = self.cat_1(attn1, x3)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)
        x4 = self.cat_2(attn2, x4)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)
        x5 = self.cat_3(attn3, x5)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        # extra
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)

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


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, num_D=2):
        super(MultiScaleDiscriminator, self).__init__()
        self.num_D = num_D

        for i in range(num_D):
            netD = Discriminator()
            setattr(self, 'layer' + str(i), netD)

        self.downsample = nn.AvgPool2d(
            4, stride=2, padding=[1, 1])

    def singleD_forward(self, model, input):
        return model(input)

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):
                input_downsampled = self.downsample(input_downsampled)
        return result