from torch import nn
from torch import optim

from typing import List
import config


def define_nexusnet_optimizer(model) -> optim.AdamW:
    optimizer = optim.AdamW(model.parameters(), config.model_lr, config.model_betas)

    return optimizer


def define_nexusgan_optimizer(
    discriminator: nn.Module, generator: nn.Module
) -> List[optim.AdamW]:
    d_optimizer = optim.AdamW(
        discriminator.parameters(), config.model_lr, config.model_betas
    )
    g_optimizer = optim.AdamW(
        generator.parameters(), config.model_lr, config.model_betas
    )

    return d_optimizer, g_optimizer
