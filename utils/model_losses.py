from torch import nn

from typing import Union
import config
from networks.nexusgan.losses import ContentLoss, GANLoss


def define_nexusnet_loss() -> nn.L1Loss:
    pixel_criterion = nn.L1Loss()
    pixel_criterion = pixel_criterion.to(device=config.device, non_blocking=True)

    return pixel_criterion


def define_nexusgan_loss() -> Union[nn.L1Loss, ContentLoss, GANLoss]:
    pixel_criterion = nn.L1Loss()
    content_criterion = ContentLoss(
        config.feature_model_extractor_nodes,
        config.feature_model_normalize_mean,
        config.feature_model_normalize_std,
    )
    adversarial_criterion = GANLoss()

    # Transfer to CUDA
    pixel_criterion = pixel_criterion.to(device=config.device)
    content_criterion = content_criterion.to(device=config.device)
    adversarial_criterion = adversarial_criterion.to(device=config.device)

    return pixel_criterion, content_criterion, adversarial_criterion
