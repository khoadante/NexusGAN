from torch import optim
from torch.optim.lr_scheduler import StepLR, MultiStepLR

import config
from typing import List


def define_nexusnet_scheduler(optimizer) -> StepLR:
    scheduler = StepLR(
        optimizer, config.lr_scheduler_step_size, config.lr_scheduler_gamma
    )

    return scheduler


def define_nexusgan_scheduler(
    d_optimizer: optim.Adam, g_optimizer: optim.Adam
) -> List[MultiStepLR]:
    d_scheduler = MultiStepLR(
        d_optimizer, config.lr_scheduler_milestones, config.lr_scheduler_gamma
    )
    g_scheduler = MultiStepLR(
        g_optimizer, config.lr_scheduler_milestones, config.lr_scheduler_gamma
    )

    return d_scheduler, g_scheduler
