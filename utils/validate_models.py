import time
import torch
from torch import nn
from torch.cuda import amp
from torch.utils.tensorboard import SummaryWriter
from classes.prefetchers import CUDAPrefetcher
from utils.info_meters import AverageMeter, ProgressMeter

import config


def validate_nexusnet(
    model: nn.Module,
    ema_model: nn.Module,
    data_prefetcher: CUDAPrefetcher,
    epoch: int,
    writer: SummaryWriter,
    niqe_model: nn.Module,
    mode: str,
) -> float:
    """Test main program

    Args:
        model (nn.Module): generator model in adversarial networks
        ema_model (nn.Module): Exponential Moving Average Model
        data_prefetcher (CUDAPrefetcher): test dataset iterator
        epoch (int): number of test epochs during training of the adversarial network
        writer (SummaryWriter): log file management function
        niqe_model (nn.Module): The model used to calculate the model NIQE metric
        mode (str): test validation dataset accuracy or test dataset accuracy

    """
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f")
    niqe_metrics = AverageMeter("NIQE", ":4.2f")
    progress = ProgressMeter(
        len(data_prefetcher), [batch_time, niqe_metrics], prefix=f"{mode}: "
    )

    # Restore the model before the EMA
    ema_model.apply_shadow()
    # Put the adversarial network model in validation mode
    model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            lr = batch_data["lr"].to(device=config.device, non_blocking=True)

            # Mixed precision
            with amp.autocast():
                sr = model(lr)

            # Statistical loss value for terminal data output
            niqe = niqe_model(sr)
            niqe_metrics.update(niqe.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % (batches // 5) == 0:
                progress.display(batch_index)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # Restoring the EMA model
    ema_model.restore()

    # Print average PSNR metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/NIQE", niqe_metrics.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return niqe_metrics.avg


def validate_nexusgan(
    model: nn.Module,
    ema_model: nn.Module,
    data_prefetcher: CUDAPrefetcher,
    epoch: int,
    writer: SummaryWriter,
    niqe_model: nn.Module,
    mode: str,
) -> float:
    """Test main program

    Args:
        model (nn.Module): generator model in adversarial networks
        ema_model (nn.Module): Exponential Moving Average Model
        data_prefetcher (CUDAPrefetcher): test dataset iterator
        epoch (int): number of test epochs during training of the adversarial network
        writer (SummaryWriter): log file management function
        niqe_model (nn.Module): The model used to calculate the model NIQE metric
        mode (str): test validation dataset accuracy or test dataset accuracy

    """
    # Calculate how many batches of data are in each Epoch
    batches = len(data_prefetcher)
    batch_time = AverageMeter("Time", ":6.3f")
    niqe_metrics = AverageMeter("NIQE", ":4.2f")
    progress = ProgressMeter(
        len(data_prefetcher), [batch_time, niqe_metrics], prefix=f"{mode}: "
    )

    # Restore the model before the EMA
    ema_model.apply_shadow()
    # Put the adversarial network model in validation mode
    model.eval()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    data_prefetcher.reset()
    batch_data = data_prefetcher.next()

    # Get the initialization test time
    end = time.time()

    with torch.no_grad():
        while batch_data is not None:
            lr = batch_data["lr"].to(device=config.device, non_blocking=True)

            # Mixed precision
            with amp.autocast():
                sr = model(lr)

            # Statistical loss value for terminal data output
            niqe = niqe_model(sr)
            niqe_metrics.update(niqe.item(), lr.size(0))

            # Calculate the time it takes to fully test a batch of data
            batch_time.update(time.time() - end)
            end = time.time()

            # Record training log information
            if batch_index % (batches // 5) == 0:
                progress.display(batch_index)

            # Preload the next batch of data
            batch_data = data_prefetcher.next()

            # After training a batch of data, add 1 to the number of data batches to ensure that the
            # terminal print data normally
            batch_index += 1

    # Restoring the EMA model
    ema_model.restore()

    # Print average PSNR metrics
    progress.display_summary()

    if mode == "Valid" or mode == "Test":
        writer.add_scalar(f"{mode}/NIQE", niqe_metrics.avg, epoch + 1)
    else:
        raise ValueError("Unsupported mode, please use `Valid` or `Test`.")

    return niqe_metrics.avg
