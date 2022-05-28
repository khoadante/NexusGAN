import os
import shutil

import torch
from torch.cuda import amp

from networks.nexusgan.models import EMA
from utils.build_models import build_nexusnet_model
from utils.model_losses import define_nexusnet_loss
from utils.model_optimizers import define_nexusnet_optimizer
from utils.model_schedulers import define_nexusnet_scheduler
from utils.prefetch_data import load_prefetchers
from utils.train_models import train_nexusnet
from utils.validate_models import validate_nexusnet
from utils.image_metrics import NIQE
import config


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_niqe = 100.0

    train_prefetcher, valid_prefetcher, test_prefetcher = load_prefetchers()
    print("Load all datasets successfully.")

    model = build_nexusnet_model()
    print("Build AESRNet model successfully.")

    pixel_criterion = define_nexusnet_loss()
    print("Define all loss functions successfully.")

    optimizer = define_nexusnet_optimizer(model)
    print("Define all optimizer functions successfully.")

    scheduler = define_nexusnet_scheduler(optimizer)
    print("Define all optimizer scheduler successfully.")

    print("Check whether the pretrained model is restored...")
    if config.resume:
        # Load checkpoint model
        checkpoint = torch.load(
            config.resume, map_location=lambda storage, loc: storage
        )
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_niqe = checkpoint["best_niqe"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = model.state_dict()
        new_state_dict = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if k in model_state_dict.keys()
        }
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)
        # Load the optimizer model
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the optimizer scheduler
        scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained model weights.")

    # Create a folder of super-resolution experiment results
    checkpoints_dir = os.path.join("checkpoints", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Initialize the gradient scaler
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    niqe_model = NIQE(config.upscale_factor, config.niqe_model_path)

    # Transfer the IQA model to the specified device
    niqe_model = niqe_model.to(device=config.device, non_blocking=True)

    # Create an Exponential Moving Average Model
    ema_model = EMA(model, config.ema_model_weight_decay)
    ema_model = ema_model.to(device=config.device, non_blocking=True)
    ema_model.register()

    for epoch in range(start_epoch, config.epochs):
        train_nexusnet(
            model,
            ema_model,
            train_prefetcher,
            pixel_criterion,
            optimizer,
            epoch,
            scaler,
        )
        _ = validate_nexusnet(
            model, ema_model, valid_prefetcher, epoch, niqe_model, "Valid"
        )
        niqe = validate_nexusnet(
            model, ema_model, test_prefetcher, epoch, niqe_model, "Test"
        )
        print("\n")

        # Update LR
        scheduler.step()

        # Automatically save the model with the highest index
        is_best = niqe < best_niqe
        best_niqe = min(niqe, best_niqe)
        torch.save(
            {
                "epoch": epoch + 1,
                "best_niqe": best_niqe,
                "state_dict": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            },
            os.path.join(checkpoints_dir, f"g_epoch_{epoch + 1}.pth.tar"),
        )
        if is_best:
            shutil.copyfile(
                os.path.join(checkpoints_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                os.path.join(results_dir, "g_best.pth.tar"),
            )
        if (epoch + 1) == config.epochs:
            shutil.copyfile(
                os.path.join(checkpoints_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                os.path.join(results_dir, "g_last.pth.tar"),
            )


if __name__ == "__main__":
    main()
