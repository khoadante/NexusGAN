import os
import shutil

import torch
from torch.cuda import amp

from networks.nexusgan.models import EMA
from utils.build_models import build_nexusgan_model
from utils.model_losses import define_nexusgan_loss
from utils.model_optimizers import define_nexusgan_optimizer
from utils.model_schedulers import define_nexusgan_scheduler
from utils.prefetch_data import load_prefetchers
from utils.train_models import train_nexusgan
from utils.validate_models import validate_nexusgan
from utils.image_metrics import NIQE

import config


def main():
    # Initialize the number of training epochs
    start_epoch = 0

    # Initialize training to generate network evaluation indicators
    best_niqe = 100.0

    train_prefetcher, valid_prefetcher, test_prefetcher = load_prefetchers()
    print("Load dataset successfully.")

    discriminator, generator = build_nexusgan_model()
    print("Build NexusGAN models successfully.")

    pixel_criterion, content_criterion, adversarial_criterion = define_nexusgan_loss()
    print("Define all loss functions successfully.")

    d_optimizer, g_optimizer = define_nexusgan_optimizer(discriminator, generator)
    print("Define all optimizer functions successfully.")

    d_scheduler, g_scheduler = define_nexusgan_scheduler(d_optimizer, g_optimizer)
    print("Define all optimizer scheduler functions successfully.")

    if config.resume:
        print("Loading NexusNet model weights")
        # Load checkpoint model
        checkpoint = torch.load(
            config.resume, map_location=lambda storage, loc: storage
        )
        generator.load_state_dict(checkpoint["state_dict"], strict=False)
        print("Loaded NexusNet model weights.")

    print("Check whether the pretrained discriminator model is restored...")
    if config.resume_d:
        # Load checkpoint model
        checkpoint = torch.load(
            config.resume_d, map_location=lambda storage, loc: storage
        )
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_niqe = checkpoint["best_niqe"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = discriminator.state_dict()
        new_state_dict = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if k in model_state_dict.keys()
        }
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        discriminator.load_state_dict(model_state_dict)
        # Load the optimizer model
        d_optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        d_scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained discriminator model weights.")

    print("Check whether the pretrained generator model is restored...")
    if config.resume_g:
        # Load checkpoint model
        checkpoint = torch.load(
            config.resume_g, map_location=lambda storage, loc: storage
        )
        # Restore the parameters in the training node to this point
        start_epoch = checkpoint["epoch"]
        best_niqe = checkpoint["best_niqe"]
        # Load checkpoint state dict. Extract the fitted model weights
        model_state_dict = generator.state_dict()
        new_state_dict = {
            k: v
            for k, v in checkpoint["state_dict"].items()
            if k in model_state_dict.keys()
        }
        # Overwrite the pretrained model weights to the current model
        model_state_dict.update(new_state_dict)
        generator.load_state_dict(model_state_dict)
        # Load the optimizer model
        g_optimizer.load_state_dict(checkpoint["optimizer"])
        # Load the scheduler model
        g_scheduler.load_state_dict(checkpoint["scheduler"])
        print("Loaded pretrained generator model weights.")

    # Create a folder of super-resolution experiment results
    checkpoints_dir = os.path.join("checkpoints", config.exp_name)
    results_dir = os.path.join("results", config.exp_name)
    if not os.path.exists(checkpoints_dir):
        os.makedirs(checkpoints_dir)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Initialize the gradient scaler.
    scaler = amp.GradScaler()

    # Create an IQA evaluation model
    niqe_model = NIQE(config.upscale_factor, config.niqe_model_path)

    # Transfer the IQA model to the specified device
    niqe_model = niqe_model.to(device=config.device, non_blocking=True)

    # Create an Exponential Moving Average Model
    ema_model = EMA(generator, config.ema_model_weight_decay)
    ema_model = ema_model.to(device=config.device, non_blocking=True)
    ema_model.register()

    for epoch in range(start_epoch, config.epochs):
        train_nexusgan(
            discriminator,
            generator,
            ema_model,
            train_prefetcher,
            pixel_criterion,
            content_criterion,
            adversarial_criterion,
            d_optimizer,
            g_optimizer,
            epoch,
            scaler,
        )
        _ = validate_nexusgan(
            generator, ema_model, valid_prefetcher, epoch, niqe_model, "Valid"
        )
        niqe = validate_nexusgan(
            generator, ema_model, test_prefetcher, epoch, niqe_model, "Test"
        )
        print("\n")

        # Update LR
        d_scheduler.step()
        g_scheduler.step()

        # Automatically save the model with the highest index
        is_best = niqe < best_niqe
        best_niqe = min(niqe, best_niqe)
        torch.save(
            {
                "epoch": epoch + 1,
                "best_niqe": best_niqe,
                "state_dict": discriminator.state_dict(),
                "optimizer": d_optimizer.state_dict(),
                "scheduler": d_scheduler.state_dict(),
            },
            os.path.join(checkpoints_dir, f"d_epoch_{epoch + 1}.pth.tar"),
        )
        torch.save(
            {
                "epoch": epoch + 1,
                "best_niqe": best_niqe,
                "state_dict": ema_model.state_dict(),
                "optimizer": g_optimizer.state_dict(),
                "scheduler": g_scheduler.state_dict(),
            },
            os.path.join(checkpoints_dir, f"g_epoch_{epoch + 1}.pth.tar"),
        )
        if is_best:
            shutil.copyfile(
                os.path.join(checkpoints_dir, f"d_epoch_{epoch + 1}.pth.tar"),
                os.path.join(results_dir, "d_best.pth.tar"),
            )
            shutil.copyfile(
                os.path.join(checkpoints_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                os.path.join(results_dir, "g_best.pth.tar"),
            )
        if (epoch + 1) == config.epochs:
            shutil.copyfile(
                os.path.join(checkpoints_dir, f"d_epoch_{epoch + 1}.pth.tar"),
                os.path.join(results_dir, "d_last.pth.tar"),
            )
            shutil.copyfile(
                os.path.join(checkpoints_dir, f"g_epoch_{epoch + 1}.pth.tar"),
                os.path.join(results_dir, "g_last.pth.tar"),
            )


if __name__ == "__main__":
    main()
