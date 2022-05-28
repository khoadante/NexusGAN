import time
import torch
import random
import numpy as np
from torch import nn, optim
from torch.cuda import amp
from torch.nn import functional as F
from classes.prefetchers import CUDAPrefetcher
from torch.utils.tensorboard import SummaryWriter
from networks.nexusgan.losses import ContentLoss, GANLoss
from utils.info_meters import AverageMeter, ProgressMeter

import utils.image_processing as imgproc
import config


def train_nexusnet(
    model: nn.Module,
    ema_model: nn.Module,
    train_prefetcher: CUDAPrefetcher,
    pixel_criterion: nn.MSELoss,
    optimizer: optim.AdamW,
    epoch: int,
    scaler: amp.GradScaler,
    writer: SummaryWriter,
) -> None:
    """Training main program

    Args:
        model (nn.Module): the generator model in the generative network
        ema_model (nn.Module): Exponential Moving Average Model
        train_prefetcher (CUDAPrefetcher): training dataset iterator
        pixel_criterion (nn.L1Loss): Calculate the pixel difference between real and fake samples
        optimizer (optim.Adam): optimizer for optimizing generator models in generative networks
        epoch (int): number of training epochs during training the generative network
        scaler (amp.GradScaler): Mixed precision training function
        writer (SummaryWrite): log file management function

    """
    # Defining JPEG image manipulation methods
    jpeg_operation = imgproc.DiffJPEG(differentiable=False).to(
        config.device, non_blocking=True
    )
    # Define image sharpening method
    usm_sharpener = imgproc.USMSharp().to(config.device, non_blocking=True)

    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)
    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":6.6f")
    progress = ProgressMeter(
        batches, [batch_time, data_time, losses], prefix=f"Epoch: [{epoch + 1}]"
    )

    # Put the generator in training mode
    model.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        hr = batch_data["hr"].to(device=config.device, non_blocking=True)
        kernel1 = batch_data["kernel1"].to(device=config.device, non_blocking=True)
        kernel2 = batch_data["kernel2"].to(device=config.device, non_blocking=True)
        sinc_kernel = batch_data["sinc_kernel"].to(
            device=config.device, non_blocking=True
        )

        # # Sharpen high-resolution images
        out = usm_sharpener(hr)

        # Get original image size
        image_height, image_width = out.size()[2:4]

        # First degradation process
        # Gaussian blur
        if (
            np.random.uniform()
            <= config.degradation_process_parameters_dict["first_blur_probability"]
        ):
            out = imgproc.blur(out, kernel1)

        # Resize
        updown_type = random.choices(
            ["up", "down", "keep"],
            config.degradation_process_parameters_dict["resize_probability1"],
        )[0]
        if updown_type == "up":
            scale = np.random.uniform(
                1, config.degradation_process_parameters_dict["resize_range1"][1]
            )
        elif updown_type == "down":
            scale = np.random.uniform(
                config.degradation_process_parameters_dict["resize_range1"][0], 1
            )
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # Noise
        if (
            np.random.uniform()
            < config.degradation_process_parameters_dict["gaussian_noise_probability1"]
        ):
            out = imgproc.random_add_gaussian_noise_pt(
                image=out,
                sigma_range=config.degradation_process_parameters_dict["noise_range1"],
                clip=True,
                rounds=False,
                gray_prob=config.degradation_process_parameters_dict[
                    "gray_noise_probability1"
                ],
            )
        else:
            out = imgproc.random_add_poisson_noise_pt(
                image=out,
                scale_range=config.degradation_process_parameters_dict[
                    "poisson_scale_range1"
                ],
                gray_prob=config.degradation_process_parameters_dict[
                    "gray_noise_probability1"
                ],
                clip=True,
                rounds=False,
            )

        # JPEG
        quality = out.new_zeros(out.size(0)).uniform_(
            *config.degradation_process_parameters_dict["jpeg_range1"]
        )
        out = torch.clamp(
            out, 0, 1
        )  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = jpeg_operation(out, quality=quality)

        # Second degradation process
        # Gaussian blur
        if (
            np.random.uniform()
            < config.degradation_process_parameters_dict["second_blur_probability"]
        ):
            out = imgproc.blur(out, kernel2)

        # Resize
        updown_type = random.choices(
            ["up", "down", "keep"],
            config.degradation_process_parameters_dict["resize_probability2"],
        )[0]
        if updown_type == "up":
            scale = np.random.uniform(
                1, config.degradation_process_parameters_dict["resize_range2"][1]
            )
        elif updown_type == "down":
            scale = np.random.uniform(
                config.degradation_process_parameters_dict["resize_range2"][0], 1
            )
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(
            out,
            size=(
                int(image_height / config.upscale_factor * scale),
                int(image_width / config.upscale_factor * scale),
            ),
            mode=mode,
        )

        # Noise
        if (
            np.random.uniform()
            < config.degradation_process_parameters_dict["gaussian_noise_probability2"]
        ):
            out = imgproc.random_add_gaussian_noise_pt(
                image=out,
                sigma_range=config.degradation_process_parameters_dict["noise_range2"],
                clip=True,
                rounds=False,
                gray_prob=config.degradation_process_parameters_dict[
                    "gray_noise_probability2"
                ],
            )
        else:
            out = imgproc.random_add_poisson_noise_pt(
                image=out,
                scale_range=config.degradation_process_parameters_dict[
                    "poisson_scale_range2"
                ],
                gray_prob=config.degradation_process_parameters_dict[
                    "gray_noise_probability2"
                ],
                clip=True,
                rounds=False,
            )

        if np.random.uniform() < 0.5:
            # Resize
            out = F.interpolate(
                out,
                size=(
                    image_height // config.upscale_factor,
                    image_width // config.upscale_factor,
                ),
                mode=random.choice(["area", "bilinear", "bicubic"]),
            )
            # Sinc blur
            out = imgproc.blur(out, sinc_kernel)

            # JPEG
            quality = out.new_zeros(out.size(0)).uniform_(
                *config.degradation_process_parameters_dict["jpeg_range2"]
            )
            out = torch.clamp(out, 0, 1)
            out = jpeg_operation(out, quality=quality)
        else:
            # JPEG
            quality = out.new_zeros(out.size(0)).uniform_(
                *config.degradation_process_parameters_dict["jpeg_range2"]
            )
            out = torch.clamp(out, 0, 1)
            out = jpeg_operation(out, quality=quality)

            # Resize
            out = F.interpolate(
                out,
                size=(
                    image_height // config.upscale_factor,
                    image_width // config.upscale_factor,
                ),
                mode=random.choice(["area", "bilinear", "bicubic"]),
            )

            # Sinc blur
            out = imgproc.blur(out, sinc_kernel)

        # Clamp and round
        lr = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

        # LR and HR crop the specified area respectively
        lr, hr = imgproc.random_crop(lr, hr, config.image_size, config.upscale_factor)

        # Initialize the generator gradient
        model.zero_grad(set_to_none=True)

        # Mixed precision training
        with amp.autocast():
            sr = model(lr)
            loss = pixel_criterion(sr, hr)

        # Backpropagation
        scaler.scale(loss).backward()
        # update generator weights
        scaler.step(optimizer)
        scaler.update()

        # Update EMA
        ema_model.update()

        # Statistical loss value for terminal data output
        losses.update(loss.item(), lr.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Record training log information
        if batch_index % config.print_frequency == 0:
            # Writer Loss to file
            writer.add_scalar(
                "Train/Loss", loss.item(), batch_index + epoch * batches + 1
            )
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After a batch of data is calculated, add 1 to the number of batches
        batch_index += 1


def train_nexusgan(
    discriminator: nn.Module,
    generator: nn.Module,
    ema_model: nn.Module,
    train_prefetcher: CUDAPrefetcher,
    pixel_criterion: nn.L1Loss,
    content_criterion: ContentLoss,
    adversarial_criterion: GANLoss,
    d_optimizer: optim.Adam,
    g_optimizer: optim.Adam,
    epoch: int,
    scaler: amp.GradScaler,
    writer: SummaryWriter,
) -> None:
    """Training main program

    Args:
        discriminator (nn.Module): discriminator model in adversarial networks
        generator (nn.Module): generator model in adversarial networks
        ema_model (nn.Module): Exponential Moving Average Model
        train_prefetcher (CUDAPrefetcher): training dataset iterator
        pixel_criterion (nn.L1Loss): Calculate the pixel difference between real and fake samples
        content_criterion (ContentLoss): Calculate the feature difference between real samples and fake samples by the feature extraction model
        adversarial_criterion (nn.BCEWithLogitsLoss): Calculate the semantic difference between real samples and fake samples by the discriminator model
        d_optimizer (optim.Adam): an optimizer for optimizing discriminator models in adversarial networks
        g_optimizer (optim.Adam): an optimizer for optimizing generator models in adversarial networks
        epoch (int): number of training epochs during training the adversarial network
        scaler (amp.GradScaler): Mixed precision training function
        writer (SummaryWrite): log file management function

    """
    # Defining JPEG image manipulation methods
    jpeg_operation = imgproc.DiffJPEG(differentiable=False)
    jpeg_operation = jpeg_operation.to(device=config.device, non_blocking=True)
    # Define image sharpening method
    usm_sharpener = imgproc.USMSharp()
    usm_sharpener = usm_sharpener.to(device=config.device, non_blocking=True)

    # Calculate how many batches of data are in each Epoch
    batches = len(train_prefetcher)

    # Print information of progress bar during training
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    pixel_losses = AverageMeter("Pixel loss", ":6.6f")
    content_losses = AverageMeter("Content loss", ":6.6f")
    adversarial_losses = AverageMeter("Adversarial loss", ":6.6f")
    d_hr_probabilities = AverageMeter("D(HR)", ":6.3f")
    d_sr_probabilities = AverageMeter("D(SR)", ":6.3f")
    progress = ProgressMeter(
        batches,
        [
            batch_time,
            data_time,
            pixel_losses,
            content_losses,
            adversarial_losses,
            d_hr_probabilities,
            d_sr_probabilities,
        ],
        prefix=f"Epoch: [{epoch + 1}]",
    )

    # Put all model in train mode.
    discriminator.train()
    generator.train()

    # Initialize the number of data batches to print logs on the terminal
    batch_index = 0

    # Initialize the data loader and load the first batch of data
    train_prefetcher.reset()
    batch_data = train_prefetcher.next()

    # Get the initialization training time
    end = time.time()

    while batch_data is not None:
        # Calculate the time it takes to load a batch of data
        data_time.update(time.time() - end)

        hr = batch_data["hr"].to(device=config.device, non_blocking=True)
        kernel1 = batch_data["kernel1"].to(device=config.device, non_blocking=True)
        kernel2 = batch_data["kernel2"].to(device=config.device, non_blocking=True)
        sinc_kernel = batch_data["sinc_kernel"].to(
            device=config.device, non_blocking=True
        )

        # Sharpen high-resolution images
        out = usm_sharpener(hr)

        # Get original image size
        image_height, image_width = out.size()[2:4]

        # First degradation process
        # Gaussian blur
        if (
            np.random.uniform()
            <= config.degradation_process_parameters_dict["first_blur_probability"]
        ):
            out = imgproc.blur(out, kernel1)

        # Resize
        updown_type = random.choices(
            ["up", "down", "keep"],
            config.degradation_process_parameters_dict["resize_probability1"],
        )[0]
        if updown_type == "up":
            scale = np.random.uniform(
                1, config.degradation_process_parameters_dict["resize_range1"][1]
            )
        elif updown_type == "down":
            scale = np.random.uniform(
                config.degradation_process_parameters_dict["resize_range1"][0], 1
            )
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)

        # Noise
        if (
            np.random.uniform()
            < config.degradation_process_parameters_dict["gaussian_noise_probability1"]
        ):
            out = imgproc.random_add_gaussian_noise_pt(
                image=out,
                sigma_range=config.degradation_process_parameters_dict["noise_range1"],
                clip=True,
                rounds=False,
                gray_prob=config.degradation_process_parameters_dict[
                    "gray_noise_probability1"
                ],
            )
        else:
            out = imgproc.random_add_poisson_noise_pt(
                image=out,
                scale_range=config.degradation_process_parameters_dict[
                    "poisson_scale_range1"
                ],
                gray_prob=config.degradation_process_parameters_dict[
                    "gray_noise_probability1"
                ],
                clip=True,
                rounds=False,
            )

        # JPEG
        quality = out.new_zeros(out.size(0)).uniform_(
            *config.degradation_process_parameters_dict["jpeg_range1"]
        )
        out = torch.clamp(
            out, 0, 1
        )  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = jpeg_operation(out, quality=quality)

        # Second degradation process
        # Gaussian blur
        if (
            np.random.uniform()
            < config.degradation_process_parameters_dict["second_blur_probability"]
        ):
            out = imgproc.blur(out, kernel2)

        # Resize
        updown_type = random.choices(
            ["up", "down", "keep"],
            config.degradation_process_parameters_dict["resize_probability2"],
        )[0]
        if updown_type == "up":
            scale = np.random.uniform(
                1, config.degradation_process_parameters_dict["resize_range2"][1]
            )
        elif updown_type == "down":
            scale = np.random.uniform(
                config.degradation_process_parameters_dict["resize_range2"][0], 1
            )
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(
            out,
            size=(
                int(image_height / config.upscale_factor * scale),
                int(image_width / config.upscale_factor * scale),
            ),
            mode=mode,
        )

        # Noise
        if (
            np.random.uniform()
            < config.degradation_process_parameters_dict["gaussian_noise_probability2"]
        ):
            out = imgproc.random_add_gaussian_noise_pt(
                image=out,
                sigma_range=config.degradation_process_parameters_dict["noise_range2"],
                clip=True,
                rounds=False,
                gray_prob=config.degradation_process_parameters_dict[
                    "gray_noise_probability2"
                ],
            )
        else:
            out = imgproc.random_add_poisson_noise_pt(
                image=out,
                scale_range=config.degradation_process_parameters_dict[
                    "poisson_scale_range2"
                ],
                gray_prob=config.degradation_process_parameters_dict[
                    "gray_noise_probability2"
                ],
                clip=True,
                rounds=False,
            )

        if np.random.uniform() < 0.5:
            # Resize
            out = F.interpolate(
                out,
                size=(
                    image_height // config.upscale_factor,
                    image_width // config.upscale_factor,
                ),
                mode=random.choice(["area", "bilinear", "bicubic"]),
            )
            # Sinc blur
            out = imgproc.blur(out, sinc_kernel)

            # JPEG
            quality = out.new_zeros(out.size(0)).uniform_(
                *config.degradation_process_parameters_dict["jpeg_range2"]
            )
            out = torch.clamp(out, 0, 1)
            out = jpeg_operation(out, quality=quality)
        else:
            # JPEG
            quality = out.new_zeros(out.size(0)).uniform_(
                *config.degradation_process_parameters_dict["jpeg_range2"]
            )
            out = torch.clamp(out, 0, 1)
            out = jpeg_operation(out, quality=quality)

            # Resize
            out = F.interpolate(
                out,
                size=(
                    image_height // config.upscale_factor,
                    image_width // config.upscale_factor,
                ),
                mode=random.choice(["area", "bilinear", "bicubic"]),
            )

            # Sinc blur
            out = imgproc.blur(out, sinc_kernel)

        # Clamp and round
        lr = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

        # LR and HR crop the specified area respectively
        lr, hr = imgproc.random_crop(lr, hr, config.image_size, config.upscale_factor)

        # Start training the generator model
        # During generator training, turn off discriminator backpropagation
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = False

        # Initialize generator model gradients
        generator.zero_grad(set_to_none=True)

        # Calculate the perceptual loss of the generator, mainly including pixel loss, feature loss and adversarial loss
        with amp.autocast():
            sr = generator(lr)
            pixel_loss = config.pixel_weight * pixel_criterion(usm_sharpener(sr), hr)
            content_loss = torch.sum(
                torch.multiply(
                    torch.tensor(config.content_weight),
                    torch.tensor(content_criterion(usm_sharpener(sr), hr)),
                )
            )
            adversarial_loss = config.adversarial_weight * adversarial_criterion(
                discriminator(sr), True
            )
            # Calculate the generator total loss value
            g_loss = pixel_loss + content_loss + adversarial_loss
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the fake samples
        scaler.scale(g_loss).backward()
        # Encourage the generator to generate higher quality fake samples, making it easier to fool the discriminator
        scaler.step(g_optimizer)
        scaler.update()
        # Finish training the generator model

        # Start training the discriminator model
        # During discriminator model training, enable discriminator model backpropagation
        for d_parameters in discriminator.parameters():
            d_parameters.requires_grad = True

        # Initialize the discriminator model gradients
        discriminator.zero_grad(set_to_none=True)

        # Calculate the classification score of the discriminator model for real samples
        with amp.autocast():
            hr_output = discriminator(hr)
            d_loss_hr = adversarial_criterion(hr_output, True)
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the real sample
        scaler.scale(d_loss_hr).backward()

        # Calculate the classification score of the discriminator model for fake samples
        with amp.autocast():
            # Use the generator model to generate fake samples
            sr = generator(lr)
            sr_output = discriminator(sr.detach().clone())
            d_loss_sr = adversarial_criterion(sr_output, False)
            # Calculate the total discriminator loss value
            d_loss = d_loss_sr + d_loss_hr
        # Call the gradient scaling function in the mixed precision API to backpropagate the gradient information of the fake samples
        scaler.scale(d_loss_sr).backward()
        # Improve the discriminator model's ability to classify real and fake samples
        scaler.step(d_optimizer)
        scaler.update()
        # Finish training the discriminator model

        # Update EMA
        ema_model.update()

        # Calculate the score of the discriminator on real samples and fake samples, the score of real samples is close to 1, and the score of fake samples is close to 0
        d_hr_probability = torch.sigmoid_(torch.mean(hr_output.detach()))
        d_sr_probability = torch.sigmoid_(torch.mean(sr_output.detach()))

        # Statistical accuracy and loss value for terminal data output
        pixel_losses.update(pixel_loss.item(), lr.size(0))
        content_losses.update(content_loss.item(), lr.size(0))
        adversarial_losses.update(adversarial_loss.item(), lr.size(0))
        d_hr_probabilities.update(d_hr_probability.item(), lr.size(0))
        d_sr_probabilities.update(d_sr_probability.item(), lr.size(0))

        # Calculate the time it takes to fully train a batch of data
        batch_time.update(time.time() - end)
        end = time.time()

        # Write the data during training to the training log file
        if batch_index % config.print_frequency == 0:
            iters = batch_index + epoch * batches + 1
            writer.add_scalar("Train/D_Loss", d_loss.item(), iters)
            writer.add_scalar("Train/G_Loss", g_loss.item(), iters)
            writer.add_scalar("Train/Pixel_Loss", pixel_loss.item(), iters)
            writer.add_scalar("Train/Content_Loss", content_loss.item(), iters)
            writer.add_scalar("Train/Adversarial_Loss", adversarial_loss.item(), iters)
            writer.add_scalar("Train/D(HR)_Probability", d_hr_probability.item(), iters)
            writer.add_scalar("Train/D(SR)_Probability", d_sr_probability.item(), iters)
            progress.display(batch_index)

        # Preload the next batch of data
        batch_data = train_prefetcher.next()

        # After training a batch of data, add 1 to the number of data batches to ensure that the terminal prints data normally
        batch_index += 1
