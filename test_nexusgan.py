import os

import cv2
import torch
from natsort import natsorted

import config

import numpy as np
from networks.nexusgan.models import Generator
from utils.image_metrics import NIQE
import utils.image_processing as imgproc


def main() -> None:
    # Initialize the super-resolution model
    model = Generator(config.in_channels, config.out_channels, config.upscale_factor).to(device=config.device,)
    print("Build NexusGAN model successfully.")

    # Load the super-resolution model weights
    checkpoint = torch.load(config.model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"Load NexusGAN model weights `{os.path.abspath(config.model_path)}` successfully.")

    # Create a folder of super-resolution experiment results
    results_dir = os.path.join("results", "test", config.exp_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Start the verification mode of the model.
    model.eval()
    # Turn on half-precision inference.
    model.half()

    # Initialize the sharpness evaluation function
    niqe = NIQE(config.upscale_factor, config.niqe_model_path)

    # Set the sharpness evaluation function calculation device to the specified model
    niqe = niqe.to(device=config.device, non_blocking=True)

    # Initialize IQA metrics
    niqe_metrics = 0.0

    # Get a list of test image file names.
    file_names = natsorted(os.listdir(config.lr_dir))
    # Get the number of test image files.
    total_files = len(file_names)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = total_files
    timings=np.zeros((repetitions,1))

    for index in range(total_files):
        lr_image_path = os.path.join(config.lr_dir, file_names[index])
        sr_image_path = os.path.join(config.sr_dir, file_names[index])

        print(f"Processing `{os.path.abspath(lr_image_path)}`...")
        # Read LR image
        lr_image = cv2.imread(lr_image_path, cv2.IMREAD_UNCHANGED)

        # Convert BGR channel image format data to RGB channel image format data
        lr_image = cv2.cvtColor(lr_image, cv2.COLOR_BGR2RGB)

        # Convert RGB channel image format data to Tensor channel image format data
        lr_tensor = imgproc.image2tensor(lr_image, range_norm=False, half=True).unsqueeze_(0)

        # Transfer Tensor channel image format data to CUDA device
        lr_tensor = lr_tensor.to(device=config.device, non_blocking=True)

        # Only reconstruct the Y channel image data.
        with torch.no_grad():
            starter.record()

            sr_tensor = model(lr_tensor)

            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[index] = curr_time

        # Save image
        sr_image = imgproc.tensor2image(sr_tensor, range_norm=False, half=True)
        sr_image = cv2.cvtColor(sr_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(sr_image_path, sr_image)

        # Cal IQA metrics
        niqe_metrics += niqe(sr_tensor).item()

    # Calculate the average value of the sharpness evaluation index,
    # and all index range values are cut according to the following values
    # NIQE range value is 0~100
    avg_niqe = 100 if niqe_metrics / total_files > 100 else niqe_metrics / total_files

    print(f"NIQE: {avg_niqe:4.2f} 100u")
    mean_syn = np.sum(timings) / repetitions
    print(mean_syn)


if __name__ == "__main__":
    main()
