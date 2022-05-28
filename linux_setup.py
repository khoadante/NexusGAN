import os

num_worker = os.cpu_count()

os.system("pip install -r requirements.txt")
os.system("python scripts/download_dataset.py")
os.system(
    f"python scripts/prepare_dataset.py --images_dir datasets/DIV2K/DIV2K_train_HR --output_dir datasets/DIV2K/Nexus/train --image_size 680 --step 340 --num_workers {num_worker}"
)
os.system(
    f"python scripts/prepare_dataset.py --images_dir datasets/DIV2K/DIV2K_valid_HR --output_dir datasets/DIV2K/Nexus/valid --image_size 680 --step 680 --num_workers {num_worker}"
)
