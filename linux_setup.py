import os

num_worker = os.cpu_count()

os.system("pip install -r requirements.txt")
os.system("python scripts/download_dataset.py")
os.system(
    f"python scripts/prepare_dataset.py --images_dir datasets/SunHays80/origin_train --output_dir datasets/SunHays80/Nexus/train --image_size 340 --step 170 --num_workers {num_worker}"
)
os.system(
    f"python scripts/prepare_dataset.py --images_dir datasets/SunHays80/origin_valid --output_dir datasets/SunHays80/Nexus/valid --image_size 340 --step 340 --num_workers {num_worker}"
)
