import os

num_worker = os.cpu_count()

os.system("pip install -r requirements.txt")
os.system("python scripts/download_sunhays80.py")
os.system(
    f"python scripts/prepare_dataset.py --images_dir datasets/SunHays80/origin_train --output_dir datasets/SunHays80/Nexus/train --image_size 510 --step 255 --num_workers {num_worker}"
)
os.system(
    f"python scripts/prepare_dataset.py --images_dir datasets/SunHays80/origin_valid --output_dir datasets/SunHays80/Nexus/valid --image_size 510 --step 510 --num_workers {num_worker}"
)
