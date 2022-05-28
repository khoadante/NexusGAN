import os
import zipfile

os.system("wget -N https://dl.dropboxusercontent.com/s/dm1uazkyprejeyf/SunHays80.zip?dl=0")

if not os.path.exists("datasets/SunHays80"):
    os.makedirs("datasets/SunHays80")

with zipfile.ZipFile("SunHays80.zip", "r") as zip_ref:
    zip_ref.extractall("datasets/SunHays80")
