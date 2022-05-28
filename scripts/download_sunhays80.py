import os
import zipfile

os.system("wget -N https://dl.dropboxusercontent.com/s/dm1uazkyprejeyf/SunHays80.zip")

with zipfile.ZipFile("SunHays80.zip", "r") as zip_ref:
    zip_ref.extractall("datasets")
