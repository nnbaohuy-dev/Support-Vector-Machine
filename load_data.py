import cv2 as cv
import os
import kagglehub

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

path = kagglehub.dataset_download(
    "paultimothymooney/chest-xray-pneumonia",
    output_dir=DATA_DIR
)
print("Path to dataset files:", path)

