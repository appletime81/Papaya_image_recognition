import pandas as pd
import glob
import cv2
import matplotlib.pyplot as plt


img_paths = glob.glob("Dataset/*.JPG")
img_paths = [img_path.replace("\\", "/") for img_path in img_paths]
for x in img_paths:
    print(x)

