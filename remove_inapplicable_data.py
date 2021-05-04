import os

inapplicable_img_paths = [
    "Dataset/2.JPG",
    "Dataset/20.JPG",
    "Dataset/32.JPG",
    "Dataset/48.JPG",
    "Dataset/55.JPG",
    "Dataset/60.JPG",
    "Dataset/61.JPG",
    "Dataset/66.JPG",
    "Dataset/85.JPG",
    "Dataset/104.JPG",
    "Dataset/109.JPG",
    "Dataset/135.JPG",
    "Dataset/137.JPG",
    "Dataset/149.JPG",
    "Dataset/154.JPG",
    "Dataset/167.JPG",
    "Dataset/172.JPG",
    "Dataset/173.JPG",
    "Dataset/175.JPG",
    "Dataset/176.JPG",
    "Dataset/179.JPG",
    "Dataset/180.JPG",
    "Dataset/181.JPG",
    "Dataset/185.JPG",
    "Dataset/190.JPG",
    "Dataset/195.JPG",
    "Dataset/198.JPG",
    "Dataset/212.JPG",
    "Dataset/220.JPG",
    "Dataset/227.JPG",
    "Dataset/230.JPG",
    "Dataset/234.JPG",
    "Dataset/235.JPG",
    "Dataset/242.JPG",
    "Dataset/249.JPG",
    "Dataset/251.JPG",
    "Dataset/259.JPG"
]

for inapplicable_img_path in inapplicable_img_paths:
    os.rename(inapplicable_img_path, f"InapplicableImages/{inapplicable_img_path.split('/')[1]}")
