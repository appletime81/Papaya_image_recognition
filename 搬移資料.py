import os
import glob
import pandas as pd


def sort_func(x):
    new_x = x.split("/")[1].split(".")[0]
    if len(new_x) == 1:
        return int(x[-5:-4])
    elif len(new_x) == 2:
        return int(x[-6:-4])
    else:
        return int(x[-7:-4])


# 搬移Test資料夾裡的圖片至Dataset資料夾
test_img_paths = glob.glob("Test/*.JPG")
test_img_paths = sorted(test_img_paths, key=sort_func)
for path in test_img_paths:
    new_name = str(int(path.split("/")[1].split(".")[0]) + 211)
    print(new_name)
    os.rename(path, f"Dataset/{new_name}.JPG")

# 搬移Sample資料夾裡的圖片至Dataset資料夾
sample_img_paths = glob.glob("Sample/*.JPG")
sample_img_paths = sorted(sample_img_paths, key=sort_func)
for path in sample_img_paths:
    new_name = f"Dataset/{path.split('/')[1].split('.')[0]}.JPG"
    print(new_name)
    os.rename(path, new_name)

# 重新編輯csv檔
img_paths = []
labels = []
sample_df = pd.read_csv("Sample_Label.csv").values
test_df = pd.read_csv("Test.csv").values
for data in sample_df:
    img_paths.append("Dataset/" + data[0].split("/")[1].replace("jpg", "JPG"))
    labels.append(data[1])

for data in test_df:
    file_name = "Dataset/" + str(int(data[0].split(".")[0]) + 211) + ".JPG"
    img_paths.append(file_name)
    labels.append(data[1])

images_info = {
    "img_path": img_paths,
    "label": labels
}

new_df = pd.DataFrame(images_info)
new_df.to_csv("Label.csv", index=False)
