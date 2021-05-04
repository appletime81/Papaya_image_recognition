import pandas as pd
from glob import glob

infoDict = {
    "img_path": [],
    "label": []
}

inapplicable_data = glob("InapplicableImages/*.JPG")
inapplicable_data = [item.replace("InapplicableImages", "Dataset") for item in inapplicable_data]

img_path_and_label_list = []
label_datas = pd.read_csv("Label.csv").values
for label_data in label_datas:
    if label_data[0] not in inapplicable_data:
        img_path_and_label_list.append([label_data[0], label_data[1]])

for img_path_and_label in img_path_and_label_list:
    infoDict["img_path"] += [img_path_and_label[0]]
    infoDict["label"] += [img_path_and_label[1]]

infoDataFrame = pd.DataFrame(infoDict)
infoDataFrame.to_csv("Label.csv", index=False)

