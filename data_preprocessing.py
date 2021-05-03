import pandas as pd
import shutil
from random import sample



def cal_each_class_num(info_dict, csv):
    datas = pd.read_csv(csv).values  # type: list
    for data in datas:
        info_dict[data[1]] += 1
    print(info_dict)


def separate_img_datas_into_two_files(csv):
    img_paths = [item[0] for item in pd.read_csv(csv).values]
    labels = [item[1] for item in pd.read_csv(csv).values]
    info_dict = {"A": [], "B": [], "C": []}

    for img_path, label in zip(img_paths, labels):
        info_dict[label] += [img_path]

    for key, value in info_dict.items():
        files = sample(value, 9)
        for file in files:
            pass


if __name__ == "__main__":
    # csv_file = "Label.csv"
    # separate_img_datas_into_two_files(csv_file)

    a = [i for i in range(100)]
    b = sample(a, 9)
    print(b)
