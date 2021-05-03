import pandas as pd
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
    train_data = {"img_path": [], "label": []}
    test_data = {"img_path": [], "label": []}

    for img_path, label in zip(img_paths, labels):
        info_dict[label] += [img_path]

    for key, value in info_dict.items():
        test_img_paths = sample(value, 9)
        for file in value:
            if file not in test_img_paths:
                train_data["img_path"] += [file]
                train_data["label"] += [key]
            else:
                test_data["img_path"] += [file]
                test_data["label"] += [key]

    train_df = pd.DataFrame(train_data)
    test_df = pd.DataFrame(test_data)

    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)


if __name__ == "__main__":
    csv_file = "Label.csv"
    separate_img_datas_into_two_files(csv_file)
