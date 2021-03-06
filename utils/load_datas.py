import pandas as pd
import numpy as np
import cv2


def load_datas(**kwargs):
    # get sample datas
    sample_img_paths = [item[0] for item in pd.read_csv(kwargs["train_csv_file"]).values]
    img = cv2.imread(sample_img_paths[0])
    img = cv2.resize(img, (64, 64))
    x_train = img
    for i, sample_img_path in enumerate(sample_img_paths):
        if i > 0:
            img = cv2.imread(sample_img_path)
            img = cv2.resize(img, (64, 64))
            x_train = np.concatenate((x_train, img), axis=0)
    x_train = x_train.reshape(len(sample_img_paths), 64, 64, 3)

    # get test datas
    test_img_paths = [item[0] for item in pd.read_csv(kwargs["test_csv_file"]).values]
    img = cv2.imread(test_img_paths[0])
    img = cv2.resize(img, (64, 64))
    x_test = img
    for i, test_img_path in enumerate(test_img_paths):
        if i > 0:
            img = cv2.imread(test_img_path)
            img = cv2.resize(img, (64, 64))
            x_test = np.concatenate((x_test, img), axis=0)
    x_test = x_test.reshape(len(test_img_paths), 64, 64, 3)

    # get train data labels
    label_dict = {
        "A": 0,
        "B": 1,
        "C": 2
    }

    y_train = []
    train_labels = pd.read_csv(kwargs["train_csv_file"]).values
    for train_label in train_labels:
        y_train.append(label_dict[train_label[1]])
    y_train = np.array(y_train)

    # get test data labels
    y_test = []
    test_labels = pd.read_csv(kwargs["test_csv_file"]).values
    for test_label in test_labels:
        y_test.append(label_dict[test_label[1]])
    y_test = np.array(y_test)

    return [(x_train, y_train), (x_test, y_test)]
