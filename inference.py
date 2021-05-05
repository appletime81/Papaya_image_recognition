import cv2
import numpy as np
import pandas as pd
import time

from glob import glob
from densenet_bc_moel import dense_net_bc_model


def load_for_one_folder(folder_path):
    img_paths = glob(f"{folder_path}/*.JPG")

    def sort_func(x):
        file_name = x.split("/")[1].split(".")[0]
        if len(x) == 1:
            return int(file_name)
        elif len(x) == 2:
            return int(file_name)
        else:
            return int(file_name)

    img_paths = sorted(img_paths, key=sort_func)

    img = cv2.imread(img_paths[0])
    img = cv2.resize(img, (64, 64))
    test_data = img
    for i, img_path in enumerate(img_paths):
        if i > 0:
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))
            test_data = np.concatenate((test_data, img), axis=0)
    test_data = test_data.reshape(len(img_paths), 64, 64, 3)
    test_data = test_data.astype("float32")
    test_data /= 255

    return test_data


def load_for_one_file(file_path):
    test_data = cv2.imread(file_path)
    test_data = cv2.resize(test_data, (64, 64))
    test_data = test_data.reshape(1, 64, 64, 3)
    test_data = test_data.astype("float32")
    test_data /= 255

    return test_data


if __name__ == "__main__":
    map_dict = {
        0: "A",
        1: "B",
        2: "C"
    }

    test_data = load_for_one_folder("Dataset")
    original_df = pd.read_csv("Label.csv")

    result = []

    if test_data.shape == 1:
        model = dense_net_bc_model()
        model.load_weights("saved_models/cifar10_densenet_model.55_95.23%.h5")
        ans = model.predict(test_data)
        ans = ans.tolist()
        ans = map_dict[ans[0].index(max(ans[0]))]
        print(ans)
    else:
        model = dense_net_bc_model()
        model.load_weights("cifar10_densenet_model.46_100%.h5")
        for i in range(test_data.shape[0]):
            temp_test_data = test_data[i].reshape(
                1,
                test_data[i].shape[0],
                test_data[i].shape[1],
                test_data[i].shape[2]
            )
            ans = model.predict(temp_test_data)
            ans = ans.tolist()
            ans = map_dict[ans[0].index(max(ans[0]))]
            result += [ans]
            if i == test_data.shape[0] - 1:
                print(f"判讀進度: 100%")
            else:
                print(f"判讀進度: {i*100/test_data.shape[0]}%")

    original_df["result"] = result
    original_df.to_csv("result.csv", index=False)
    time.sleep(0.1)

    # 秀出與ground truth不一樣的判讀結果
    df = pd.read_csv("result.csv")
    for i in range(len(df)):
        if df["label"][i] != df["result"][i]:
            print(df["img_path"][i])
