import glob


def load_datas(path):
    paths = glob.glob(path + "/*.JPG")
    print(paths)
    for i in range(len(paths)):
        print(paths[i])