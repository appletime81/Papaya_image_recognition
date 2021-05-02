from utils.load_datas import load_datas
import time
start = time.time()

params = {
    "sample_img_path": "Sample/*JPG",
    "test_img_path": "Test/*JPG",
    "sample_csv_file": "Sample_Label.csv",
    "test_csv_file": "Test.csv",
}
(x_train, y_train), (x_test, y_test) = load_datas(**params)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
print(time.time() - start)