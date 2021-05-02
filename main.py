from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    Dropout,
    MaxPooling2D
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow import constant, add
import pandas as pd
import numpy as np
import cv2


def load_datas():
    training_datas = pd.read_csv("Sample_Label .csv").values
    test_datas = pd.read_csv("Test.csv").values


def create_model():
    model = Sequential()
    model.add(Conv2D())

    return model


(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

