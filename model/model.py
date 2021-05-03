from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    Dropout,
    MaxPool2D,
    MaxPooling2D
)


def cnn_model():
    # model = Sequential(
    #     [
    #         Conv2D(64, 3, activation="relu", padding="same", input_shape=[256, 256, 3]),
    #         MaxPooling2D(2),
    #         Conv2D(128, 3, activation="relu", padding="same"),
    #         Conv2D(128, 3, activation="relu", padding="same"),
    #         MaxPooling2D(2),
    #         Conv2D(256, 3, activation="relu", padding="same"),
    #         Conv2D(256, 3, activation="relu", padding="same"),
    #         MaxPooling2D(2),
    #         Flatten(),
    #         Dense(128, activation="relu"),
    #         Dropout(0.5),
    #         Dense(64, activation="relu"),
    #         Dropout(0.5),
    #         Dense(3, activation="softmax")
    #     ]
    # )
    # input_shape = (224, 224, 3)
    #
    # model = Sequential([
    #             Conv2D(64, (3, 3), input_shape=input_shape, padding='same', activation='relu'),
    #             Conv2D(64, (3, 3), activation='relu', padding='same'),
    #             MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    #             Conv2D(128, (3, 3), activation='relu', padding='same'),
    #             Conv2D(128, (3, 3), activation='relu', padding='same', ),
    #             MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    #             Flatten(),  # 平铺层
    #             Dropout(0.2),
    #             Dense(128, activation='relu'),
    #             Dropout(0.2),
    #             Dense(3, activation='softmax')
    # ])

    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # 第一卷积层
    model.add(Conv2D(64, (3, 3), activation='relu'))  # 第二卷积层
    model.add(MaxPooling2D(pool_size=(2, 2)))  # 池化层
    model.add(Flatten())  # 平铺层
    model.add(Dropout(0.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='softmax'))

    return model
