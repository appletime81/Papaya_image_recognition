import os
import numpy as np

from utils.load_datas import load_datas
from densenet_bc_moel import dense_net_bc_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    ReduceLROnPlateau,
    LearningRateScheduler,
    TensorBoard
)


def lr_schedule(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print("Learning rate: ", lr)
    return lr


def training_processing(x_train, y_train, x_test, y_test):
    # training parameters
    batch_size = 3
    epochs = 200

    # saved model directory
    save_dir = os.path.join(os.getcwd(), "saved_models")
    model_name = "dense_net_model.{epoch:02d}.h5"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)

    # prepare callbacks for model saving and for learning rate reducer
    checkpoint = ModelCheckpoint(filepath=filepath, monitor="val_acc", verbose=1, save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
    tensorboard_call_back = TensorBoard(log_dir="./log", histogram_freq=1, write_grads=True)
    callbacks = [checkpoint, lr_reducer, lr_scheduler, tensorboard_call_back]

    # create and compile model
    model = dense_net_bc_model()
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(1e-3),
        metrics=["acc"]
    )

    # start training
    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        shuffle=True,
        callbacks=callbacks
    )

    # evaluate model
    scores = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])


if __name__ == "__main__":
    params = {
        "train_csv_file": "train.csv",
        "test_csv_file": "test.csv"
    }

    (x_train, y_train), (x_test, y_test) = load_datas(**params)
    training_processing(x_train, y_train, x_test, y_test)
