import time
from utils.load_datas import load_datas
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard
from model.model import cnn_model
from datetime import datetime
start = time.time()

# set global params
params = {
    "train_csv_file": "train.csv",
    "test_csv_file": "test.csv",
}


def compile_model(model):
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['acc'])
    return model


def train_model(model, x_train, y_train, batch_size=32, epochs=50):
    tensorboard_call_back = TensorBoard(log_dir="./log", histogram_freq=1, write_grads=True)
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=[tensorboard_call_back]
    )
    return history, model


if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_datas(**params)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    x_train = x_train.astype("float32")
    x_test = x_test.astype("float32")
    x_train /= 255
    x_test /= 255

    y_train = to_categorical(y_train, 3)
    y_test = to_categorical(y_test, 3)

    model = cnn_model()
    model = compile_model(model)
    history, model = train_model(model, x_train, y_train, batch_size=3, epochs=200)
    model.save_weights(f"papaya_model_{datetime.now().strftime('%Y%m%H%M')}.h5")
    loss, accuracy = model.evaluate(x_test, y_test)
    print('Test:')
    print('Loss: %s\nAccuracy: %s' % (loss, accuracy))

