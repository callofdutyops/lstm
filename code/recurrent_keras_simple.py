import matplotlib.pyplot as plt
import numpy as np
import time

from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential
np.random.seed(1234)

from helper.helper import *


def build_model():
    model = Sequential()
    layers = [1, 50, 100, 1]

    model.add(LSTM(
        layers[1],
        input_shape=(None, layers[0]),
        return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(
        layers[2],
        return_sequences=False))
    model.add(Dropout(0.2))

    model.add(Dense(
        layers[3]))
    model.add(Activation("linear"))

    start = time.time()
    model.compile(loss="mse", optimizer="rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model


def run_network(model=None, datas=None):
    global_start_time = time.time()
    epochs = 10
    sequence_length = 50

    if datas is None:
        raw_datas = read_single_line_file(CURRENT_FILE)
        datas = make_lstm_trainable(raw_datas, sequence_length)
        X_train, y_train, X_test, y_test = datas
    else:
        X_train, y_train, X_test, y_test = datas

    if model is None:
        model = build_model()

    try:
        model.fit(
            X_train, y_train,
            batch_size=8192, epochs=epochs, validation_split=0.05)
        predicted = model.predict(X_test)
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0

    try:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(y_test)
        plt.plot(predicted)
        plt.show()
    except Exception as e:
        print(str(e))
    print('Training duration (s) : ', time.time() - global_start_time)

    return model, y_test, predicted


if __name__ == '__main__':
    run_network()
