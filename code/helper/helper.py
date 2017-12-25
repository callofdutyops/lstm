import numpy as np
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


DATASET_DIR = "../dataset"

CURRENT_FILENAME = "current.txt"
SPINDLE_FOLLOWING_ERROR_FILENAME = "spindle_following_error.txt"
SPINDLE_LOAD_FILENAME = "spindle_load.txt"
SPINDLE_ROTATION_ERROR_FILENAME = "spindle_rotation_error.txt"
SPINDLE_TEMP_FILENAME = "spindle_temp.txt"

CURRENT_FILE = os.path.join(DATASET_DIR, CURRENT_FILENAME)
SPINDLE_FOLLOWING_ERROR_FILE = os.path.join(DATASET_DIR, SPINDLE_FOLLOWING_ERROR_FILENAME)
SPINDLE_LOAD_FILE = os.path.join(DATASET_DIR, SPINDLE_LOAD_FILENAME)
SPINDLE_ROTATION_ERROR_FILE = os.path.join(DATASET_DIR, SPINDLE_ROTATION_ERROR_FILENAME)
SPINDLE_TEMP_FILE = os.path.join(DATASET_DIR, SPINDLE_TEMP_FILENAME)


def read_single_line_file(path_to_dataset):

    with open(path_to_dataset) as f:
        reader = csv.reader(f, delimiter=",")
        datas = []
        for line in reader:
            try:
                datas.append(float(line[0]))
            except ValueError:
                pass

    print("Data loaded from csv. Formatting...")

    return datas


def make_lstm_trainable(datas, sequence_length=50,
                        train_size_percentage=0.8):
    print("Data formatting...")

    result = []
    for index in range(len(datas) - sequence_length):
        result.append(datas[index: index + sequence_length])
    result = np.array(result)

    assert result.shape[1] == sequence_length

    #TODO make this to sklearn MinMaxScaler
    result_mean = result.mean()
    result -= result_mean
    print("Shift : ", result_mean)
    print("Data  : ", result.shape)

    train_size = int(round(train_size_percentage * result.shape[0]))
    train = result[:train_size, :]
    X_train = train[:, :-1]
    y_train = train[:, -1]
    X_test = result[train_size:, :-1]
    y_test = result[train_size:, -1]

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    print("Data formatting complete.")

    return [X_train, y_train, X_test, y_test]
