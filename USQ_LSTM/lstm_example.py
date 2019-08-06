import numpy as np
from keras.utils import np_utils
from sklearn.model_selection import train_test_split

from speechemotionrecognition.dnn import LSTM
from speechemotionrecognition.utilities import get_data, \
    get_feature_vector_from_mfcc

_CLASS_LABELS = ("Neutral", "Angry", "Happy", "Sad")

_DATA_PATH = './dataset'


def extract_data(flatten):
    data, labels = get_data(_DATA_PATH, class_labels=_CLASS_LABELS,
                            flatten=flatten)
    x_train, x_test, y_train, y_test = train_test_split(
        data,
        labels,
        test_size=0.2,
        random_state=42)
    return np.array(x_train), np.array(x_test), np.array(y_train), np.array(
        y_test), len(_CLASS_LABELS)


def get_feature_vector(file_path, flatten):
    return get_feature_vector_from_mfcc(file_path, flatten, mfcc_len=39)


def get_class_name(index):
    return _CLASS_LABELS[index]


def lstm_example():
    to_flatten = False
    x_train, x_test, y_train, y_test, num_labels = extract_data(
        flatten=to_flatten)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    print('Starting LSTM')
    model = LSTM(input_shape=x_train[0].shape,
                 num_classes=num_labels, save_path="./lstm.h5")
    print("Training")
    # model.train(x_train, y_train, x_test, y_test_train, n_epochs=10)
    model.restore_model()
    print("Evaluating")
    # model.evaluate(x_test, y_test)
    # model.save_model()
    filename = './dataset/Sad/1002_IEO_SAD_LO.wav'
    # filename = './laugh.wav'
    print("\nPredicted: {}\nActual: {}".format(
        get_class_name(
            model.predict_one(
                get_feature_vector_from_mfcc(filename, flatten=to_flatten))),
        get_class_name(3)))


if __name__ == "__main__":
    lstm_example()
