"""
This example demonstrates how to use `LSTM` model from
`speechemotionrecognition` package
"""

from common import extract_data, get_class_name
from keras.utils import np_utils

from speechemotionrecognition.dnn import LSTM
from speechemotionrecognition.utilities import get_feature_vector_from_mfcc

_CLASS_LABELS = ("Neutral", "Angry", "Happy", "Sad")


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
    model.train(x_train, y_train, x_test, y_test_train, n_epochs=10)
    print("Evaluating")
    model.evaluate(x_test, y_test)
    model.save_model()
    filename = '../dataset/Sad/09b03Ta.wav'
    # filename = './laugh.wav'
    print("\nPredicted: {}\nActual: {}".format(
        get_class_name(
            model.predict_one(
                get_feature_vector_from_mfcc(filename, flatten=to_flatten))),
        get_class_name(2)))


def lstm_saved_model():
    to_flatten = False
    x_train, x_test, y_train, y_test, num_labels = extract_data(
        flatten=to_flatten)
    y_train = np_utils.to_categorical(y_train)
    y_test_train = np_utils.to_categorical(y_test)
    print('Starting LSTM')
    model = LSTM(input_shape=x_train[0].shape,
                 num_classes=num_labels, save_path="./lstm.h5")
    print("Loading")
    model.restore_model()
    # filename = '../dataset/03a01Wa.wav'
    filename = '../dataset/angry.wav'
    print("\nPredicted: {}\nActual: {}".format(
        get_class_name(
            model.predict_one(get_feature_vector_from_mfcc(filename, flatten=to_flatten))),
        get_class_name(3)))


if __name__ == '__main__':
    lstm_example()
    # lstm_saved_model()
