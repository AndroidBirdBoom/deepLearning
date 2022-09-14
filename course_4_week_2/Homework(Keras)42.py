import numpy as np
import keras
from keras import layers
from keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, Flatten, Dense
from keras.models import Model
from keras import optimizers, losses
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
import matplotlib.pyplot as plt

from kt_utils import *


def HappyModel(input_shape):
    """
    Implementation of the HappyModel.

    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Keras
    """

    X_input = Input(input_shape)

    Z1 = Conv2D(32, (3, 3), strides=(1, 1), padding='SAME')(X_input)

    bn0 = BatchNormalization(name='bn0')(Z1)

    A1 = Activation(activation='relu')(bn0)

    pool1 = MaxPooling2D((2, 2), name='maxpool')(A1)

    flatten1 = Flatten()(pool1)

    Z2 = Dense(1, activation='sigmoid', name='fc')(flatten1)

    model = Model(inputs=X_input, outputs=Z2, name='happymodel')

    return model


def model(input_shape):
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(32, (7, 7), strides=(1, 1), name='conv0')(X)

    X = BatchNormalization(axis=3, name='bn0')(X)

    X = Activation('relu')(X)

    X = MaxPooling2D((2, 2), name='max_pool')(X)

    X = Flatten()(X)

    X = Dense(1, activation='sigmoid', name='fc')(X)

    model = Model(inputs=X_input, outputs=X, name='HappyModel')

    return model


if __name__ == "__main__":
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    Y_train = Y_train_orig.T
    Y_test = Y_test_orig.T

    # print("number of training examples = " + str(X_train.shape[0]))
    # print("number of test examples = " + str(X_test.shape[0]))
    # print("X_train shape: " + str(X_train.shape))
    # print("Y_train shape: " + str(Y_train.shape))
    # print("X_test shape: " + str(X_test.shape))
    # print("Y_test shape: " + str(Y_test.shape))

    happyModel = HappyModel((64, 64, 3))
    happyModel.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    happyModel.fit(x=X_train, y=Y_train, batch_size=16, epochs=20)

    preds = happyModel.evaluate(x=X_test, y=Y_test)
    print("Loss = " + str(preds[0]))
    print("Test Accuracy = " + str(preds[1]))

    image_path = '2.png'
    img = image.load_img(image_path, target_size=(64, 64))

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    print(happyModel.predict(x))

