import os, sys
module_path = os.path.abspath(os.path.join('.'))
sys.path.append(module_path)

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers import TimeDistributed
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

from keras import backend as K
K.set_image_dim_ordering('th')

def VGG_16(weights_path=None, shape=(48, 48)):
    model = Sequential()
    model.add(TimeDistributed(ZeroPadding2D((1,1), input_shape=(1, 48, 48))))
    model.add(TimeDistributed(Convolution2D(32, 3, 3, activation='relu')))
    model.add(TimeDistributed(ZeroPadding2D((1,1))))
    model.add(TimeDistributed(Convolution2D(32, 3, 3, activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2,2))))

    model.add(TimeDistributed(ZeroPadding2D((1,1))))
    model.add(TimeDistributed(Convolution2D(64, 3, 3, activation='relu')))
    model.add(TimeDistributed(ZeroPadding2D((1,1))))
    model.add(TimeDistributed(Convolution2D(64, 3, 3, activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2,2))))

    model.add(TimeDistributed(ZeroPadding2D((1,1))))
    model.add(TimeDistributed(Convolution2D(128, 3, 3, activation='relu')))
    model.add(TimeDistributed(ZeroPadding2D((1,1))))
    model.add(TimeDistributed(Convolution2D(128, 3, 3, activation='relu')))
    model.add(TimeDistributed(ZeroPadding2D((1,1))))
    model.add(TimeDistributed(Convolution2D(128, 3, 3, activation='relu')))
    model.add(TimeDistributed(MaxPooling2D((2,2), strides=(2,2))))
    
    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(1024, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(512, activation='relu')))
    model.add(TimeDistributed(Dropout(0.5)))
    model.add(TimeDistributed(Dense(6, activation='softmax')))
    
    print ("Create model successfully")
    if weights_path:
        model.load_weights(weights_path)

    model.compile(optimizer='adam', loss='categorical_crossentropy', \
        metrics=['accuracy'])

    model.save(weights_path.replace(".h5",".model.h5"))

    return model
