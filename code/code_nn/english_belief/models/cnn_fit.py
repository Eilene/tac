# coding=utf-8

import keras
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential

from src.english_belief.features.matrix_features import gen_matrix_features, convert_features
from src.english_belief.utils.all_utils_package import *


def cnn_fit(x_train, y_train, classnum):
    # 训练集进一步划分开发集
    input_shape = x_train.shape[1:]  # 与samples个数无关
    split_at = len(x_train) - len(x_train) // 10  # 这是多少？
    # print split_at
    (x_train_new, x_dev) = x_train[:split_at], x_train[split_at:]
    (y_train_new, y_dev) = y_train[:split_at], y_train[split_at:]

    # 转换标签格式
    # print y_train_new
    y_train_new = keras.utils.to_categorical(y_train_new, classnum)
    y_dev = keras.utils.to_categorical(y_dev, classnum)
    # print y_train_new
    # print y_train_new.shape

    # 开始建立CNN模型
    batch_size = 32
    epochs = 3

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(classnum, activation='softmax'))
    # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=keras.losses.binary_crossentropy, optimizer='Adam', metrics=['accuracy'])
    model.summary()

    model.fit(x_train_new, y_train_new, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_dev, y_dev))
    score = model.evaluate(x_dev, y_dev, verbose=0)
    print 'Test loss:', score[0]
    print 'Test accuracy:', score[1]

    return model


