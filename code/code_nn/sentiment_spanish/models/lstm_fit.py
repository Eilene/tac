# coding=utf-8

import keras
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM
from keras.models import Sequential

from src.sentiment_english.features.lstm_features import *
from src.sentiment_english.utils.all_utils_package import *


def lstm_fit(x_train, y_train, embedding_matrix, nb_words, classnum):
    # 训练集进一步划分开发集
    split_at = len(x_train) - len(x_train) // 10  # 这是多少？
    # print split_at
    y_train = np.asarray(y_train)
    (x_train_new, x_dev) = x_train[:split_at], x_train[split_at:]
    (y_train_new, y_dev) = y_train[:split_at], y_train[split_at:]

    # 转换标签格式
    y_train_new = keras.utils.to_categorical(y_train_new, classnum)
    y_dev = keras.utils.to_categorical(y_dev, classnum)
    # print x_train, y_train

    print('Shape of train data tensor:', x_train_new.shape)
    print('Shape of train label tensor:', y_train_new.shape)

    batch_size = 32
    embedding_layer = Embedding(nb_words + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    print('Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.add(Dense(classnum, activation='sigmoid'))
    # model.layers[1].trainable = False

    # try using different optimizers and different optimizer configs
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    model.compile(loss=keras.losses.binary_crossentropy, optimizer='Adam', metrics=['accuracy'])
    model.summary()

    print('Train...')
    model.fit(x_train_new, y_train_new,
              batch_size=batch_size,
              epochs=1,
              validation_data=(x_dev, y_dev))
    score, acc = model.evaluate(x_dev, y_dev,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return model
