# -*- coding:UTF-8 -*-
from Constant import *
from keras.models import  Sequential
import numpy as np
from keras import layers

def train_test(x,y,cata_feature_num):
    # Explicitly set apart 10% for validation data that we never train over.
    x = np.array(x)
    y = np.array(y)
    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    print('Training Data:')
    print(x_train.shape)
    print(y_train.shape)

    print('Validation Data:')
    print(x_val.shape)
    print(y_val.shape)

    # Try replacing GRU, or SimpleRNN.

    RNN = layers.LSTM
    HIDDEN_SIZE = 128
    BATCH_SIZE = 128
    LAYERS = 2

    Rows = Sentence_length#一个样本行的个数
    Columns = num_feature_num + cata_feature_num#一个样本列的个数
    Out_put = 3

    print('Build model...')
    model = Sequential()
    model.add(RNN(HIDDEN_SIZE, input_shape=(Rows, Columns), trainable=True))
    model.add(layers.RepeatVector(1))
    for _ in range(LAYERS):
        model.add(RNN(HIDDEN_SIZE, return_sequences=True))

    model.add(layers.TimeDistributed(layers.Dense(Out_put)))
    model.add(layers.Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()

    for iteration in range(1, 10):
        print()
        print('-' * 50)
        print('Iteration', iteration)
        model.fit(x_train, y_train, batch_size=BATCH_SIZE,validation_data=(x_val, y_val))
        score, acc = model.evaluate(x_val, y_val, batch_size=BATCH_SIZE)
        print('Test score:', score)
        print('Test accuracy:', acc)

