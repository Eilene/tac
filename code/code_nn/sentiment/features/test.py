# -*- coding:UTF-8 -*-
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,Adam, SGD
from keras.callbacks import EarlyStopping
from keras.regularizers import l2

def test(x_train,y_train):
    x_train = np.array(x_train)
    length = x_train.shape[-1]
    sample_num = x_train.shape[-2]
    x_train_averg = []
    for i in x_train:
        temp = np.zeros(length)
        for j in i:
           temp = temp + j
        temp = temp / float(sample_num)
        x_train_averg.append(list(temp))
    x_train_averg = np.array(x_train_averg)
    y_train_averg = np.array(y_train)
    model = train_model(x_train_averg,y_train_averg)
    return model


# 模型训练
def train_model(x_train_averg,y_train_averg):
    split_at = len(x_train_averg) - len(x_train_averg) // 10
    # print split_at
    (x_train_data, x_val_data) = x_train_averg[:split_at], x_train_averg[split_at:]
    (y_train_data, y_val_data) = y_train_averg[:split_at], y_train_averg[split_at:]
    # 输出类别个数
    num_classes = 2
    y_train_data = keras.utils.to_categorical(y_train_data, num_classes)
    y_val_data = keras.utils.to_categorical(y_val_data, num_classes)

    batch_size = 32
    epochs = 50
    # 特征个数
    shape = x_train_averg.shape[1]

    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=epochs * 0.1)

    model.fit(x_train_data, y_train_data,batch_size=batch_size, epochs=epochs,verbose=1,validation_data = (x_val_data,y_val_data), callbacks=[early_stopping])
    return model

