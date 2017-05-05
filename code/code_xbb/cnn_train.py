# -*- coding:UTF-8 -*-
from Constant import *
import keras
from feature import get_feature
#导入各种用到的模块组件
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

#mode = 1有无情感二分类,mode = 2正负情感分类

def train(mode):
    if(mode == 1):
        x,y = get_feature(mode,Have_none_path)
    else:
        x, y = get_feature(mode, Pos_neg_path)
    input_shape = x.shape[1:]#与samples个数无关
    split_at = len(x) - len(x) // 10
    (x_train, x_val) = x[:split_at], x[split_at:]
    (y_train, y_val) = y[:split_at], y[split_at:]

    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, 2)
    y_val = keras.utils.to_categorical(y_val, 2)
    print x_train,y_train

    #开始建立CNN模型
    batch_size = 128
    epochs = 3

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    #sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=keras.losses.binary_crossentropy,optimizer='Adam',metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=1, validation_data=(x_val, y_val))
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    #将模型保存在本地
    if(mode == 1):
        model.save(Have_none_model)
    else:
        model.save(Pos_neg_model)

# mode = 1
mode = 2#表示pos/neg情感分类
train(mode)

#测试模型部分，用的是训练过程和提取特征过程都没接触的数据
from keras.models import load_model
mode = 2
x,y = get_feature(mode,Test_pos_neg_path)
model = load_model(Pos_neg_model)
probabilities = model.predict(x)
y_predict = []
print probabilities
for i in probabilities:
    if(i[0]<i[1]):
        y_predict.append([1])
    else:
        y_predict.append([0])
accuracy = np.mean(y_predict == y)
print("Prediction Accuracy: %.2f%%" % (accuracy*100))