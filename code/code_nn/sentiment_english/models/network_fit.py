# coding=utf-8
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,Adam, SGD
from keras.callbacks import EarlyStopping

from src.sentiment_english.utils.all_utils_package import *
from src.sentiment_english.features.embedding_vector_features import gen_embeddings_vector_features

from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score


# 模型训练
def train_model(x_train_avg, y_train_avg, num_classes):
    x_train_avg = np.array(x_train_avg)
    y_train_avg = np.array(y_train_avg)

    split_at = len(x_train_avg) - len(x_train_avg) // 10
    # print split_at
    (x_train_data, x_val_data) = x_train_avg[:split_at], x_train_avg[split_at:]
    (y_train_data, y_val_data) = y_train_avg[:split_at], y_train_avg[split_at:]
    # 输出类别个数
    # num_classes = 2
    y_train_data = keras.utils.to_categorical(y_train_data, num_classes)
    y_val_data = keras.utils.to_categorical(y_val_data, num_classes)

    batch_size = 32
    epochs = 50
    # 特征个数
    shape = x_train_avg.shape[1]

    model = Sequential()
    model.add(Dense(32, activation='relu', input_shape=(shape,)))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                             decay=0.0), metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=epochs * 0.1)

    model.fit(x_train_data, y_train_data, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(x_val_data, y_val_data), callbacks=[early_stopping])

    return model


# grid search


# 建立模型
def create_model(shape, num_classes, drop_rate=0.5, optimizer='adam', hidden_unit1=32, hidden_unit2=16,
                 activation="relu", init_mode="uniform"):
    model = Sequential()
    model.add(Dense(hidden_unit1, activation=activation, kernel_initializer=init_mode, input_shape=(shape,)))
    model.add(Dropout(drop_rate))
    model.add(Dense(hidden_unit2, activation=activation))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    return model


def grid_search(x_train, y_train, num_classes):
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    y_train = keras.utils.to_categorical(y_train, num_classes)
    shape = x_train.shape[1]  # 特征个数

    model = KerasClassifier(build_fn=create_model, shape=shape, num_classes=num_classes, verbose=0)
    # define the grid search parameters
    batch_size = [8]
    # batch_size = [8,16,32,64]
    epochs = [10,50]
    drop_rate = [0.3]
    # drop_rate = [0.0,0.3,0.5]
    optimizer = ['RMSprop']
    # optimizer = ['RMSprop','Adam']
    hidden_unit = [32,16]
    #hidden_unit = [16,24,32,64]
    init_mode = ['uniform',"normal"]
    #init_mode = ['uniform', 'lecun_uniform', 'normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    activation = ['relu']
    #activation = ['relu', 'tanh', 'sigmoid']

    #param_grid = dict(hidden_unit=hidden_unit,batch_size=batch_size)

    param_grid = dict(
        batch_size=batch_size, nb_epoch=epochs,drop_rate=drop_rate,optimizer=optimizer,
                      hidden_unit1=hidden_unit, hidden_unit2=hidden_unit, init_mode=init_mode, activation=activation)

    print param_grid

    # 自定义评分函数
    # tag_score = make_scorer(f1_score, average='micro')
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=10)
    model.fit(x_train, y_train)
    grid_result = grid.fit(x_train, y_train)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    return grid_result
