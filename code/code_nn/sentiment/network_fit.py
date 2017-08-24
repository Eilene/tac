# coding=utf-8
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop,Adam, SGD
from keras.callbacks import EarlyStopping

from utils.evaluation import evaluation_3classes
from utils.find_source import find_sources
from utils.read_file_info_records import *
from utils.write_best import write_best_files
from utils.constants import *
from utils.file_records_other_modification import without_none, to_dict
from utils.get_labels import get_merged_labels
from utils.predict_by_proba import *
from utils.attach_predict_labels import *
from features.embedding_vector_features import gen_embeddings_vector_features
from utils.read_embedding_index import *


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
