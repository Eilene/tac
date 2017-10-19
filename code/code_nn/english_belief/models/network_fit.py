# coding=utf-8
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping

from src.english_belief.utils.all_utils_package import *
from src.english_belief.features.embedding_vector_features import gen_embeddings_vector_features


# 模型训练
def network_fit(x_train, y_train, num_classes, drop_rate=0.5, optimizer='adam', hidden_unit1=32, hidden_unit2=16,
                 activation="relu", init_mode="uniform", batch_size=8, epochs=20):
    x_train = np.array(x_train)
    y_train = np.array(y_train)

    split_at = len(x_train) - len(x_train) // 10
    # print split_at
    (x_train_data, x_val_data) = x_train[:split_at], x_train[split_at:]
    (y_train_data, y_val_data) = y_train[:split_at], y_train[split_at:]
    # 输出类别个数
    # num_classes = 2
    y_train_data = keras.utils.to_categorical(y_train_data, num_classes)
    y_val_data = keras.utils.to_categorical(y_val_data, num_classes)

    shape = x_train.shape[1]

    model = Sequential()
    model.add(Dense(hidden_unit1, activation=activation, kernel_initializer=init_mode, input_shape=(shape,)))
    model.add(Dropout(drop_rate))
    model.add(Dense(hidden_unit2, activation=activation))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # 训练
    early_stopping = EarlyStopping(monitor='val_loss', patience=epochs * 0.1)
    model.fit(x_train_data, y_train_data, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(x_val_data, y_val_data), callbacks=[early_stopping])

    return model


# 网格搜索
def grid_search_network(x_train, y_train, num_classes, k=10, pos_thred=0.1, neg_thred=0.3):
    # 定义网格搜索参数
    batch_size = [8]
    # batch_size = [8,16,32,64]
    epochs = [50]
    drop_rate = [0.5]
    # drop_rate = [0.0,0.3,0.5]
    optimizer = ['RMSprop']
    # optimizer = ['RMSprop','Adam']
    hidden_unit = [32]
    # hidden_unit = [16,24,32,64]
    init_mode = ['uniform', "normal"]
    # init_mode = ['uniform', 'lecun_uniform', 'normal', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
    activation = ['relu']
    # activation = ['relu', 'tanh', 'sigmoid']

    # 构建组合，便于搜索
    param_list = []
    for epoch in epochs:
        for im in init_mode:
            for hu in hidden_unit:
                param = dict(batch_size=batch_size[0], epoch=epoch, drop_rate=drop_rate[0], optimizer=optimizer[0],
                          hidden_unit1=hu, hidden_unit2=16, init_mode=im, activation=activation[0])
                param_list.append(param)
    print len(param_list)

    # 对每种参数组合进行交叉验证
    max_f1 = 0
    chosen_param = param_list[0]
    data_num = len(y_train)
    sub_data_num = int(data_num / k)
    for param in param_list:
        print param
        # 数据集分成k份，循环训练测试，得到结果
        avg_f1 = 0
        for index in range(0, k):
            # 划分
            x_test = x_train[sub_data_num*index: sub_data_num*(index+1)]
            y_test = y_train[sub_data_num*index: sub_data_num*(index+1)]
            x_train_new = x_train[:sub_data_num*index] + x_train[sub_data_num*(index+1):]
            y_train_new = y_train[:sub_data_num*index] + y_train[sub_data_num*(index+1):]
            # 训练
            model = network_fit(x_train_new, y_train_new, num_classes, drop_rate=param['drop_rate'], optimizer=param['optimizer'],
                                 hidden_unit1=param['hidden_unit1'], hidden_unit2=param['hidden_unit2'],
                                 activation=param['activation'], init_mode=param['init_mode'], epochs=param['epoch'])
            # 测试
            probas = model.predict(x_test)
            y_predict = predict_by_proba(probas)  # 这里是用于3分类的情况;2类也行，但是就只能测正负
            # 如果要测全部要先分再过滤，先放放
            print y_test
            print y_predict
            f1 = evaluation_3classes(y_test, y_predict)  # 3类的测试评价
            avg_f1 += f1
        avg_f1 /= k
        if max_f1 <= avg_f1:
            max_f1 = avg_f1
            chosen_param = param

    print chosen_param
    return chosen_param


