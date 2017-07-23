# coding=utf-8

from sklearn import metrics
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
import numpy as np

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# from keras.models import load_model


def read_embedding_index(filename):
    embeddings_index = {}
    dim = 100
    embedding_vectors_fp = open(filename)
    for line in embedding_vectors_fp:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    embedding_vectors_fp.close()
    # print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index, dim


def gen_embeddings_matrix(context, clip_length, embeddings_index, dim):
    embeddings_matrix = []

    # 添加词向量，生成矩阵
    lemm = WordNetLemmatizer()
    sencs = nltk.sent_tokenize(context.decode('utf-8'))
    words = []
    for senc in sencs:
        words.extend(nltk.word_tokenize(senc))
    for word in words:
        lemmed = lemm.lemmatize(word)
        if lemmed in embeddings_index:
            word_vector = embeddings_index[lemmed]
        else:
            word_vector = [0.01] * dim
        embeddings_matrix.append(word_vector)

    # 截断补齐
    curr_length = len(embeddings_matrix)
    if curr_length < clip_length:
        embeddings_matrix += [[0.1] * 100] * (clip_length - curr_length)
    if curr_length > clip_length:
        embeddings_matrix = embeddings_matrix[:clip_length]

    return embeddings_matrix


def convert_samples(features, labels, clip_length):
    labels = np.array(labels)
    print 'Label shape: ', labels.shape
    matrix_length = clip_length
    data = np.empty((labels.shape[0], matrix_length, 100, 1), dtype='float32')
    # Train_X = sequence.pad_sequences(Train_X, maxlen=Sentence_length)
    count = 0
    for feature in features:
        data[count, :, :, 0] = feature
        count += 1
        features = data

    return features, labels


def gen_entity_samples(entity_info_df, embeddings_index, dim, clip_length):
    # 标签
    labels = []
    str_labels = entity_info_df['label_polarity']  # 文件有问题,第一行列数问题，改了不知合不合理
    datanum = len(str_labels)
    # print str_labels
    # print datanum
    # labels = [0]  * datanum
    for i in range(datanum):
        # print str_labels[i]
        if str_labels[i] == 'pos':
            # labels[i] = 1
            labels.append([1])
        else:
            labels.append([0])

    # 特征
    # 上下文词向量矩阵
    features = []
    contexts = entity_info_df['content']
    # print len(contexts)
    for i in range(len(contexts)):
        embeddings_matrix = gen_embeddings_matrix(contexts[i], clip_length, embeddings_index, dim)
        features.append(embeddings_matrix)

    # 转化成单通道输入方式
    features, labels = convert_samples(features, labels, clip_length)

    return features, labels


def gen_relation_samples(relation_info_df, embeddings_index, dim, clip_length):
    # 标签
    labels = []
    str_labels = relation_info_df['label_polarity']
    datanum = len(str_labels)
    # labels = [0] * datanum
    for i in range(datanum):
        if str_labels[i] == 'pos':
            # labels[i] = 1
            labels.append([1])
        else:
            labels.append([0])

    # 特征
    # 词向量矩阵
    # 两个参数+触发词
    features = []
    rel_arg1_contexts = relation_info_df['rel_arg1_context']
    rel_arg2_contexts = relation_info_df['rel_arg2_context']
    trigger_contexts = relation_info_df['trigger_context']
    for i in range(len(rel_arg1_contexts)):
        # 参数1
        embeddings_matrix1 = gen_embeddings_matrix(rel_arg1_contexts[i], clip_length, embeddings_index, dim)
        # 参数2
        embeddings_matrix2 = gen_embeddings_matrix(rel_arg2_contexts[i], clip_length, embeddings_index, dim)
        # 触发词
        embeddings_matrix3 = []
        if trigger_contexts[i] == 'None':
            word_vector = [0.01] * dim
            for j in range(clip_length):
                embeddings_matrix3.append(word_vector)
        else:
            embeddings_matrix3 = gen_embeddings_matrix(trigger_contexts[i], clip_length, embeddings_index, dim)
        # 合并
        embeddings_matrix = embeddings_matrix1
        for k in range(clip_length):
            for j in range(dim):
                embeddings_matrix[k][j] += embeddings_matrix2[k][j]
                embeddings_matrix[k][j] += embeddings_matrix3[k][j]
                embeddings_matrix[k][j] /= 3

        features.append(embeddings_matrix)

    # 转化成单通道输入方式
    features, labels = convert_samples(features, labels, clip_length)

    return features, labels


def gen_event_samples(event_info_df, em_args_info_df, embeddings_index, dim, clip_length):
    # 标签
    labels = []
    str_labels = event_info_df['label_polarity']
    datanum = len(str_labels)
    # labels = [0] * datanum
    for i in range(datanum):
        if str_labels[i] == 'pos':
            # labels[i] = 1
            labels.append([1])
        else:
            labels.append([0])

    # 特征
    # 词向量矩阵
    # 触发词+各个参数
    features = []
    trigger_contexts = event_info_df['trigger_context']

    for i in range(len(trigger_contexts)):
        # 触发词
        embeddings_matrix3 = gen_embeddings_matrix(trigger_contexts[i], clip_length, embeddings_index, dim)
        # 各个参数（似乎上下文都一样，取一个即可）
        # 合并
        embeddings_matrix = embeddings_matrix3

        features.append(embeddings_matrix)

    # 转化成单通道输入方式
    features, labels = convert_samples(features, labels, clip_length)

    return features, labels


# def split_data(x, y, portion):
#     train_num = int(len(y) * portion)
#     x_train = x[:train_num]
#     x_test = x[train_num:]
#     y_train = y[:train_num]
#     y_test = y[train_num:]
#     return x_train, y_train, x_test, y_test


# def shuffle(x, y):
#     data = np.array(x)
#     print data.shape
#     data = np.hstack((data, y))
#     
#     return x, y


def evaluation(y_test, y_predict):
    accuracy = metrics.accuracy_score(y_test, y_predict)
    precision = metrics.precision_score(y_test, y_predict)
    recall = metrics.recall_score(y_test, y_predict)
    f1 = metrics.f1_score(y_test, y_predict)
    print "Accuracy: ", accuracy
    print "Precision: ", precision
    print "Recall: ", recall
    print "F1: ", f1


def cnn_fit(x_train, y_train):
    # 训练集进一步划分开发集
    input_shape = x_train.shape[1:]  # 与samples个数无关
    split_at = len(x_train) - len(x_train) // 10  # 这是多少？
    # print split_at
    (x_train_new, x_dev) = x_train[:split_at], x_train[split_at:]
    (y_train_new, y_dev) = y_train[:split_at], y_train[split_at:]

    # 转换标签格式
    y_train = keras.utils.to_categorical(y_train, 2)
    y_dev = keras.utils.to_categorical(y_dev, 2)
    # print x_train, y_train

    # 开始建立CNN模型
    batch_size = 128
    epochs = 3

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=keras.losses.binary_crossentropy, optimizer='Adam', metrics=['accuracy'])
    model.summary()

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_dev, y_dev))
    score = model.evaluate(x_dev, y_dev, verbose=0)
    print 'Test loss:', score[0]
    print 'Test accuracy:', score[1]

    return model


def cnn_predict(model, x_test, y_test):
    probabilities = model.predict(x_test)
    y_predict = []
    # print probabilities
    for i in probabilities:
        if (i[0] < i[1]):
            y_predict.append([1])
        else:
            y_predict.append([0])
    accuracy = np.mean(y_predict == y_test)
    print("Prediction Accuracy: %.2f%%" % (accuracy * 100))
    return y_predict

