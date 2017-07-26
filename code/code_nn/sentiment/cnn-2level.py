# coding=utf-8

import nltk
from nltk.stem.wordnet import WordNetLemmatizer
# import numpy as np

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
# from keras.models import load_model

from constants import *
from read_file_info_records import *
from write_best import *
from evaluation import *

import random
import copy


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
    sencs = nltk.sent_tokenize(str(context).decode('utf-8'))
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
    data = np.empty((labels.shape[0], clip_length, 100, 1), dtype='float32')
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
            labels.append([2])
        elif str_labels[i] == 'neg':
            labels.append([1])
        else:
            labels.append([0])

    # 特征
    # 上下文词向量矩阵
    features = []
    contexts = entity_info_df['entity_mention_context']
    # print len(contexts)
    for i in range(len(contexts)):
        embeddings_matrix = gen_embeddings_matrix(contexts[i], clip_length, embeddings_index, dim)
        features.append(embeddings_matrix)

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
            labels.append([2])
        elif str_labels[i] == 'neg':
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
            labels.append([2])
        elif str_labels[i] == 'neg':
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

    return features, labels


def cnn_fit(train_x, train_y, mode):
    x_t = copy.deepcopy(train_x)
    y_t = copy.deepcopy(train_y)
    samplenum = len(x_t)
    if mode == 1:
        print train_y
        samplenum = 0
        for i in range(len(y_t)):
            if y_t[i] != [0]:
                y_t[i] = [1]
                samplenum += 1
        samplenum *= 2
        print train_y
    else:
        print train_y
        x_train_pn = []
        y_train_pn = []
        for i in range(len(train_y)):
            if train_y[i] != [0]:
                x_train_pn.append(train_x[i])
                y_train_pn.append([train_y[i][0]-1])
        x_t = x_train_pn
        y_t = y_train_pn
        samplenum = len(y_t) * 2
        # print y_t
        # print len(y_t)

    # 重采样
    x_t, y_t = resampling(x_t, y_t, samplenum)  # 用于有无的好

    # 转化成单通道输入方式
    x_t, y_t = convert_samples(x_t, y_t, clip_length)

    # 训练集进一步划分开发集
    input_shape = x_t.shape[1:]  # 与samples个数无关
    split_at = len(x_t) - len(x_t) // 10  # 这是多少？
    # print split_at
    (x_train_new, x_dev) = x_t[:split_at], x_t[split_at:]
    (y_train_new, y_dev) = y_t[:split_at], y_t[split_at:]

    # 转换标签格式
    y_train_new = keras.utils.to_categorical(y_train_new, 2)
    y_dev = keras.utils.to_categorical(y_dev, 2)
    # print x_t, y_t

    # 开始建立CNN模型
    batch_size = 128
    epochs = 3

    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=keras.losses.binary_crossentropy, optimizer='Adam', metrics=['accuracy'])
    model.summary()

    model.fit(x_train_new, y_train_new, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_dev, y_dev))
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


def get_train_samples(train_files, embeddings_index, dim, clip_length):
    x_train = []
    y_train = []

    for file_info in train_files:
        if 'entity' in file_info:
            x_entity_train, y_entity_train = gen_entity_samples(pd.DataFrame(file_info['entity']),
                                                                embeddings_index, dim, clip_length)
            x_train.extend(x_entity_train)
            y_train.extend(y_entity_train)
        if 'relation' in file_info:
            x_relation_train, y_relation_train = gen_relation_samples(pd.DataFrame(file_info['relation']),
                                                                      embeddings_index, dim, clip_length)
            x_train.extend(x_relation_train)
            y_train.extend(y_relation_train)
        if 'event' in file_info:
            x_event_train, y_event_train = gen_event_samples(pd.DataFrame(file_info['event']),
                                                             pd.DataFrame(file_info['em_args']),
                                                             embeddings_index, dim, clip_length)
            x_train.extend(x_event_train)
            y_train.extend(y_event_train)

    return x_train, y_train


def resampling(x, y, samplenum):
    x_new = []
    y_new = []

    # 正负样本分开
    pos_index = []
    neg_index = []
    datanum = len(y)
    for i in range(datanum):
        if y[i] == [1]:
            pos_index.append(i)
        else:
            neg_index.append(i)

    # 正负样本均衡采样，有放回采样
    pos = 0
    neg = 0
    pos_num = len(pos_index)
    neg_num = len(neg_index)
    for i in range(samplenum):
        flag = random.randint(1, 2)
        if flag == 1:
            index = random.randint(0, pos_num-1)
            x_new.append(x[pos_index[index]])
            y_new.append(y[pos_index[index]])
            pos += 1
        else:
            index = random.randint(0, neg_num-1)
            x_new.append(x[neg_index[index]])
            y_new.append(y[neg_index[index]])
            neg += 1
    print 'st vs none (pos vs neg): ', pos, neg

    return x_new, y_new


def test_process(model1, model2, test_files, embeddings_index, dim, clip_length):
    y_test = []
    y_predict = []

    for file_info in test_files:
        if 'entity' in file_info:
            x_entity_test, y_entity_test = gen_entity_samples(pd.DataFrame(file_info['entity']),
                                                              embeddings_index, dim, clip_length)
            y_test.extend(y_entity_test)
            x_entity_test, y_entity_test = convert_samples(x_entity_test, y_entity_test, clip_length)
            # 两层模型
            y_entity_predict1 = cnn_predict(model1, x_entity_test, y_entity_test)
            y_entity_predict2 = cnn_predict(model2, x_entity_test, y_entity_test)
            y_entity_predict2 = [[x[0]+1] for x in y_entity_predict2]  # 全加1
            y_entity_predict = [y_entity_predict2[i] if y_entity_predict1[i] != 0 else y_entity_predict1[i] for i in 
                                range(len(y_entity_predict1))]
            print file_info['filename'], y_entity_predict
            y_predict.extend(y_entity_predict)
            for i in range(len(file_info['entity'])):
                if y_entity_predict[i] == [2]:
                    file_info['entity'][i]['predict_polarity'] = 'pos'
                elif y_entity_predict[i] == [1]:
                    file_info['entity'][i]['predict_polarity'] = 'neg'
                else:
                    file_info['entity'][i]['predict_polarity'] = 'none'
        if 'relation' in file_info:
            x_relation_test, y_relation_test = gen_relation_samples(pd.DataFrame(file_info['relation']),
                                                                    embeddings_index, dim, clip_length)
            y_test.extend(y_relation_test)
            x_relation_test, y_relation_test = convert_samples(x_relation_test, y_relation_test, clip_length)
            # 两层模型
            y_relation_predict1 = cnn_predict(model1, x_relation_test, y_relation_test)
            y_relation_predict2 = cnn_predict(model2, x_relation_test, y_relation_test)
            y_relation_predict2 = [[x[0]+1] for x in y_relation_predict2]  # 全加1
            y_relation_predict = [y_relation_predict2[i] if y_relation_predict1[i] != 0 else y_relation_predict1[i] for i in 
                                range(len(y_relation_predict1))]
            y_predict.extend(y_relation_predict)
            for i in range(len(file_info['relation'])):
                if y_relation_predict[i] == [2]:
                    file_info['relation'][i]['predict_polarity'] = 'pos'
                elif y_relation_predict[i] == [1]:
                    file_info['relation'][i]['predict_polarity'] = 'neg'
                else:
                    file_info['relation'][i]['predict_polarity'] = 'none'
        if 'event' in file_info:
            x_event_test, y_event_test = gen_event_samples(pd.DataFrame(file_info['event']),
                                                           pd.DataFrame(file_info['em_args']),
                                                           embeddings_index, dim, clip_length)
            y_test.extend(y_event_test)
            x_event_test, y_event_test = convert_samples(x_event_test, y_event_test, clip_length)
            # 两层模型
            y_event_predict1 = cnn_predict(model1, x_event_test, y_event_test)
            y_event_predict2 = cnn_predict(model2, x_event_test, y_event_test)
            y_event_predict2 = [[x[0]+1] for x in y_event_predict2]  # 全加1
            y_event_predict = [y_event_predict2[i] if y_event_predict1[i] != 0 else y_event_predict1[i] for i in 
                                range(len(y_event_predict1))]
            y_predict.extend(y_event_predict)
            for i in range(len(file_info['event'])):
                if y_event_predict[i] == [2]:
                    file_info['event'][i]['predict_polarity'] = 'pos'
                elif y_event_predict[i] == [1]:
                    file_info['event'][i]['predict_polarity'] = 'neg'
                else:
                    file_info['event'][i]['predict_polarity'] = 'none'

    return test_files, y_test, y_predict


if __name__ == '__main__':
    # 读取各文件中间信息
    file_info_records = read_file_info_records(ere_dir, entity_info_dir, relation_info_dir, event_info_dir, em_args_dir)

    # 按文件划分训练和测试集
    portion = 0.8
    trainnum = int(len(file_info_records) * 0.8)
    train_files = file_info_records[:trainnum]
    test_files = file_info_records[trainnum:]

    # 训练部分
    # 提取特征，生成样本
    clip_length = 40
    embeddings_index, dim = read_embedding_index(glove_100d_path)
    print 'Train samples extraction...'
    x_train, y_train = get_train_samples(train_files, embeddings_index, dim, clip_length)
    print 'Train data number:', len(y_train)
    # cnn训练
    model1 = cnn_fit(x_train, y_train, 1)  # 分有无
    model2 = cnn_fit(x_train, y_train, 2)  # 分正负

    # 测试部分
    # 提取特征
    test_files, y_test, y_predict = test_process(model1, model2, test_files, embeddings_index, dim, clip_length)
    # 有无的评价
    y_test_none = y_test
    y_predict_none = y_predict
    for i in range(len(y_test)):
        if y_test_none[i] != [0]:
            y_test_none[i] = [1]
        if y_predict_none[i] != [0]:
            y_predict_none[i] = [1]
    evaluation(y_test_none, y_predict_none)

    # 写入
    write_best_files(test_files, output_dir)

