# coding=utf-8

import random

import keras
import nltk
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential
from nltk.stem.wordnet import WordNetLemmatizer
from pattern.en import sentiment
import numpy as np

from constants import *
from read_file_info_records import *
from evaluation import *
from write_best import *
from find_source import *


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
    str_labels = entity_info_df['label_polarity'].values  # 文件有问题,第一行列数问题，改了不知合不合理
    datanum = len(str_labels)
    # print 'aa', datanum, str_labels
    # labels = [0]  * datanum
    for i in range(datanum):
        if str_labels[i] == 'pos':
            # labels[i] = 1
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


def cnn_fit(x_train, y_train):
    # 训练集进一步划分开发集
    input_shape = x_train.shape[1:]  # 与samples个数无关
    split_at = len(x_train) - len(x_train) // 10  # 这是多少？
    # print split_at
    (x_train_new, x_dev) = x_train[:split_at], x_train[split_at:]
    (y_train_new, y_dev) = y_train[:split_at], y_train[split_at:]

    # 转换标签格式
    y_train_new = keras.utils.to_categorical(y_train_new, 2)
    y_dev = keras.utils.to_categorical(y_dev, 2)
    # print x_train, y_train

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
        if (i[0] < i[1]):  # 可调阈值，过滤0
            y_predict.append([1])
        else:
            y_predict.append([0])
    accuracy = np.mean(y_predict == y_test)
    print("Prediction Accuracy: %.2f%%" % (accuracy * 100))
    return y_predict


def resampling(x, y):
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
    samplenum = int(datanum * 2)
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
    print 'pos vs neg', pos, neg

    return x_new, y_new


# 下面几个函数阈值怎么调合适

def scoring(text):  # 应是有情感词的就分高，不能正负相抵；意在过滤
    score = 0
    sencs = nltk.sent_tokenize(str(text).decode('utf-8'))
    lemm = WordNetLemmatizer()
    words = []
    for senc in sencs:
        words.extend(nltk.word_tokenize(senc))
    for word in words:
        lemmed = lemm.lemmatize(word)
        polarity = sentiment(lemmed)[0]
        if abs(polarity) >= 0.45:
            score += 1
    return score


def context_scoring(contexts):
    scores = []
    for text in contexts:
        scores.append(scoring(text))
    return scores


def predict_by_scores(scores):
    pred = [[0] if score < 1 else [1] for score in scores]
    return pred


def get_train_samples(train_files, embeddings_index, dim, clip_length):
    x_train = []
    y_train = []

    for file_info in train_files:
        if 'entity' in file_info:
            entity_df = file_info['entity']
            entity_df = entity_df[entity_df.label_polarity != 'none'].reset_index()  # none的去掉
            if len(entity_df) != 0:
                x_entity_train, y_entity_train = gen_entity_samples(entity_df, embeddings_index, dim, clip_length)
                x_train.extend(x_entity_train)
                y_train.extend(y_entity_train)
        if 'relation' in file_info:
            relation_df = file_info['relation']
            relation_df = relation_df[relation_df.label_polarity != 'none'].reset_index()
            if len(relation_df) != 0:
                x_relation_train, y_relation_train = gen_relation_samples(relation_df, embeddings_index, dim, clip_length)
                x_train.extend(x_relation_train)
                y_train.extend(y_relation_train)
        if 'event' in file_info:
            event_df = file_info['event']
            event_df = event_df[event_df.label_polarity != 'none'].reset_index()
            if len(event_df) != 0:
                x_event_train, y_event_train = gen_event_samples(event_df, pd.DataFrame(file_info['em_args']),
                                                                 embeddings_index, dim, clip_length)
                x_train.extend(x_event_train)
                y_train.extend(y_event_train)

    # 重采样
    # x_train, y_train = resampling(x_train, y_train)  # 效果不好，因为评测指标是有无算F，不管正负，正负只要对的多就好

    # 转化成单通道输入方式
    x_train, y_train = convert_samples(x_train, y_train, clip_length)

    return x_train, y_train


def test_process(model, test_files, embeddings_index, dim, clip_length):
    y_test = []
    y_predict = []

    for file_info in test_files:
        if 'entity' in file_info:
            x_entity_test, y_entity_test = gen_entity_samples(file_info['entity'],
                                                              embeddings_index, dim, clip_length)
            y_test.extend(y_entity_test)
            x_entity_test, y_entity_test = convert_samples(x_entity_test, y_entity_test, clip_length)

            # 先用词典过滤none，再正负分类
            entity_df = file_info['entity']
            contexts = entity_df['entity_mention_context']
            scores = context_scoring(contexts)
            print 'entity', scores
            y_entity_predict1 = predict_by_scores(scores)  # 过滤none
            y_entity_predict2 = cnn_predict(model, x_entity_test, y_entity_test)
            y_entity_predict2 = [[x[0]+1] for x in y_entity_predict2]  # 全加1
            y_entity_predict = [y_entity_predict2[i] if y_entity_predict1[i] != [0] else y_entity_predict1[i] for i in
                                range(len(y_entity_predict1))]
            # print file_info['filename'], y_entity_predict
            y_predict.extend(y_entity_predict)

            # 加入记录
            file_info['entity'] = entity_df.to_dict(orient='records')
            for i in range(len(file_info['entity'])):
                if y_entity_predict[i] == [2]:
                    file_info['entity'][i]['predict_polarity'] = 'pos'
                elif y_entity_predict[i] == [1]:
                    file_info['entity'][i]['predict_polarity'] = 'neg'
                else:
                    file_info['entity'][i]['predict_polarity'] = 'none'
                    
        if 'relation' in file_info:
            x_relation_test, y_relation_test = gen_relation_samples(file_info['relation'],
                                                                    embeddings_index, dim, clip_length)
            y_test.extend(y_relation_test)
            x_relation_test, y_relation_test = convert_samples(x_relation_test, y_relation_test, clip_length)
            
            # 先用词典过滤none，再正负分类
            relation_df = file_info['relation']
            rel_arg1_contexts = relation_df['rel_arg1_context']
            rel_arg2_contexts = relation_df['rel_arg2_context']
            contexts = []
            for i in range(len(rel_arg1_contexts)):
                context = rel_arg1_contexts[i] + ' ' + rel_arg2_contexts[i]
                contexts.append(context)
            scores = context_scoring(contexts)
            print 'relation', scores
            y_relation_predict1 = predict_by_scores(scores)  # 过滤none
            y_relation_predict2 = cnn_predict(model, x_relation_test, y_relation_test)
            y_relation_predict2 = [[x[0]+1] for x in y_relation_predict2]  # 全加1
            y_relation_predict = [y_relation_predict2[i] if y_relation_predict1[i] != [0] else y_relation_predict1[i] for i in
                                range(len(y_relation_predict1))]
            y_predict.extend(y_relation_predict)

            # 加入记录
            file_info['relation'] = relation_df.to_dict(orient='records')
            for i in range(len(file_info['relation'])):
                if y_relation_predict[i] == [2]:
                    file_info['relation'][i]['predict_polarity'] = 'pos'
                elif y_relation_predict[i] == [1]:
                    file_info['relation'][i]['predict_polarity'] = 'neg'
                else:
                    file_info['relation'][i]['predict_polarity'] = 'none'

        if 'event' in file_info:
            x_event_test, y_event_test = gen_event_samples(file_info['event'],
                                                           file_info['em_args'],
                                                           embeddings_index, dim, clip_length)
            y_test.extend(y_event_test)
            x_event_test, y_event_test = convert_samples(x_event_test, y_event_test, clip_length)

            # 先用词典过滤none，再正负分类
            event_df = file_info['event']
            contexts = event_df['trigger_context']
            scores = context_scoring(contexts)
            print 'event', scores
            y_event_predict1 = predict_by_scores(scores)
            y_event_predict2 = cnn_predict(model, x_event_test, y_event_test)
            y_event_predict2 = [[x[0]+1] for x in y_event_predict2]  # 全加1
            y_event_predict = [y_event_predict2[i] if y_event_predict1[i] != [0] else y_event_predict1[i] for i in
                                range(len(y_event_predict1))]
            y_predict.extend(y_event_predict)

            # 加入记录
            file_info['event'] = event_df.to_dict(orient='records')
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
    print 'Read data...'
    file_info_records = read_file_info_records(ere_dir, entity_info_dir, relation_info_dir, event_info_dir, em_args_dir)

    # 按文件划分训练和测试集
    print 'Split into train and test dataset...'
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
    print 'Train...'
    model = cnn_fit(x_train, y_train)  # 分正负

    # 测试部分
    print 'Test...'
    test_files, y_test, y_predict = test_process(model, test_files, embeddings_index, dim, clip_length)
    # 测试评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # 寻找源
    print 'Find sources... '
    test_files = find_sources(test_files, source_dir, ere_dir)

    # 写入
    print 'Write into best files...'
    write_best_files(test_files, output_dir)

