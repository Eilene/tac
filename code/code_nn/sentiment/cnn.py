# coding=utf-8

import keras
from keras import backend
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.models import Sequential

from utils.evaluation import evaluation_3classes
from utils.filter_none_with_stdict import *
from utils.find_source import find_sources
from utils.read_embedding_index import *
from utils.read_file_info_records import *
# from utils.resampling import *
from utils.write_best import write_best_files
from utils.constants import *
from utils.file_records_other_modification import without_none, to_dict
from utils.get_labels import get_merged_labels
from utils.predict_by_proba import *
from utils.attach_predict_labels import *


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
            # print lemmed, word_vector
        else:
            word_vector = [0.01] * dim
        embeddings_matrix.append(word_vector)

    # 截断补齐
    curr_length = len(embeddings_matrix)
    if curr_length < clip_length:
        embeddings_matrix += [[0.1] * dim] * (clip_length - curr_length)
    if curr_length > clip_length:
        embeddings_matrix = embeddings_matrix[:clip_length]

    return embeddings_matrix


def gen_entity_features(entity_info_df, embeddings_index, dim, clip_length):
    # 特征
    # 上下文词向量矩阵
    features = []
    contexts = entity_info_df['entity_mention_context']
    targets = entity_info_df['entity_mention_text']
    windows = entity_info_df['window_text']
    window_length = 6
    target_length = 10
    context_length = clip_length - window_length - target_length
    # print len(contexts)
    for i in range(len(contexts)):
        embeddings_matrix = gen_embeddings_matrix(contexts[i], context_length, embeddings_index, dim)
        target_matrix = gen_embeddings_matrix(targets[i], target_length, embeddings_index, dim)
        embeddings_matrix.extend(target_matrix)
        window_matrix = gen_embeddings_matrix(windows[i], window_length, embeddings_index, dim)
        embeddings_matrix.extend(window_matrix)
        features.append(embeddings_matrix)

    return features


def gen_relation_features(relation_info_df, embeddings_index, dim, clip_length):
    # 特征
    # 词向量矩阵
    # 两个参数+触发词
    features = []
    rel_arg1_contexts = relation_info_df['rel_arg1_context']
    rel_arg2_contexts = relation_info_df['rel_arg2_context']
    trigger_contexts = relation_info_df['trigger_context']
    trigger_offsets = relation_info_df['trigger_offset']
    rel_arg1_texts = relation_info_df['rel_arg1_text']
    rel_arg2_texts = relation_info_df['rel_arg2_text']
    trigger_texts = relation_info_df['trigger_text']
    rel_arg1_windows = relation_info_df['rel_arg1_window_text']
    rel_arg2_windows = relation_info_df['rel_arg1_window_text']
    # trigger_windows = relation_info_df['trigger_window_text']
    window_length = 6
    target_length = 10
    context_length = clip_length - window_length - target_length
    for i in range(len(rel_arg1_contexts)):
        # 参数1
        embeddings_matrix1 = gen_embeddings_matrix(rel_arg1_contexts[i], context_length, embeddings_index, dim)
        target_matrix1 = gen_embeddings_matrix(rel_arg1_texts[i], target_length, embeddings_index, dim)
        embeddings_matrix1.extend(target_matrix1)
        window_matrix1 = gen_embeddings_matrix(rel_arg1_windows[i], window_length, embeddings_index, dim)
        embeddings_matrix1.extend(window_matrix1)
        # 参数2
        embeddings_matrix2 = gen_embeddings_matrix(rel_arg2_contexts[i], context_length, embeddings_index, dim)
        target_matrix2 = gen_embeddings_matrix(rel_arg2_texts[i], target_length, embeddings_index, dim)
        embeddings_matrix2.extend(target_matrix2)
        window_matrix2 = gen_embeddings_matrix(rel_arg2_windows[i], window_length, embeddings_index, dim)
        embeddings_matrix2.extend(window_matrix2)
        # 触发词
        embeddings_matrix3 = []
        if int(trigger_offsets[i]) == 0:
            word_vector = [0.01] * dim
            for j in range(clip_length):
                embeddings_matrix3.append(word_vector)
            for j in range(16):
                embeddings_matrix3.append(word_vector)
        else:
            embeddings_matrix3 = gen_embeddings_matrix(trigger_contexts[i], context_length, embeddings_index, dim)
            target_matrix3 = gen_embeddings_matrix(trigger_texts[i], target_length, embeddings_index, dim)
            embeddings_matrix3.extend(target_matrix3)
            # window_matrix3 = gen_embeddings_matrix(trigger_windows[i], window_length, embeddings_index, dim)
            # embeddings_matrix3.extend(window_matrix3)
            # trigger暂没有提窗口特征
            word_vector = [0.01] * dim
            for j in range(window_length):
                embeddings_matrix3.append(word_vector)
        # 合并
        embeddings_matrix = embeddings_matrix1
        for k in range(clip_length):
            for j in range(dim):
                embeddings_matrix[k][j] += embeddings_matrix2[k][j]
                embeddings_matrix[k][j] += embeddings_matrix3[k][j]
                embeddings_matrix[k][j] /= 3

        features.append(embeddings_matrix)

    return features


def gen_event_features(event_info_df, em_args_info_df, embeddings_index, dim, clip_length):
    # 特征
    # 词向量矩阵
    # 触发词+各个参数
    features = []
    trigger_contexts = event_info_df['trigger_context']
    trigger_texts = event_info_df['trigger_text']
    trigger_windows = event_info_df['trigger_window_text']
    window_length = 6
    target_length = 10
    context_length = clip_length - window_length - target_length
    for i in range(len(trigger_contexts)):
        # 触发词
        embeddings_matrix3 = gen_embeddings_matrix(trigger_contexts[i], context_length, embeddings_index, dim)
        target_matrix3 = gen_embeddings_matrix(trigger_texts[i], target_length, embeddings_index, dim)
        embeddings_matrix3.extend(target_matrix3)
        window_matrix3 = gen_embeddings_matrix(trigger_windows[i], window_length, embeddings_index, dim)
        embeddings_matrix3.extend(window_matrix3)
        # 各个参数（似乎上下文都一样，取一个即可）
        # 合并
        embeddings_matrix = embeddings_matrix3

        features.append(embeddings_matrix)

    return features


def convert_samples(features, labels):
    labels = np.array(labels)
    # print 'Label shape: ', labels.shape
    matrix_length = len(features[0])
    count = 0
    print backend.image_dim_ordering()
    if backend.image_dim_ordering() == 'th':  # 竟然输出是tf，且倒过来真报错
        data = np.empty((labels.shape[0], 1, matrix_length, 100), dtype='float32')
        # Train_X = sequence.pad_sequences(Train_X, maxlen=Sentence_length)
        for feature in features:
            data[count, 0, :, :] = feature  # 通道维顺序？？不同后端不同，是不是要改？？
            count += 1
            features = data
    else:
        data = np.empty((labels.shape[0], matrix_length, 100, 1), dtype='float32')
        # Train_X = sequence.pad_sequences(Train_X, maxlen=Sentence_length)
        for feature in features:
            data[count, :, :, 0] = feature
            count += 1
            features = data

    print 'Features shape:', features.shape

    return features, labels


def gen_cnn_features(file_records, embeddings_index, dim, clip_length):
    features = []

    for file_info in file_records:
        if 'entity' in file_info:
            entity_df = file_info['entity']
            if len(entity_df) != 0:
                x_entity = gen_entity_features(entity_df, embeddings_index, dim, clip_length)
                features.extend(x_entity)
        if 'relation' in file_info:
            relation_df = file_info['relation']
            if len(relation_df) != 0:
                x_relation = gen_relation_features(relation_df, embeddings_index, dim, clip_length)
                features.extend(x_relation)
        if 'event' in file_info:
            event_df = file_info['event']
            if len(event_df) != 0:
                x_event = gen_event_features(event_df, file_info['em_args'], embeddings_index, dim, clip_length)
                features.extend(x_event)

    return features


def cnn_fit(x_train, y_train, classnum):
    # 训练集进一步划分开发集
    input_shape = x_train.shape[1:]  # 与samples个数无关
    split_at = len(x_train) - len(x_train) // 10  # 这是多少？
    # print split_at
    (x_train_new, x_dev) = x_train[:split_at], x_train[split_at:]
    (y_train_new, y_dev) = y_train[:split_at], y_train[split_at:]

    # 转换标签格式
    # print y_train_new
    y_train_new = keras.utils.to_categorical(y_train_new, classnum)
    y_dev = keras.utils.to_categorical(y_dev, classnum)
    # print y_train_new
    # print y_train_new.shape

    # 开始建立CNN模型
    batch_size = 128
    epochs = 5

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
    model.add(Dense(classnum, activation='softmax'))
    # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss=keras.losses.binary_crossentropy, optimizer='Adam', metrics=['accuracy'])
    model.summary()

    model.fit(x_train_new, y_train_new, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_dev, y_dev))
    score = model.evaluate(x_dev, y_dev, verbose=0)
    print 'Test loss:', score[0]
    print 'Test accuracy:', score[1]

    return model


if __name__ == '__main__':
    mode = True  # True:DF,false:NW

    # 读取各文件中间信息
    print 'Read data...'
    df_file_records, nw_file_records = \
        read_file_info_records(ere_dir, entity_info_dir, relation_info_dir, event_info_dir, em_args_dir)
    print 'DF files:', len(df_file_records), ' NW files:', len(nw_file_records)

    # DF全部作为训练数据，NW分成训练和测试数据, 合并训练的NW和DF，即可用原来流程进行训练测试
    if mode is True:
        print '*** DF ***'
        print 'Split into train and test dataset...'
        portion = 0.8
        trainnum = int(len(df_file_records) * 0.8)
        train_files = df_file_records[:trainnum]
        test_files = df_file_records[trainnum:]
    else:
        print '*** NW ***'
        print 'Merge and split into train and test dataset...'
        portion = 0.2
        nw_trainnum = int(len(nw_file_records) * portion)
        train_files = df_file_records + nw_file_records[:nw_trainnum]
        test_files = nw_file_records[nw_trainnum:]

    # 训练部分
    # 提取特征及标签
    total_clip_length = 56
    embeddings_index, dim = read_embedding_index(glove_100d_path)
    print 'Train samples extraction...'
    train_files = without_none(train_files)  # 训练文件去掉none的样本
    x_train = gen_cnn_features(train_files, embeddings_index, dim, total_clip_length)  # 提取特征
    y_train = get_merged_labels(train_files)  # 只有1,2两类
    y_train = [y-1 for y in y_train]  # 改为0,1
    x_train, y_train = convert_samples(x_train, y_train)  # 转换为通道模式
    # 训练
    print 'Train...'
    model = cnn_fit(x_train, y_train, 2)  # 分正负

    # 测试部分
    # 提取特征及标签
    print 'Test samples extraction...'
    x_test = gen_cnn_features(test_files, embeddings_index, dim, total_clip_length)  # 提取特征
    y_test = get_merged_labels(test_files)  # 0,1,2三类
    x_test, y_test = convert_samples(x_test, y_test)
    # 测试
    print 'Test...'
    probabilities = model.predict(x_test)
    y_predict = predict_by_proba(probabilities, 0.1)

    # 评价
    y_test = y_test.tolist()
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # 测试结果写入记录
    test_files = to_dict(test_files)
    test_files = attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    test_files = find_sources(test_files, source_dir, ere_dir)
    # test_files = use_annotation_source(test_files)

    # 写入
    print 'Write into best files...'
    write_best_files(test_files, predict_dir)

