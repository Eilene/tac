# coding=utf-8
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from keras import backend
import numpy as np

from pattern.en import lemma
from pattern.en import sentiment

# def gen_external_features(contexts, embeddings_index, dim, clip_length):
#     features = []
#     for i in range(len(contexts)):
#         embeddings_matrix = gen_embeddings_matrix(contexts[i], clip_length, embeddings_index, dim)
#         features.append(embeddings_matrix)
#     return features


# 增加每个词的词性特征？
# 一个上下文的情感词计数等特征
# target的实体等特征


# 词性计数，情感词计数
def senti_count(texts):
    senti_count_list = []
    for i in range(len(texts)):
        # 词性
        words = nltk.word_tokenize(texts[i])
        pos_senti_count = 0
        neg_senti_count = 0
        for word in words:
            lemmed = lemma(word)
            polarity = sentiment(lemmed)[0]
            # print lemmed, polarity
            # if polarity >= 0.1:  # 这个阈值？？  # 或者直接累值？？
            #     pos_senti_count += 1
            # elif polarity <= -0.1:
            #     neg_senti_count += 1
            if polarity > 0:
                pos_senti_count += polarity
            elif polarity < 0:
                neg_senti_count += polarity
        senti_count_list.append([pos_senti_count, neg_senti_count])

    return senti_count_list


def gen_embeddings_matrix(context, clip_length, embeddings_index, dim):
    embeddings_matrix = []

    # 添加词向量，生成矩阵
    lemm = WordNetLemmatizer()
    sencs = nltk.sent_tokenize(str(context).decode('utf-8'))
    words = []
    for senc in sencs:
        words.extend(nltk.word_tokenize(senc))
    count = 0
    for word in words:
        lemmed = lemm.lemmatize(word)
        if lemmed in embeddings_index:
            word_vector = embeddings_index[lemmed]
            # print lemmed, word_vector
        else:
            word_vector = [0.01] * dim
            count += 1
        embeddings_matrix.append(word_vector)

        # 每个词再加点什么

    # print 'total, unknown word: ', len(words), count

    # 截断补齐
    curr_length = len(embeddings_matrix)
    # print curr_length
    if curr_length < clip_length:
        embeddings_matrix += [[0.1] * dim] * (clip_length - curr_length)
    if curr_length > clip_length:
        embeddings_matrix = embeddings_matrix[:clip_length]

    return embeddings_matrix


def gen_entity_features(entity_info_df, embeddings_index, dim, clip_length):
    # 特征
    # 上下文词向量矩阵
    features = []
    contexts = entity_info_df['entity_mention_context3']
    targets = entity_info_df['entity_mention_text']
    windows = entity_info_df['window_text']
    window_length = 0
    target_length = 0
    context_length = clip_length - window_length - target_length
    # print len(contexts)
    for i in range(len(contexts)):
        embeddings_matrix = gen_embeddings_matrix(contexts[i], context_length, embeddings_index, dim)
        # target_matrix = gen_embeddings_matrix(targets[i], target_length, embeddings_index, dim)
        # embeddings_matrix.extend(target_matrix)
        # window_matrix = gen_embeddings_matrix(windows[i], window_length, embeddings_index, dim)
        # embeddings_matrix.extend(window_matrix)

        features.append(embeddings_matrix)

    # 后面加上情感词计数，类别特征等
    # 不行，类别特征需要整体提取再独热，这样分开不行
    senti_count_list = senti_count(contexts)
    for i in range(len(features)):
        pos_vec = senti_count_list[i]
        # print pos_vec
        length = len(pos_vec)
        if length < dim:
            pos_vec.extend([0.1] * (dim-length))
        else:
            pos_vec = pos_vec[:dim]
        # print pos_vec
        features[i].append(pos_vec)

    return features


def gen_relation_features(relation_info_df, embeddings_index, dim, clip_length):
    # 特征
    # 词向量矩阵
    # 两个参数+触发词
    features = []
    rel_arg1_contexts = relation_info_df['rel_arg1_context3']
    rel_arg2_contexts = relation_info_df['rel_arg2_context3']
    trigger_contexts = relation_info_df['trigger_context']
    trigger_offsets = relation_info_df['trigger_offset']
    rel_arg1_texts = relation_info_df['rel_arg1_text']
    rel_arg2_texts = relation_info_df['rel_arg2_text']
    trigger_texts = relation_info_df['trigger_text']
    rel_arg1_windows = relation_info_df['rel_arg1_window_text']
    rel_arg2_windows = relation_info_df['rel_arg1_window_text']
    # trigger_windows = relation_info_df['trigger_window_text']
    window_length = 0
    target_length = 0
    context_length = clip_length - window_length - target_length
    for i in range(len(rel_arg1_contexts)):
        # 参数1
        embeddings_matrix1 = gen_embeddings_matrix(rel_arg1_contexts[i], context_length, embeddings_index, dim)
        # target_matrix1 = gen_embeddings_matrix(rel_arg1_texts[i], target_length, embeddings_index, dim)
        # embeddings_matrix1.extend(target_matrix1)
        # window_matrix1 = gen_embeddings_matrix(rel_arg1_windows[i], window_length, embeddings_index, dim)
        # embeddings_matrix1.extend(window_matrix1)
        # 参数2
        embeddings_matrix2 = gen_embeddings_matrix(rel_arg2_contexts[i], context_length, embeddings_index, dim)
        # target_matrix2 = gen_embeddings_matrix(rel_arg2_texts[i], target_length, embeddings_index, dim)
        # embeddings_matrix2.extend(target_matrix2)
        # window_matrix2 = gen_embeddings_matrix(rel_arg2_windows[i], window_length, embeddings_index, dim)
        # embeddings_matrix2.extend(window_matrix2)
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
            # target_matrix3 = gen_embeddings_matrix(trigger_texts[i], target_length, embeddings_index, dim)
            # embeddings_matrix3.extend(target_matrix3)
            # window_matrix3 = gen_embeddings_matrix(trigger_windows[i], window_length, embeddings_index, dim)
            # embeddings_matrix3.extend(window_matrix3)
            # trigger暂没有提窗口特征
            # word_vector = [0.01] * dim
            # for j in range(window_length):
            #     embeddings_matrix3.append(word_vector)
        # 合并
        embeddings_matrix = embeddings_matrix1
        for k in range(clip_length):
            for j in range(dim):
                embeddings_matrix[k][j] += embeddings_matrix2[k][j]
                embeddings_matrix[k][j] += embeddings_matrix3[k][j]
                embeddings_matrix[k][j] /= 3

        features.append(embeddings_matrix)

    # 后面加上情感词计数，类别特征等
    senti_count_list = senti_count(rel_arg1_contexts)
    for i in range(len(features)):
        pos_vec = senti_count_list[i]
        length = len(pos_vec)
        if length < dim:
            pos_vec.extend([0.1] * (dim-length))
        else:
            pos_vec = pos_vec[:dim]
        features[i].append(pos_vec)

    return features


def gen_event_features(event_info_df, em_args_info_df, embeddings_index, dim, clip_length):
    # 特征
    # 词向量矩阵
    # 触发词+各个参数
    features = []
    trigger_contexts = event_info_df['trigger_context3']
    trigger_texts = event_info_df['trigger_text']
    trigger_windows = event_info_df['trigger_window_text']
    window_length = 0
    target_length = 0
    context_length = clip_length - window_length - target_length
    for i in range(len(trigger_contexts)):
        # 触发词
        embeddings_matrix3 = gen_embeddings_matrix(trigger_contexts[i], context_length, embeddings_index, dim)
        # target_matrix3 = gen_embeddings_matrix(trigger_texts[i], target_length, embeddings_index, dim)
        # embeddings_matrix3.extend(target_matrix3)
        # window_matrix3 = gen_embeddings_matrix(trigger_windows[i], window_length, embeddings_index, dim)
        # embeddings_matrix3.extend(window_matrix3)
        # 各个参数（似乎上下文都一样，取一个即可）
        # 合并
        embeddings_matrix = embeddings_matrix3

        features.append(embeddings_matrix)

    # 后面加上情感词计数，类别特征等
    senti_count_list = senti_count(trigger_contexts)
    for i in range(len(features)):
        pos_vec = senti_count_list[i]
        length = len(pos_vec)
        if length < dim:
            pos_vec.extend([0.1] * (dim-length))
        else:
            pos_vec = pos_vec[:dim]
        features[i].append(pos_vec)
    # 好像还可以

    return features


def convert_features(features):
    sample_num = len(features)
    matrix_length = len(features[0])
    count = 0
    dim = len(features[0][0])
    print matrix_length
    # print backend.image_dim_ordering()
    if backend.image_dim_ordering() == 'th':  # 竟然输出是tf，且倒过来真报错
        data = np.empty((sample_num, 1, matrix_length, dim), dtype='float32')
        # Train_X = sequence.pad_sequences(Train_X, maxlen=Sentence_length)
        for feature in features:
            data[count, 0, :, :] = feature  # 通道维顺序？？不同后端不同，是不是要改？？
            count += 1
            features = data
    else:
        data = np.empty((sample_num, matrix_length, dim, 1), dtype='float32')
        # Train_X = sequence.pad_sequences(Train_X, maxlen=Sentence_length)
        for feature in features:
            data[count, :, :, 0] = feature
            count += 1
            features = data

    print 'Features shape:', features.shape

    return features


def gen_matrix_features(file_records, embeddings_index, dim, clip_length):
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
