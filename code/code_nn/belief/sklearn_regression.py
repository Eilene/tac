# coding=utf-8

from sklearn import linear_model

from utils.attach_predict_labels import attach_predict_labels
from utils.constants import *
from utils.file_records_other_modification import to_dict
from utils.find_source import find_sources
from utils.get_labels import get_merged_labels
from utils.read_file_info_records import *
from utils.write_best import write_best_files
from utils.evaluation import evaluation

import nltk
from pattern.en import lemma
from pattern.en import sentiment
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


def gen_general_features(file_records):

    # 按文件顺序合并
    contexts = []
    texts = []
    types = pd.DataFrame(columns=['entity_type', 'entity_mention_noun_type', 'relation_type', 'relation_subtype',
                                  'event_mention_type', 'event_mention_subtype', 'rel_arg1_entity_type',
                                  'rel_arg1_mention_noun_type', 'rel_arg2_entity_type', 'rel_arg2_mention_noun_type'])
    # 未独热前：entity_type, entity_mention_noun_type, relation_type, relation_subtype, event_mention_type,
    # event_mention_subtype
    # rel_arg1_entity_type, rel_arg1_mention_noun_type, rel_arg2_entity_type, rel_arg2_mention_noun_type
    # 这几个如何与entity的共用？
    # 可所有的都加进来

    # (暂时只有target、sentence(context)层次，post和file层次待补，可能需中间文件补)

    for file_info in file_records:
        if 'entity' in file_info:
            entity_df = file_info['entity']
            # print len(entity_df)  # 0也没关系，就正好是没加
            entity_contexts = entity_df['entity_mention_context']
            entity_texts = entity_df['entity_mention_text']
            contexts.extend(entity_contexts.tolist())
            texts.extend(entity_texts.tolist())

            entity_types = entity_df[['entity_type', 'entity_mention_noun_type']].copy()
            entity_types['relation_type'] = ''
            entity_types['relation_subtype'] = ''
            entity_types['event_mention_type'] = ''
            entity_types['event_mention_subtype'] = ''
            entity_types['rel_arg1_entity_type'] = ''
            entity_types['rel_arg1_mention_noun_type'] = ''
            entity_types['rel_arg2_entity_type'] = ''
            entity_types['rel_arg2_mention_noun_type'] = ''
            types = pd.concat([types, entity_types])

        if 'relation' in file_info:
            relation_df = file_info['relation']
            rel_arg1_contexts = relation_df['rel_arg1_context']
            rel_arg2_contexts = relation_df['rel_arg2_context']
            relation_contexts = []
            rel_arg1_texts = relation_df['rel_arg1_text']
            rel_arg2_texts = relation_df['rel_arg2_text']
            relation_texts = []
            for i in range(len(rel_arg1_contexts)):
                # if rel_arg1_contexts[i] == np.nan:  # 寻么填充和这个都不管用。。
                #     rel_arg1_contexts[i] = ''
                # if rel_arg2_contexts[i] == np.nan:
                #     rel_arg2_contexts[i] = ''
                context = str(rel_arg1_contexts[i]) + ' ' + str(rel_arg2_contexts[i])
                relation_contexts.append(context)
                text = rel_arg1_texts[i] + ' ' + rel_arg2_texts[i]
                relation_texts.append(text)
            contexts.extend(relation_contexts)
            texts.extend(relation_texts)

            relation_types = relation_df[['relation_type', 'relation_subtype', 'rel_arg1_entity_type',
                                         'rel_arg1_mention_noun_type', 'rel_arg2_mention_noun_type']].copy()
            relation_types['entity_type'] = ''
            relation_types['entity_mention_noun_type'] = ''
            relation_types['event_mention_type'] = ''
            relation_types['event_mention_subtype'] = ''
            types = pd.concat([types, relation_types])

        if 'event' in file_info:
            event_df = file_info['event']
            event_contexts = event_df['trigger_context']
            event_texts = event_df['trigger_text']
            contexts.extend(event_contexts.tolist())
            texts.extend(event_texts.tolist())

            event_types = event_df[['event_mention_type', 'event_mention_subtype']].copy()
            event_types['entity_type'] = ''
            event_types['entity_mention_noun_type'] = ''
            event_types['relation_type'] = ''
            event_types['relation_subtype'] = ''
            event_types['rel_arg1_entity_type'] = ''
            event_types['rel_arg1_mention_noun_type'] = ''
            event_types['rel_arg2_entity_type'] = ''
            event_types['rel_arg2_mention_noun_type'] = ''
            types = pd.concat([types, event_types])

    # print types
    # print types.values.shape
    types = types.values.tolist()

    # context层次

    # tfidf
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words='english', max_features=300, binary=True)
    tfidf_features = vec.fit_transform(contexts).toarray()
    tfidf_features = tfidf_features.tolist()
    features = tfidf_features

    # 情感极性，主动性，词性，情感词计数
    # reserved_dim = 40  # 统一维数
    # for i in range(len(contexts)):
    #     # 词性
    #     pos = nltk.pos_tag(nltk.word_tokenize(contexts[i]))
    #     length = len(pos)
    #
    #     # 情感极性，主动性
    #     polarity = []
    #     pos_count = 0  # 情感词计数
    #     neg_count = 0
    #     for j in range(length):
    #         lemmed = lemma(pos[j][0])
    #         polarity.append(sentiment(lemmed)[0])  # 极性
    #         polarity.append(sentiment(lemmed)[1])  # 主动性
    #         if sentiment(lemmed)[0] >= 0.5:
    #             pos_count += 1
    #         if sentiment(lemmed)[0] <= -0.5:
    #             neg_count += 1
    #     # print polarity
    #     # 统一维数
    #     while length < reserved_dim:
    #         pos.append(('', ''))
    #         polarity.append(0.0)
    #         polarity.append(0.0)
    #         length = len(pos)
    #     if length > reserved_dim:
    #         pos = pos[:reserved_dim]
    #         polarity = polarity[:reserved_dim * 2]
    #
    #     # 词性加入分类特征
    #     feature_cata = []
    #     for j in range(reserved_dim):
    #         feature_cata.append(pos[j][1])
    #     features_cata.append(feature_cata)
    #
    #     # 极性等用数值特征
    #     features[i].extend(polarity)
    #     features[i].append(pos_count)
    #     features[i].append(neg_count)
    #
    # # 词向量（之前试是降低了，可再试试）
    #
    #
    # target层次

    # 情感极性，主动性，词性，情感词计数
    features_cata = []  # 存放类别特征
    reserved_dim = 10  # 统一维数
    for i in range(len(texts)):
        # 词性
        pos = nltk.pos_tag(nltk.word_tokenize(texts[i]))
        length = len(pos)

        # 情感极性，主动性
        polarity = []
        pos_count = 0  # 情感词计数
        neg_count = 0
        for j in range(length):
            lemmed = lemma(pos[j][0])
            polarity.append(sentiment(lemmed)[0])  # 极性
            polarity.append(sentiment(lemmed)[1])  # 主动性
            if sentiment(lemmed)[0] >= 0.5:
                pos_count += 1
            if sentiment(lemmed)[0] <= -0.5:
                neg_count += 1
        # print polarity
        # 统一维数
        while length < reserved_dim:
            pos.append(('', ''))
            polarity.append(0.0)
            polarity.append(0.0)
            length = len(pos)
        if length > reserved_dim:
            pos = pos[:reserved_dim]
            polarity = polarity[:reserved_dim * 2]

        # 词性加入分类特征
        feature_cata = []
        for j in range(reserved_dim):
            feature_cata.append(pos[j][1])
        features_cata.append(feature_cata)
        # 这里的独热方式好像不太合适，是不是会有很多重复的，应该是弄一个大字典，有则1，无则0；再看;但好像也需

        # 极性等用数值特征
        features[i].extend(polarity)
        features[i].append(pos_count)
        features[i].append(neg_count)

    # type加入
    for i in range(len(types)):
        # print features_cata[i]
        features_cata[i].extend(types[i])

    # 独热编码
    features_cata = pd.DataFrame(features_cata)
    features_cata = pd.get_dummies(features_cata)
    features_cata = features_cata.values
    # print features_cata.shape
    features_cata = features_cata.tolist()

    # 合并
    for i in range(len(features)):
        features[i].extend(features_cata[i])
    print 'Feature num and dim:', len(features), len(features[0])

    return features


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
        # 不太好，NW的参数可能得变一变

    # 提取特征及标签
    print "Samples extraction..."
    x_all = gen_general_features(train_files+test_files)  # 提取特征
    y_train = get_merged_labels(train_files)
    y_test = get_merged_labels(test_files)  # 0,1,2三类
    # 特征分割训练测试集
    trainlen = len(y_train)
    x_train = x_all[:trainlen]
    x_test = x_all[trainlen:]
    print 'Train data number:', len(y_train)
    print 'Test data number:', len(y_test)
    print 'Train labels:', y_train
    print 'Test labels:', y_test

    # 可重采样，但不能把cb弱化，再看

    # 训练
    print 'Train...'
    # regr = linear_model.LinearRegression(normalize=True)  # 使用线性回归
    regr = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    regr.fit(x_train, y_train)

    # 测试
    print 'Test...'
    y_predict = regr.predict(X=x_test)  # 预测
    print y_predict
    for i in range(len(y_predict)):
        if y_predict[i] < 0.5:
            y_predict[i] = 0
        elif y_predict[i] < 1.5:
            y_predict[i] = 1
        elif y_predict[i] < 2.0:  # 较好，还可调
            y_predict[i] = 2
        else:
            y_predict[i] = 3
    y_predict = [int(y) for y in y_predict]

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Predict labels: ', y_predict
    y_test_2c = [0 if y != 3 else 1 for y in y_test]
    y_predict_2c = [0 if y != 3 else 1 for y in y_predict]
    evaluation(y_test_2c, y_predict_2c)  # 2类的测试评价

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    find_sources(test_files, source_dir, ere_dir)
    # test_files = use_annotation_source(test_files)

    # 写入文件
    print 'Write into best files...'
    write_best_files(test_files, predict_dir)