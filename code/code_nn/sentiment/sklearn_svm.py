# coding=utf-8

import nltk

import numpy as np

from pattern.en import lemma
from pattern.en import sentiment

from sklearn import svm
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.constants import *
from utils.read_file_info_records import *
from utils.file_records_other_modification import without_none, to_dict
from utils.filter_none_with_stdict import *
# from utils.resampling import *
from utils.evaluation import evaluation_3classes
from utils.attach_predict_labels import attach_predict_labels
from utils.find_source import find_sources
from utils.write_best import write_best_files
from utils.get_labels import get_merged_labels
from utils.predict_by_proba import *


def gen_sklearn_features(train_files, test_files):
    # 特征
    # 按文件顺序合并
    file_records = train_files + test_files
    contexts = []
    texts = []
    for file_info in file_records:
        if 'entity' in file_info:
            entity_df = file_info['entity']
            # print len(entity_df)  # 0也没关系，就正好是没加
            entity_contexts = entity_df['entity_mention_context']
            entity_texts = entity_df['entity_mention_text']
            contexts.extend(entity_contexts.tolist())
            texts.extend(entity_texts.tolist())
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
        if 'event' in file_info:
            event_df = file_info['event']
            event_contexts = event_df['trigger_context']
            event_texts = event_df['trigger_text']
            contexts.extend(event_contexts.tolist())
            texts.extend(event_texts.tolist())

    # tfidf
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words='english', max_features=300, binary=True)
    tfidf_features = vec.fit_transform(contexts).toarray()
    tfidf_features = tfidf_features.tolist()
    features = tfidf_features

    # 情感极性，主动性，词性
    features_cata = []  # 存放类别特征
    reserved_dim = 10  # 统一维数
    for i in range(len(texts)):
        # 词性
        pos = nltk.pos_tag(nltk.word_tokenize(texts[i]))
        length = len(pos)

        # 情感极性，主动性
        polarity = []
        for j in range(length):
            lemmed = lemma(pos[j][0])
            polarity.append(sentiment(lemmed)[0])  # 极性
            polarity.append(sentiment(lemmed)[1])  # 主动性
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

        features[i].extend(polarity)  # 极性用数值特征

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

    # 提取特征及标签
    print "Samples extraction..."
    train_files = without_none(train_files)  # 训练文件去掉none的样本
    x_all = gen_sklearn_features(train_files, test_files)  # 提取特征
    y_train = get_merged_labels(train_files)  # 只有1,2两类
    y_test = get_merged_labels(test_files)  # 0,1,2三类
    # 特征分割训练测试集
    trainlen = len(y_train)
    x_train = x_all[:trainlen]
    x_test = x_all[trainlen:]
    print 'Train data number:', len(y_train)
    print 'Test data number:', len(y_test)
    print 'Train labels:', y_train
    print 'Test labels:', y_test

    # 训练
    print 'Train...'
    # clf = MultinomialNB()  # 不接受负值
    clf = svm.SVC(probability=True)
    clf.fit(x_train, y_train)
    joblib.dump(clf, 'svm_model.m')  # 保存训练模型

    # 测试
    print 'Test...'
    clf = joblib.load('svm_model.m')
    y_pred_proba = clf.predict_proba(x_test)
    y_predict = predict_by_proba(y_pred_proba, 0.0)
    # 测试文件根据打分过滤掉none的样本
    y_predict1 = filter_none(test_files)
    # y_predict1 = filter_none_with_window_text(test_files)
    y_predict = [y_predict[i] if y_predict1[i] != 0 else y_predict1[i] for i in range(len(y_predict))]

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Filter labels:', y_predict1
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # 测试结果写入记录
    test_files = to_dict(test_files)
    test_files = attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    test_files = find_sources(test_files, source_dir, ere_dir)
    # test_files = use_annotation_source(test_files)

    # 写入文件
    print 'Write into best files...'
    write_best_files(test_files, predict_dir)


