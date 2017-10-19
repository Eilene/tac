# coding=utf-8

import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from src.english_sentiment.features.general_features import *
from src.english_sentiment.features.embedding_vector_features import gen_embeddings_vector_features
from src.english_sentiment.utils.all_utils_package import *
from gensim.models.doc2vec import Doc2Vec

def only_two_classes(x, y, label):
    x_new = []
    y_new = []

    for i in range(len(y)):
        if y[i] != label:
            x_new.append(x[i])
            y_new.append(y[i])

    return x_new, y_new

clf_name = 'rf'


def regression_dev(genre):
    print
    if genre is True:
        print '*** DF ***'
    else:
        print '*** NW ***'

    # 读取各文件中间信息
    print 'Read data...'
    df_file_records, nw_file_records = \
        read_file_info_records(train_ere_dir, train_entity_info_dir, train_relation_info_dir, train_event_info_dir,
                               train_em_args_dir)
    # print df_file_records
    print 'DF files:', len(df_file_records), ' NW files:', len(nw_file_records)

    # DF全部作为训练数据，NW分成训练和测试数据, 合并训练的NW和DF，即可用原来流程进行训练测试
    if genre is True:
        print 'Split into train and test dataset...'
        portion = 0.8
        trainnum = int(len(df_file_records) * portion)
        train_files = df_file_records[:trainnum]  # 这里train_files没有用
        test_files = df_file_records[trainnum:]
        # testnum = int(len(df_file_records) * (1-portion))
        # train_files = df_file_records[testnum:]  # 这里train_files没有用
        # test_files = df_file_records[:testnum]
    else:
        print 'Merge and split into train and test dataset...'
        portion = 0.2
        nw_trainnum = int(len(nw_file_records) * portion)
        train_files = df_file_records + nw_file_records[:nw_trainnum]
        test_files = nw_file_records[nw_trainnum:]

    # 提取特征及标签
    print "Samples extraction..."

    # 标签
    y_train = get_merged_labels(train_files)
    y_test = get_merged_labels(test_files)  # 0,1,2三类

    # tfidf和类别等特征
    import pandas as pd
    if genre is True:
        filepath = '../../../data/output/english_sentiment/general_features_dev_df.csv'
    else:
        filepath = '../../../data/output/english_sentiment/general_features_dev_nw.csv'
    trainlen = len(y_train)

    x_all = gen_general_features(train_files + test_files)
    x_train = x_all[:trainlen]
    x_test = x_all[trainlen:]
    x_all_df = pd.DataFrame(x_all)
    x_all_df.to_csv(filepath, index=False)
    print len(x_all), trainlen

    # x_all_df = pd.read_csv(filepath)
    # x_all = x_all_df.values.tolist()
    # print len(x_all), len(y_train)+len(y_test), len(x_all[0])
    # x_train = x_all[:trainlen]
    # x_test = x_all[trainlen:]

    # 词向量特征
    # total_clip_length = 50
    # embeddings_index, dim = read_embedding_index(glove_6b_100d_path)  # 100 or 300?
    # x_train = gen_embeddings_vector_features(train_files, embeddings_index, dim, total_clip_length)
    # x_test = gen_embeddings_vector_features(test_files, embeddings_index, dim, total_clip_length)

    # doc2vec特征
    # print 'Load doc2vec...'
    # model = Doc2Vec.load(docmodel_path)
    # doc2vec_model = model.docvecs
    # x_train = []
    # for i in range(len(y_train)):
    #     x_train.append(doc2vec_model[i].tolist())
    # x_test = []
    # for i in range(len(y_test)):
    #     x_test.append(doc2vec_model[len(y_train)+i].tolist())

    # 特征拼接
    # for i in range(len(x_train)):
    #     x_train[i].extend(x_train1[i])
    # for i in range(len(x_test)):
    #     x_train[i].extend(x_test1[i])

    x_train, y_train = up_resampling_3classes(x_train, y_train)  # 重采样
    print 'Train data number:', len(y_train)
    print 'Test data number:', len(y_test)
    print 'Train labels:', y_train
    print 'Test labels:', y_test

    # 三个分类器的标签
    print "Labels regenerate..."
    x_train1, y_train1 = only_two_classes(x_train, y_train, 2)  # none vs neg
    x_train2, y_train2 = only_two_classes(x_train, y_train, 1)  # none vs pos
    x_train3, y_train3 = only_two_classes(x_train, y_train, 0)  # neg vs pos

    # 重采样
    print "Resampling..."
    negnum = y_train3.count(1)
    posnum = y_train3.count(2)
    x_train1, y_train1 = up_resampling_2classes(x_train1, y_train1)
    x_train2, y_train2 = up_resampling_2classes(x_train2, y_train2)

    # 训练
    print 'Train...'
    if clf_name == 'svm':
        clf1 = svm.SVC()
        clf2 = svm.SVC()
        clf3 = svm.SVC()
    elif clf_name == 'rf':
        clf1 = RandomForestClassifier(oob_score=True, random_state=10)
        clf2 = RandomForestClassifier(oob_score=True, random_state=10)
        clf3 = RandomForestClassifier(oob_score=True, random_state=10)
    else:
        clf1 = LogisticRegression()
        clf2 = LogisticRegression()
        clf3 = LogisticRegression()
    clf1.fit(x_train1, y_train1)
    clf2.fit(x_train2, y_train2)
    clf3.fit(x_train3, y_train3)

    # 测试
    print 'Test...'
    y_predict1 = clf1.predict(x_test)
    y_predict2 = clf2.predict(x_test)
    y_predict3 = clf3.predict(x_test)

    # 生成最终y_predict
    # 每个样本，若只有1个1，则对应该类；多个或0个，则概率最大类别为输入类别（按理说应该用输出值。。）
    y_predict = [0] * len(y_test)
    for i in range(len(y_predict)):
        y_candid = [y_predict1[i], y_predict2[i], y_predict3[i]]
        y_candid = np.asarray(y_candid)
        counts = np.bincount(y_candid)
        y = np.argmax(counts)
        y_predict[i] = y

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Predict labels1:', y_predict1
    print 'Predict labels2: ', y_predict2
    print 'Predict labels3: ', y_predict2
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    if genre is True:
        dev_y_predict_dir = dev_df_y_predict
    else:
        dev_y_predict_dir = dev_nw_y_predict
    # y_predict保存至csv
    if os.path.exists(dev_y_predict_dir) is False:
        os.makedirs(dev_y_predict_dir)
    # 分类器预测的
    y_predict_df = pd.DataFrame(y_predict, columns=['y_predict'])
    y_predict_df.to_csv(dev_y_predict_dir+clf_name+'_1vs1_y_predict.csv', index=False)

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    if genre is True:
        # 寻找源
        print 'Find sources... '
        find_sources(test_files, train_source_dir, train_ere_dir)

        # 写入文件
        print 'Write into best files...'
        write_best_files(test_files, dev_df_predict_dir)

    else:
        # 写入文件
        print 'Write into best files...'
        write_best_files(test_files, dev_nw_predict_dir)

if __name__ == '__main__':
    # regression_dev(True)
    regression_dev(False)

# 不行，全0了；可下采样