# coding=utf-8

from sklearn import svm
from sklearn.externals import joblib

from utils.constants import *
from utils.read_file_info_records import *
from utils.file_records_other_modification import to_dict
from utils.resampling import resampling
from utils.evaluation import evaluation_3classes
from utils.attach_predict_labels import attach_predict_labels
from utils.find_source import find_sources
from utils.write_best import write_best_files
from utils.get_labels import get_merged_labels
from utils.predict_by_proba import *
from utils.extract_features import gen_vector_features


def only_pos_neg(x, y):
    x_new = []
    y_new = []

    for i in range(len(y)):
        if y[i] != 0:
            x_new.append(x[i])
            y_new.append(y[i])

    return x_new, y_new

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
    x_all = gen_vector_features(train_files, test_files)  # 提取特征
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
    # 分出两种样本
    y_train1 = [1 if y != 0 else 0 for y in y_train]
    x_train2, y_train2 = only_pos_neg(x_train, y_train)
    y_train2 = [y-1 for y in y_train2]
    x_train1, y_train1 = resampling(x_train, y_train1, len(y_train2))  # 重采样
    x_train2, y_train2 = resampling(x_train2, y_train2)  # 重采样

    # 训练
    print 'Train...'
    # clf = MultinomialNB()  # 不接受负值
    clf1 = svm.SVC(probability=True)
    clf1.fit(x_train1, y_train1)
    joblib.dump(clf1, 'svm_model1.m')  # 保存训练模型
    clf2 = svm.SVC(probability=True)
    clf2.fit(x_train1, y_train1)
    joblib.dump(clf2, 'svm_model2.m')  # 保存训练模型

    # 测试
    print 'Test...'
    clf1 = joblib.load('svm_model1.m')
    y_pred_proba1 = clf1.predict_proba(x_test)
    y_predict1 = predict_by_proba(y_pred_proba1, 0.0)
    clf2 = joblib.load('svm_model2.m')
    y_pred_proba2 = clf2.predict_proba(x_test)
    y_predict2 = predict_by_proba(y_pred_proba2, 0.0)
    y_predict = [y_predict2[i] if y_predict1[i] != 0 else 0 for i in range(len(y_predict1))]

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Predict labels1:', y_predict1
    print 'Predict labels2: ', y_predict2
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


