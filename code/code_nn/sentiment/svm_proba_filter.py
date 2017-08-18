# coding=utf-8

from sklearn import svm
from sklearn.externals import joblib

from features.general_features import gen_general_features
from utils.attach_predict_labels import attach_predict_labels
from utils.constants import *
from utils.evaluation import evaluation_3classes
from utils.file_records_other_modification import without_none, to_dict
from utils.find_source import find_sources
from utils.get_labels import get_merged_labels
from utils.predict_by_proba import *
from utils.read_file_info_records import *
from utils.write_best import write_best_files

if __name__ == '__main__':
    mode = False  # True:DF,false:NW

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
    without_none(train_files)  # 训练文件去掉none的样本
    x_all = gen_general_features(train_files+test_files)  # 提取特征
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
    y_predict = predict_by_proba(y_pred_proba, 0.3)
    # 测试文件根据打分过滤掉none的样本
    # y_predict1 = filter_none(test_files)
    # # y_predict1 = filter_none_with_window_text(test_files)
    # y_predict = [y_predict[i] if y_predict1[i] != 0 else y_predict1[i] for i in range(len(y_predict))]

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    # print 'Filter labels:', y_predict1
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

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


