# coding=utf-8

from sklearn import linear_model

from utils.constants import *
from utils.read_file_info_records import *
from utils.file_records_other_modification import to_dict
from utils.evaluation import evaluation_3classes
from utils.attach_predict_labels import attach_predict_labels
from utils.find_source import find_sources
from utils.write_best import write_best_files
from utils.get_labels import get_merged_labels
from utils.extract_features import gen_vector_features
from utils.resampling import resampling_3classes

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
    # x_all = gen_sklearn_features(train_files, test_files)  # 提取特征
    x_all = gen_vector_features(train_files, test_files)  # 提取特征
    y_train = get_merged_labels(train_files)
    y_test = get_merged_labels(test_files)  # 0,1,2三类
    # 特征分割训练测试集
    trainlen = len(y_train)
    x_train = x_all[:trainlen]
    x_test = x_all[trainlen:]
    x_train, y_train = resampling_3classes(x_train, y_train)  # 重采样
    print 'Train data number:', len(y_train)
    print 'Test data number:', len(y_test)
    print 'Train labels:', y_train
    print 'Test labels:', y_test


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
        if y_predict[i] < 0.7:
            y_predict[i] = 0
        elif y_predict[i] < 1.7:
            y_predict[i] = 1
        else:
            y_predict[i] = 2
    y_predict = [int(y) for y in y_predict]

    # 评价
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

    # 写入文件
    print 'Write into best files...'
    write_best_files(test_files, predict_dir)