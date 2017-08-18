# coding=utf-8

from sklearn import svm
from sklearn.externals import joblib

from utils.constants import *
from utils.read_file_info_records import *
from features.general_features import gen_general_features
from utils.get_labels import get_merged_labels
from utils.resampling import resampling

from utils.attach_predict_labels import attach_predict_labels
from utils.evaluation import evaluation_3classes
from utils.file_records_other_modification import to_dict
from utils.find_source import find_sources
from utils.write_best import write_best_files

import numpy as np


def only_two_classes(x, y, label):
    x_new = []
    y_new = []

    for i in range(len(y)):
        if y[i] != label:
            x_new.append(x[i])
            y_new.append(y[i])

    return x_new, y_new

if __name__ == '__main__':
    mode = True  # True:DF,false:NW
    clf_name = 'svm'  # 可选lr，svm等

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
    x_all = gen_general_features(train_files+test_files)  # 提取特征
    y_train = get_merged_labels(train_files)  # 0,1,2三类
    y_test = get_merged_labels(test_files)  # 0,1,2三类

    # 特征分割训练测试集
    trainlen = len(y_train)
    x_train = x_all[:trainlen]
    x_test = x_all[trainlen:]
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
    print posnum
    x_train1, y_train1 = resampling(x_train1, y_train1, negnum*2)
    y_train2 = [1 if y == 2 else 0 for y in y_train2]  # 调整下，用于采样（后续在改改采样接口吧）
    x_train2, y_train2 = resampling(x_train2, y_train2, posnum*2)
    print y_train2
    y_train3 = [1 if y == 2 else 0 for y in y_train3]
    x_train3, y_train3 = resampling(x_train3, y_train3, posnum+negnum)

    # 训练
    print 'Train...'
    clf1 = svm.SVC()
    clf1.fit(x_train1, y_train1)
    joblib.dump(clf1, '1vs1_svm_model1.m')  # 保存训练模型
    clf2 = svm.SVC()
    clf2.fit(x_train2, y_train2)
    joblib.dump(clf2, '1vs1_svm_model2.m')  # 保存训练模型
    clf3 = svm.SVC()
    clf3.fit(x_train3, y_train3)
    joblib.dump(clf3, '1vs1_svm_model3.m')  # 保存训练模型
    
    # 测试
    print 'Test...'
    clf1 = joblib.load('1vs1_svm_model1.m')
    y_predict1 = clf1.predict(x_test)
    clf2 = joblib.load('1vs1_svm_model2.m')
    y_predict2 = clf2.predict(x_test)
    clf3 = joblib.load('1vs1_svm_model3.m')
    y_predict3 = clf3.predict(x_test)
    y_predict2 = [2 if y == 1 else 0 for y in y_predict2]  # 调整回来
    y_predict3 = [2 if y == 1 else 1 for y in y_predict3]  # 调整回来

    # 生成最终y_predict
    # 每个样本，若只有1个1，则对应该类；多个或0个，则概率最大类别为输入类别（按理说应该用输出值。。）
    y_predict = [0] * len(y_test)
    for i in range(len(y_predict)):
        y_candid = [y_predict1[i], y_predict2[i], y_predict3[i]]
        y_candid = np.asarray(y_candid)
        counts = np.bincount(y_candid)
        print counts
        y = np.argmax(counts)
        print y
        y_predict[i] = y

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Predict labels1:', y_predict1
    print 'Predict labels2: ', y_predict2
    print 'Predict labels3: ', y_predict2
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # y_predict保存至csv
    if os.path.exists(y_predict_dir) is False:
        os.makedirs(y_predict_dir)
    # 分类器预测的
    y_predict_df = pd.DataFrame(y_predict, columns=['y_predict'])
    y_predict_df.to_csv(y_predict_dir+clf_name+'_1vs1_y_predict.csv', index=False)

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