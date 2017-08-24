# coding=utf-8

from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from features.general_features import gen_general_features
from utils.attach_predict_labels import attach_predict_labels
from utils.constants import *
from utils.evaluation import evaluation_3classes
from utils.file_records_other_modification import to_dict
from utils.get_labels import get_merged_labels
from utils.predict_by_proba import *
from utils.read_file_info_records import *
from utils.resampling import resampling_2classes
from utils.write_best import write_best_files
from utils.find_source import find_sources

if __name__ == '__main__':
    mode = True  # True:DF,false:NW
    clf_name = 'lr'  # 可选lr，svm等

    # 读取各文件中间信息
    print 'Read data...'
    train_df_file_records, train_nw_file_records = \
        read_file_info_records(train_ere_dir, train_entity_info_dir, train_relation_info_dir, train_event_info_dir,
                               train_em_args_dir)
    test_df_file_records, test_nw_file_records = \
        read_file_info_records(test_ere_dir, test_entity_info_dir, test_relation_info_dir, test_event_info_dir,
                               test_em_args_dir)
    print 'Train set: DF files:', len(train_df_file_records), ' NW files:', len(train_nw_file_records)
    print 'Test set: DF files:', len(test_df_file_records), ' NW files:', len(test_nw_file_records)

    # 论坛或新闻
    if mode is True:
        print '*** DF ***'
        train_files = train_df_file_records
        test_files = test_df_file_records
    else:
        print '*** NW ***'
        train_files = train_df_file_records + train_nw_file_records
        test_files = test_nw_file_records

    # 提取特征及标签
    print "Samples extraction..."
    x_all = gen_general_features(train_files+test_files)  # 提取特征
    y_train = get_merged_labels(train_files)  # 0,1,2三类

    # 特征分割训练测试集
    trainlen = len(y_train)
    x_train = x_all[:trainlen]
    x_test = x_all[trainlen:]
    print 'Train data number:', len(y_train)
    print 'Test data number:', len(x_test)
    print 'Train labels:', y_train

    # 三个分类器的标签
    print "Labels regenerate..."
    y_train1 = [1 if y == 0 else 0 for y in y_train]
    y_train2 = [1 if y == 1 else 0 for y in y_train]
    y_train3 = [1 if y == 2 else 0 for y in y_train]

    # 重采样
    print "Resampling..."
    negnum = y_train2.count(1)
    posnum = y_train3.count(1)
    x_train1, y_train1 = resampling_2classes(x_train, y_train1, negnum+posnum)  # 参数填一个，后面默认1:1
    x_train2, y_train2 = resampling_2classes(x_train, y_train2, negnum)
    x_train3, y_train3 = resampling_2classes(x_train, y_train3, posnum)

    # 训练
    print 'Train...'
    clf1 = LogisticRegression()
    # clf1 = svm.SVC(probability=True)
    clf1.fit(x_train1, y_train1)
    clf2 = LogisticRegression()
    # clf2 = svm.SVC(probability=True)
    clf2.fit(x_train2, y_train2)
    clf3 = LogisticRegression()
    # clf3 = svm.SVC(probability=True)
    clf3.fit(x_train3, y_train3)

    # 测试
    print 'Test...'
    y_pred_proba1 = clf1.predict_proba(x_test)
    y_predict1 = predict_by_proba(y_pred_proba1)
    y_pred_proba2 = clf2.predict_proba(x_test)
    y_predict2 = predict_by_proba(y_pred_proba2)
    y_pred_proba3 = clf3.predict_proba(x_test)
    y_predict3 = predict_by_proba(y_pred_proba3)

    # 生成最终y_predict
    # 每个样本，若只有1个1，则对应该类；多个或0个，则概率最大类别为输入类别（按理说应该用输出值。。）
    y_predict = [0] * len(y_predict1)
    for i in range(len(y_predict)):
        candid = [y_predict1[i], y_predict2[i], y_predict3[i]]
        if candid.count(1) == 1:
            y_predict[i] = candid.index(1)
            # print candid, candid.index(1)
        else:
            candid_proba = [y_pred_proba1[i][1], y_pred_proba2[i][1], y_pred_proba3[i][1]]
            y_predict[i] = candid_proba.index(max(candid_proba))  # 第一个匹配项索引位置

    # y_predict保存至csv
    if os.path.exists(test_y_predict_dir) is False:
        os.makedirs(test_y_predict_dir)
    # 分类器预测的
    y_predict_df = pd.DataFrame(y_predict, columns=['y_predict'])
    y_predict_df.to_csv(test_y_predict_dir+clf_name+'_1vsothers_y_predict.csv', index=False)

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    find_sources(test_files, test_source_dir, test_ere_dir)

    # 写入文件
    print 'Write into best files...'
    write_best_files(test_files, test_predict_dir)
