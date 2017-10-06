# coding=utf-8

import numpy as np

from src.sentiment_english.utils.attach_predict_labels import attach_predict_labels
from src.sentiment_english.utils.constants import *
from src.sentiment_english.utils.evaluation import evaluation_3classes
from src.sentiment_english.utils.file_records_other_modification import to_dict
from src.sentiment_english.utils.get_labels import get_merged_labels
from src.sentiment_english.utils.read_file_info_records import *
from src.sentiment_english.utils.write_best import write_best_files
from src.sentiment_english.utils.find_source import find_sources

if __name__ == '__main__':
    mode = True

    # 读取各文件中间信息
    print 'Read data...'
    df_file_records, nw_file_records = \
        read_file_info_records(train_ere_dir, train_entity_info_dir, train_relation_info_dir, train_event_info_dir,
                               train_em_args_dir)
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

    # 提取test标签
    y_test = get_merged_labels(test_files)  # 0,1,2三类

    # 读取各分类器的y_predict
    y_predicts = []
    for parent, dirnames, filenames in os.walk(dev_y_predict_dir):
        for filename in filenames:
            y_pred_df = pd.read_csv(dev_y_predict_dir+filename)
            y_predicts.append(y_pred_df.values.T[0].tolist())

    # 投票，合并
    y_predict = [0] * len(y_test)
    for i in range(len(y_test)):
        y_candid = [int(y_predict[i]) for y_predict in y_predicts]
        print y_candid
        y_candid = np.asarray(y_candid)
        counts = np.bincount(y_candid)
        print counts
        y = np.argmax(counts)
        print y
        y_predict[i] = y

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
    find_sources(test_files, train_source_dir, train_ere_dir)
    # use_annotation_source(test_files)

    # 写入文件
    print 'Write into best files...'
    write_best_files(test_files, dev_predict_dir)




