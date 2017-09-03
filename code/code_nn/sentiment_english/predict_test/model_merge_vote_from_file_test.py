# coding=utf-8

import numpy as np

from src.sentiment_english.utils.attach_predict_labels import attach_predict_labels
from src.sentiment_english.utils.constants import *
from src.sentiment_english.utils.file_records_other_modification import to_dict
from src.sentiment_english.utils.read_file_info_records import *
from src.sentiment_english.utils.write_best import write_best_files
from src.sentiment_english.utils.find_source import find_sources

if __name__ == '__main__':
    mode = True

    # 读取各文件中间信息
    print 'Read data...'
    test_df_file_records, test_nw_file_records = \
        read_file_info_records(test_ere_dir, test_entity_info_dir, test_relation_info_dir, test_event_info_dir,
                               test_em_args_dir)
    print 'Test set: DF files:', len(test_df_file_records), ' NW files:', len(test_nw_file_records)

    # 论坛或新闻
    if mode is True:
        print '*** DF ***'
        test_files = test_df_file_records
    else:
        print '*** NW ***'
        test_files = test_nw_file_records

    # 读取各分类器的y_predict
    y_predicts = []
    for parent, dirnames, filenames in os.walk(test_y_predict_dir):
        for filename in filenames:
            y_pred_df = pd.read_csv(test_y_predict_dir+filename)
            y_predicts.append(y_pred_df.values.T[0].tolist())

    # 投票，合并
    y_predict = [0] * len(y_predicts[0])
    for i in range(len(y_predict)):
        y_candid = [int(y_predict[i]) for y_predict in y_predicts]
        print y_candid
        y_candid = np.asarray(y_candid)
        counts = np.bincount(y_candid)
        print counts
        y = np.argmax(counts)
        print y
        y_predict[i] = y

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    find_sources(test_files, test_source_dir, test_ere_dir)
    # use_annotation_source(test_files)

    # 写入文件
    print 'Write into best files...'
    write_best_files(test_files, test_predict_dir)




