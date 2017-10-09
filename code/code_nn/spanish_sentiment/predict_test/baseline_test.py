# coding=utf-8

from src.spanish_sentiment.utils.all_utils_package import *


def baseline_test(genre):
    print
    if genre is True:
        print '*** DF ***'
    else:
        print '*** NW ***'

    # 读取各文件中间信息
    print 'Read data...'
    train_df_file_records, train_nw_file_records = \
        read_file_info_records(train_ere_dir, train_entity_info_dir, train_relation_info_dir, train_event_info_dir,
                               train_em_args_dir)
    test_df_file_records = \
        read_file_info_records(test_df_ere_dir, test_df_entity_info_dir, test_df_relation_info_dir, 
                               test_df_event_info_dir, test_df_em_args_dir, False)
    test_nw_file_records = \
        read_file_info_records(test_nw_ere_dir, test_nw_entity_info_dir, test_nw_relation_info_dir,
                               test_nw_event_info_dir, test_nw_em_args_dir, False)
    # print df_file_records
    print 'Train set: DF files:', len(train_df_file_records), ' NW files:', len(train_nw_file_records)
    print 'Test set: DF files:', len(test_df_file_records), ' NW files:', len(test_nw_file_records)

    # 论坛或新闻
    if genre is True:
        test_files = test_df_file_records
    else:
        test_files = test_nw_file_records

    # 修改test_files，全赋neg
    set_neg(test_files)

    to_dict(test_files)

    if genre is True:
        # 寻找源
        print 'Find sources... '
        find_sources(test_files, test_df_source_dir, test_df_ere_dir)

        # 写入文件
        print 'Write into best files...'
        write_best_files(test_files, test_df_predict_dir)

    else:
        # 写入文件
        print 'Write into best files...'
        write_best_files(test_files, test_nw_predict_dir)

if __name__ == '__main__':
    baseline_test(True)
    baseline_test(False)


