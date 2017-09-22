# coding=utf-8

from src.belief_chinese.utils.all_utils_package import *


if __name__ == '__main__':
    mode = False  # True:DF,false:NW

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

    # DF全部作为训练数据，NW分成训练和测试数据, 合并训练的NW和DF，即可用原来流程进行训练测试
    if mode is True:
        print '*** DF ***'
        test_files = test_df_file_records
    else:
        print '*** NW ***'
        test_files = test_nw_file_records

    # 修改test_files，全赋neg
    set_cb(test_files)

    # 寻找源
    print 'Find sources... '
    to_dict(test_files)
    # find_sources(test_files, test_source_dir, test_ere_dir)

    # 写入文件
    print 'Write into best files...'
    write_best_files(test_files, test_predict_dir)

