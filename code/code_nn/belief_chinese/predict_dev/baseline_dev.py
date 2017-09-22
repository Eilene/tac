# coding=utf-8

from src.belief_chinese.utils.all_utils_package import *

if __name__ == '__main__':
    mode = True  # True:DF,false:NW

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
        trainnum = int(len(df_file_records) * portion)
        train_files = df_file_records[:trainnum]  # 这里train_files没有用
        test_files = df_file_records[trainnum:]
    else:
        print '*** NW ***'
        print 'Merge and split into train and test dataset...'
        portion = 0.2
        nw_trainnum = int(len(nw_file_records) * portion)
        train_files = df_file_records + nw_file_records[:nw_trainnum]
        test_files = nw_file_records[nw_trainnum:]

    # 对test_files 搞出y_test和y_predict，进行评价
    y_test = get_merged_labels(test_files)  # 0:na, 1:rob, 2:ncb, 3:cb
    y_predict = [3] * len(y_test)  # 全cb
    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Predict labels: ', y_predict
    y_test_2c = [1 if y != 0 else 0 for y in y_test]
    y_predict_2c = [1 if y != 0 else 0 for y in y_predict]
    evaluation(y_test_2c, y_predict_2c)  # 2类的测试评价

    # 修改test_files，全赋cb
    set_cb(test_files)

    # 寻找源
    print 'Find sources... '
    to_dict(test_files)
    find_sources(test_files, train_source_dir, train_ere_dir)

    # 写入文件
    print 'Write into best files...'
    write_best_files(test_files, dev_predict_dir)

