# coding=utf-8

from utils.constants import *
from utils.read_file_info_records import *
from utils.get_labels import get_merged_labels
from utils.find_source import find_sources
from utils.write_best import write_best_files
from utils.evaluation import evaluation_3classes, evaluation_source
from utils.file_records_other_modification import to_dict
from utils.attach_predict_labels import set_neg


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
    y_test = get_merged_labels(test_files)  # 0:none, 1:neg, 2:pos
    y_predict = [1] * len(y_test)  # 全neg
    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # 修改test_files，全赋neg
    test_files = set_neg(test_files)

    # 全部数据试一下
    # print
    # print 'All df data source evaluation:'
    # df_file_records = to_dict(df_file_records)
    # df_file_records_dict = find_sources(df_file_records, source_dir, ere_dir)
    # evaluation_source(df_file_records)
    # print

    # 寻找源
    print 'Find sources... '
    test_files = to_dict(test_files)
    test_files = find_sources(test_files, source_dir, ere_dir)
    # test_files = use_annotation_source(test_files)
    # 加一个找源准确率的评价
    evaluation_source(test_files)

    # 写入文件
    print 'Write into best files...'
    write_best_files(test_files, predict_dir)


