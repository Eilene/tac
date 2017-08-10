# coding=utf-8

from utils.read_file_info_records import *
from utils.constants import *
from utils.find_source import *
from utils.write_best import *
from utils.evaluation import *


def get_labels(str_labels):
    datanum = len(str_labels)
    labels = [0] * datanum
    for i in range(datanum):
        if str_labels[i] == 'pos':
            labels[i] = 2
        elif str_labels[i] == 'neg':
            labels[i] = 1
        else:
            labels[i] = 0
    return labels


def get_merged_labels(file_records):
    labels = []
    for i in range(len(file_records)):
        if 'entity' in file_records[i]:
            str_labels = file_records[i]['entity']['label_polarity']
            entity_labels = get_labels(str_labels)
            labels.extend(entity_labels)
        if 'relation' in file_records[i]:
            str_labels = file_records[i]['relation']['label_polarity']
            relation_labels = get_labels(str_labels)
            labels.extend(relation_labels)
        if 'event' in file_records[i]:
            str_labels = file_records[i]['event']['label_polarity']
            event_labels = get_labels(str_labels)
            labels.extend(event_labels)
    return labels


def set_neg(file_records):
    for i in range(len(file_records)):
        if 'entity' in file_records[i]:
            file_records[i]['entity']['predict_polarity'] = 'neg'
        if 'relation' in file_records[i]:
            file_records[i]['relation']['predict_polarity'] = 'neg'
        if 'event' in file_records[i]:
            file_records[i]['event']['predict_polarity'] = 'neg'
    return file_records


def to_dict(file_records):
    for i in range(len(file_records)):
        if 'entity' in file_records[i]:
            file_records[i]['entity'] = file_records[i]['entity'].to_dict(orient='records')
        if 'relation' in file_records[i]:
            file_records[i]['relation'] = file_records[i]['relation'].to_dict(orient='records')
        if 'event' in file_records[i]:
            file_records[i]['event'] = file_records[i]['event'].to_dict(orient='records')
    return file_records

if __name__ == '__main__':
    mode = False  # True:DF,false:NW

    # 读取各文件中间信息
    print 'Read data...'
    df_file_records, nw_file_records = \
        read_file_info_records(ere_dir, entity_info_dir, relation_info_dir, event_info_dir, em_args_dir)
    print len(df_file_records), len(nw_file_records)

    # DF全部作为训练数据，NW分成训练和测试数据, 合并训练的NW和DF，即可用原来流程进行训练测试
    if mode is True:
        print '**DF**'
        print 'Split into train and test dataset...'
        portion = 0.8
        trainnum = int(len(df_file_records) * 0.8)
        train_files = df_file_records[:trainnum]
        test_files = df_file_records[trainnum:]
    else:
        print '**NW**'
        print 'Merge and split into train and test dataset...'
        portion = 0.2
        nw_trainnum = int(len(nw_file_records) * portion)
        train_files = df_file_records + nw_file_records[:nw_trainnum]
        test_files = nw_file_records[nw_trainnum:]
        print nw_trainnum

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

    # 寻找源
    print 'Find sources... '
    test_files = to_dict(test_files)
    test_files = find_sources(test_files, source_dir, ere_dir)

    # 写入文件
    print 'Write into best files...'
    write_best_files(test_files, predict_dir)
