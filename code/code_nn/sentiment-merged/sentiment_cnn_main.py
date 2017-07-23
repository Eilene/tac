# coding=utf-8

import pandas as pd
from constants import glove_100d_filepath
from attach_predict_labels import *
from write_best import *
from find_source import *
from sentiment_cnn import *


def divide_data_by_file(entity_info_df, relation_info_df, event_info_df):
    file_records = []

    files = []
    entity_grouped = entity_info_df.groupby('file')
    for name, group in entity_grouped:
        files.append(name)
        file_info = {}
        file_info['filename'] = name
        file_info['entity'] = group
        file_info['relation'] = relation_info_df[relation_info_df['file'] == name]
        if file_info['relation'].count == 0:
            file_info['relation'] = None
        file_info['event'] = event_info_df[event_info_df['file'] == name]
        if file_info['event'].count == 0:
            file_info['event'] = None
        file_records.append(file_info)
    # 没有entity的文件，再过下
    relation_grouped = relation_info_df.groupby('file')
    for name, group in relation_grouped:
        if name in files:
            continue
        files.append(name)
        file_info = {}
        file_info['filename'] = name
        file_info['entity'] = None
        file_info['relation'] = group
        file_info['event'] = event_info_df[event_info_df['file'] == name]
        if file_info['event'].count == 0:
            file_info['event'] = None
        file_records.append(file_info)
    # 没有上面两者的文件，再过下
    event_grouped = event_info_df.groupby('file')
    for name, group in event_grouped:
        if name in files:
            continue
        files.append(name)
        file_info = {}
        file_info['filename'] = name
        file_info['entity'] = None
        file_info['relation'] = None
        file_info['event'] = group
        file_records.append(file_info)

    return file_records


# 读取文件
entity_filename = 'entity_pos_neg_info.csv'
entity_info_df = pd.read_csv(entity_filename)
relation_filename = 'relation_pos_neg_info.csv'
relation_info_df = pd.read_csv(relation_filename)
event_filename = 'event_pos_neg_info.csv'
event_info_df = pd.read_csv(event_filename)
em_args_filename = 'event_pos_neg_info_em_args.csv'
em_args_info_df = pd.read_csv(em_args_filename)

# 按文件划分与结合数据
file_records = divide_data_by_file(entity_info_df, relation_info_df, event_info_df)

# 按文件划分训练集和测试集
portion = 0.8
train_num = int(len(file_records) * 0.8)
train_files = file_records[:train_num]
test_files = file_records[train_num:]

train_entity_df = pd.DataFrame([], columns=entity_info_df.columns)
train_relation_df = pd.DataFrame([], columns=relation_info_df.columns)
train_event_df = pd.DataFrame([], columns=event_info_df.columns)
for file_info in train_files:
    if file_info['entity'] is not None:
        # print file_info['entity']
        train_entity_df = train_entity_df.append(file_info['entity'], ignore_index=True)
    if file_info['relation'] is not None:
        train_relation_df = train_relation_df.append(file_info['relation'], ignore_index=True)
    if file_info['event'] is not None:
        train_event_df = train_event_df.append(file_info['event'], ignore_index=True)

test_entity_df = pd.DataFrame([], columns=entity_info_df.columns)
test_relation_df = pd.DataFrame([], columns=relation_info_df.columns)
test_event_df = pd.DataFrame([], columns=event_info_df.columns)
for file_info in test_files:
    if file_info['entity'] is not None:
        test_entity_df = test_entity_df.append(file_info['entity'], ignore_index=True)
    if file_info['relation'] is not None:
        test_relation_df = test_relation_df.append(file_info['relation'], ignore_index=True)
    if file_info['event'] is not None:
        test_event_df = test_event_df.append(file_info['event'], ignore_index=True)

# 提取特征
embeddings_index, dim = read_embedding_index(glove_100d_filepath)
clip_length = 40
x_entity_train, y_entity_train = gen_entity_samples(train_entity_df, embeddings_index, dim, clip_length)
x_relation_train, y_relation_train = gen_relation_samples(train_relation_df, embeddings_index, dim, clip_length)
x_event_train, y_event_train = gen_event_samples(train_event_df, em_args_info_df, embeddings_index, dim, clip_length)
x_entity_test, y_entity_test = gen_entity_samples(test_entity_df, embeddings_index, dim, clip_length)
x_relation_test, y_relation_test = gen_relation_samples(test_relation_df, embeddings_index, dim, clip_length)
x_event_test, y_event_test = gen_event_samples(test_event_df, em_args_info_df, embeddings_index, dim, clip_length)

# 合并
# 训练集和测试集都各分一部分
x_train = np.vstack((x_entity_train, x_relation_train))
x_train = np.vstack((x_train, x_event_train))
y_train = np.vstack((y_entity_train, y_relation_train))
y_train = np.vstack((y_train, y_event_train))
x_test = np.vstack((x_entity_test, x_relation_test))
x_test = np.vstack((x_test, x_event_test))
y_test = np.vstack((y_entity_test, y_relation_test))
y_test = np.vstack((y_test, y_event_test))
# print y_test
print 'Test data number: ', len(y_test)

# cnn
# 训练+验证
model = cnn_fit(x_train, y_train)
# model.save('relation_pos_neg_model.h5')
# 测试评价
# model = load_model('relation_pos_neg_model.h5')  # 有问题，编码不对
print
print 'Predict...'
y_predict = cnn_predict(model, x_test, y_test)
print y_predict
# 进一步评价
evaluation(y_test, y_predict)

# label，加入test_files，并且df转字典列表
print "Generate output results..."
y_predict = y_test
test_files = attach_predict_labels(test_files, y_predict)

# 找源，加入test_files
test_files = find_source(test_files)

# test_files写入best文件
print 'Write files...'
write_best_files(test_files)






