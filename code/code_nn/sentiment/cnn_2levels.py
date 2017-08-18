# coding=utf-8
from cnn_fit import *
from utils.resampling import resampling


def only_pos_neg(x, y):
    x_new = []
    y_new = []

    for i in range(len(y)):
        if y[i] != 0:
            x_new.append(x[i])
            y_new.append(y[i])

    return x_new, y_new

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
        train_files = df_file_records[:trainnum]
        test_files = df_file_records[trainnum:]
    else:
        print '*** NW ***'
        print 'Merge and split into train and test dataset...'
        portion = 0.2
        nw_trainnum = int(len(nw_file_records) * portion)
        train_files = df_file_records + nw_file_records[:nw_trainnum]
        test_files = nw_file_records[nw_trainnum:]

    # 训练部分
    # 提取特征及标签
    total_clip_length = 56
    embeddings_index, dim = read_embedding_index(glove_100d_path)
    print 'Train samples extraction...'
    x_train = gen_cnn_features(train_files, embeddings_index, dim, total_clip_length)  # 提取特征
    y_train = get_merged_labels(train_files)  # 0,1,2三类
    # 分出两种样本
    y_train1 = [1 if y != 0 else 0 for y in y_train]
    x_train2, y_train2 = only_pos_neg(x_train, y_train)
    y_train2 = [y-1 for y in y_train2]
    x_train1, y_train1 = resampling(x_train, y_train1, len(y_train2) * 2)  # 重采样
    x_train2, y_train2 = resampling(x_train2, y_train2)  # 重采样
    x_train1, y_train1 = convert_samples(x_train1, y_train1)  # 转换为通道模式
    x_train2, y_train2 = convert_samples(x_train2, y_train2)  # 转换为通道模式
    # 训练
    print 'Train...'
    # 有无
    model1 = cnn_fit(x_train1, y_train1, 2)  # 有无
    # 正负
    model2 = cnn_fit(x_train2, y_train2, 2)

    # 测试部分
    # 提取特征及标签
    print 'Test samples extraction...'
    x_test = gen_cnn_features(test_files, embeddings_index, dim, total_clip_length)  # 提取特征
    y_test = get_merged_labels(test_files)  # 0,1,2三类
    x_test, y_test = convert_samples(x_test, y_test)
    # 测试
    print 'Test...'
    probabilities1 = model1.predict(x_test)
    y_predict1 = predict_by_proba(probabilities1, 0.0)
    probabilities2 = model1.predict(x_test)
    y_predict2 = predict_by_proba(probabilities2, 0.0)
    y_predict = [y_predict2[i] if y_predict1[i] != 0 else 0 for i in range(len(y_predict1))]

    # 评价
    y_test = y_test.tolist()
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Predict labels 1: ', y_predict1
    print 'Predict labels 2: ', y_predict2
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    find_sources(test_files, source_dir, ere_dir)
    # test_files = use_annotation_source(test_files)

    # 写入
    print 'Write into best files...'
    write_best_files(test_files, predict_dir)

