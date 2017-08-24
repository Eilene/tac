# coding=utf-8
from cnn_fit import *
from utils.resampling import resampling_2classes


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

    # 训练部分
    # 提取特征及标签
    total_clip_length = 56
    embeddings_index, dim = read_embedding_index(glove_100d_path)
    print 'Train samples extraction...'
    x_train = gen_matrix_features(train_files, embeddings_index, dim, total_clip_length)  # 提取特征
    y_train = get_merged_labels(train_files)  # 0,1,2三类
    # 分出两种样本
    y_train1 = [1 if y != 0 else 0 for y in y_train]
    x_train2, y_train2 = only_pos_neg(x_train, y_train)
    y_train2 = [y-1 for y in y_train2]
    x_train1, y_train1 = resampling_2classes(x_train, y_train1, len(y_train2))  # 重采样
    x_train2, y_train2 = resampling_2classes(x_train2, y_train2)  # 重采样
    x_train1 = convert_features(x_train1)  # 转换为通道模式
    x_train2 = convert_features(x_train2)  # 转换为通道模式
    # 训练
    print 'Train...'
    # 有无
    model1 = cnn_fit(x_train1, y_train1, 2)  # 有无
    # 正负
    model2 = cnn_fit(x_train2, y_train2, 2)

    # 测试部分
    # 提取特征及标签
    print 'Test samples extraction...'
    x_test = gen_matrix_features(test_files, embeddings_index, dim, total_clip_length)  # 提取特征
    x_test = convert_features(x_test)
    # 测试
    print 'Test...'
    probabilities1 = model1.predict(x_test)
    y_predict1 = predict_by_proba(probabilities1)
    probabilities2 = model2.predict(x_test)
    y_predict2 = predict_by_proba_3classes_threshold(probabilities2, 0.0)
    y_predict = [y_predict2[i] if y_predict1[i] != 0 else 0 for i in range(len(y_predict1))]

    # y_predict保存至csv
    if os.path.exists(test_y_predict_dir) is False:
        os.makedirs(test_y_predict_dir)
    # 分类器预测的
    y_predict_df = pd.DataFrame(y_predict, columns=['y_predict'])
    y_predict_df.to_csv(test_y_predict_dir+'cnn_2levels_y_predict.csv', index=False)

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    find_sources(test_files, test_source_dir, test_ere_dir)

    # 写入
    print 'Write into best files...'
    write_best_files(test_files, test_predict_dir)

