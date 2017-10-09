# coding=utf-8
from src.english_sentiment.models.cnn_fit import *
from src.english_sentiment.utils.resampling import up_resampling_3classes


def cnn_multi_classes_dev(genre):
    print
    if genre is True:
        print '*** DF ***'
    else:
        print '*** NW ***'

    # 读取各文件中间信息
    print 'Read data...'
    df_file_records, nw_file_records = \
        read_file_info_records(train_ere_dir, train_entity_info_dir, train_relation_info_dir, train_event_info_dir,
                               train_em_args_dir)
    print 'DF files:', len(df_file_records), ' NW files:', len(nw_file_records)

    # DF全部作为训练数据，NW分成训练和测试数据, 合并训练的NW和DF，即可用原来流程进行训练测试
    if genre is True:
        print 'Split into train and test dataset...'
        portion = 0.8
        trainnum = int(len(df_file_records) * portion)
        train_files = df_file_records[:trainnum]
        test_files = df_file_records[trainnum:]
    else:
        print 'Merge and split into train and test dataset...'
        portion = 0.2
        nw_trainnum = int(len(nw_file_records) * portion)
        train_files = df_file_records + nw_file_records[:nw_trainnum]
        test_files = nw_file_records[nw_trainnum:]

    # 训练部分
    # 提取特征及标签
    total_clip_length = 56
    embeddings_index, dim = read_embedding_index(glove_6b_100d_path)
    print 'Train samples extraction...'
    x_train = gen_matrix_features(train_files, embeddings_index, dim, total_clip_length)  # 提取特征
    y_train = get_merged_labels(train_files)  # 0,1,2三类
    print 'Resampling...'
    x_train, y_train = up_resampling_3classes(x_train, y_train)  # 重采样
    x_train = convert_features(x_train)  # 转换为通道模式
    # 训练
    print 'Train...'
    model = cnn_fit(x_train, y_train, 3)  # 分三类

    # 测试部分
    # 提取特征及标签
    print 'Test samples extraction...'
    x_test = gen_matrix_features(test_files, embeddings_index, dim, total_clip_length)  # 提取特征
    y_test = get_merged_labels(test_files)  # 0,1,2三类
    x_test = convert_features(x_test)
    # 测试
    print 'Test...'
    probabilities = model.predict(x_test)
    y_predict = predict_by_proba(probabilities)

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # y_predict保存至csv
    if genre is True:
        dev_y_predict_dir = dev_df_y_predict
    else:
        dev_y_predict_dir = dev_nw_y_predict
    if os.path.exists(dev_y_predict_dir) is False:
        os.makedirs(dev_y_predict_dir)
    # 分类器预测的
    y_predict_df = pd.DataFrame(y_predict, columns=['y_predict'])
    y_predict_df.to_csv(dev_y_predict_dir+'cnn_multi_classes_y_predict.csv', index=False)

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    # 寻找源
    if genre is True:
        print 'Find sources... '
        find_sources(test_files, train_source_dir, train_ere_dir)

        # 写入文件
        print 'Write into best files...'
        write_best_files(test_files, dev_df_predict_dir)

    else:
        # 写入文件
        print 'Write into best files...'
        write_best_files(test_files, dev_nw_predict_dir)

if __name__ == '__main__':
    cnn_multi_classes_dev(True)
    cnn_multi_classes_dev(False)

# 2\3轮怎么全0了，不行，先放弃；可下采样或减少轮数
# 1轮都1了
# test数据报错


