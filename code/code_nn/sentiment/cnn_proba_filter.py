# coding=utf-8

from cnn_fit import *
from utils.filter_none_with_stdict import *

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
    without_none(train_files)  # 训练文件去掉none的样本
    x_train = gen_matrix_features(train_files, embeddings_index, dim, total_clip_length)  # 提取特征
    y_train = get_merged_labels(train_files)  # 只有1,2两类
    y_train = [y-1 for y in y_train]  # 改为0,1
    x_train, y_train = convert_samples(x_train, y_train)  # 转换为通道模式
    # 训练
    print 'Train...'
    model = cnn_fit(x_train, y_train, 2)  # 分正负

    # 测试部分
    # 提取特征及标签
    print 'Test samples extraction...'
    x_test = gen_matrix_features(test_files, embeddings_index, dim, total_clip_length)  # 提取特征
    y_test = get_merged_labels(test_files)  # 0,1,2三类
    x_test, y_test = convert_samples(x_test, y_test)
    # 测试
    print 'Test...'
    probabilities = model.predict(x_test)
    y_predict_cnn = predict_by_proba_3classes_threshold(probabilities, 0.1)
    # 测试文件根据打分过滤掉none的样本
    y_predict_filter = filter_none(test_files)
    y_predict = [y_predict_cnn[i] if y_predict_filter[i] != 0 else 0 for i in range(len(y_predict_cnn))]

    # 评价
    y_test = y_test.tolist()
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'CNN predict labels: ', y_predict_cnn
    print 'Filter labels:', y_predict_filter
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # y_predict保存至csv
    if os.path.exists(y_predict_dir) is False:
        os.makedirs(y_predict_dir)
    # 分类器预测的
    y_predict_df = pd.DataFrame(y_predict_cnn, columns=['y_predict'])
    y_predict_df.to_csv(y_predict_dir+'cnn_y_predict.csv', index=False)

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    find_sources(test_files, source_dir, ere_dir)
    # use_annotation_source(test_files)

    # 写入
    print 'Write into best files...'
    write_best_files(test_files, predict_dir)

