# coding=utf-8

from lstm_fit import *
from utils.filter_none_with_stdict import filter_none


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

    # 提取特征及标签
    print "Generate samples..."
    embeddings_index, dim = read_embedding_index(glove_100d_path)
    without_none(train_files)  # 训练文件去掉none的样本
    file_records = train_files + test_files
    x_all, embedding_matrix, nb_words = gen_lstm_features(file_records, embeddings_index)
    y_train = get_merged_labels(train_files)  # 只有1,2两类
    y_train = [y-1 for y in y_train]  # 改为0,1
    y_test = get_merged_labels(test_files)  # 0,1,2三类
    # 特征分割训练测试集
    trainlen = len(y_train)
    x_train = x_all[:trainlen]
    x_test = x_all[trainlen:]
    print 'Train data number:', len(y_train)
    print 'Test data number:', len(y_test)

    # 训练
    print 'Train...'
    model = lstm_fit(x_train, y_train, embedding_matrix, nb_words, 2)

    # 测试
    print 'Test...'
    probabilities = model.predict(x_test)
    y_predict_lstm = predict_by_proba_3classes_threshold(probabilities, 0.2)
    # 测试文件根据打分过滤掉none的样本
    y_predict_filter = filter_none(test_files)
    y_predict = [y_predict_lstm[i] if y_predict_filter[i] != 0 else 0 for i in range(len(y_predict_lstm))]

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Lstm predict labels: ', y_predict_lstm
    print 'Filter labels:', y_predict_filter
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # y_predict保存至csv
    if os.path.exists(test_y_predict_dir) is False:
        os.makedirs(test_y_predict_dir)
    # 分类器预测的
    y_predict_df = pd.DataFrame(y_predict_lstm, columns=['y_predict'])
    y_predict_df.to_csv(test_y_predict_dir+'lstm_y_predict.csv', index=False)

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    find_sources(test_files, test_source_dir, test_ere_dir)

    # 写入文件
    write_best_files(test_files, test_predict_dir)
