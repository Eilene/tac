# coding=utf-8

from src.sentiment_english.features.general_features import gen_general_features
from src.sentiment_english.models.network_fit import *
from src.sentiment_english.utils.resampling import up_resampling_3classes

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

    # 提取特征及标签
    print 'Samples extraction...'
    total_clip_length = 56
    embeddings_index, dim = read_embedding_index(glove_100d_path)
    # 词向量特征
    x_train = gen_embeddings_vector_features(train_files, embeddings_index, dim, total_clip_length)  # 提取特征
    x_test = gen_embeddings_vector_features(test_files, embeddings_index, dim, total_clip_length)  # 提取特征
    # 标签
    y_train = get_merged_labels(train_files)  # 0,1,2三类
    y_test = get_merged_labels(test_files)  # 0,1,2三类
    # tfidf和类别等特征
    # x_all = gen_general_features(train_files + test_files)
    # x_train = x_all[:len(y_train)]
    # x_test = x_all[len(y_train):]
    # 拼起来
    # x_train = []
    # x_test = []
    # for i in range(len(x_train1)):
    #     x = x_train1[i] + x_train2[i]
    #     x_train.append(x)
    # for i in range(len(x_test1)):
    #     x = x_test1[i] + x_test2[i]
    #     x_test.append(x)

    # 重采样
    print 'Resampling...'
    x_train, y_train = up_resampling_3classes(x_train, y_train)

    # 训练
    # print 'Train...'
    # model = network_fit(x_train, y_train, 3)  # 分三类

    print 'Grid search...'
    param = grid_search_network(x_train, y_train, 3, 5)

    # 训练
    print 'Train...'
    model = network_fit(x_train, y_train, 3, drop_rate=param['drop_rate'], optimizer=param['optimizer'],
                                 hidden_unit1=param['hidden_unit1'], hidden_unit2=param['hidden_unit2'],
                                 activation=param['activation'], init_mode=param['init_mode'], epochs=param['epoch'])
    # 分三类
    # 交叉验证反而选出来的更差？？

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
    if os.path.exists(dev_y_predict_dir) is False:
        os.makedirs(dev_y_predict_dir)
    # 分类器预测的
    y_predict_df = pd.DataFrame(y_predict, columns=['y_predict'])
    y_predict_df.to_csv(dev_y_predict_dir+'network_3classes_y_predict.csv', index=False)

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    find_sources(test_files, train_source_dir, train_ere_dir)

    # 写入
    print 'Write into best files...'
    write_best_files(test_files, dev_predict_dir)