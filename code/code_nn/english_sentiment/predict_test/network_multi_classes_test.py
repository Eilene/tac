# coding=utf-8

# from src.english_sentiment.features.general_features import gen_general_features
from src.english_sentiment.models.network_fit import *
from src.english_sentiment.utils.resampling import up_resampling_3classes
from gensim.models.doc2vec import Doc2Vec


def network_multi_classes_test(genre):
    print
    if genre is True:
        print '*** DF ***'
    else:
        print '*** NW ***'

    # 读取各文件中间信息
    print 'Read data...'
    train_df_file_records, train_nw_file_records = \
        read_file_info_records(train_ere_dir, train_entity_info_dir, train_relation_info_dir, train_event_info_dir,
                               train_em_args_dir)
    test_df_file_records = \
        read_file_info_records(test_df_ere_dir, test_df_entity_info_dir, test_df_relation_info_dir, 
                               test_df_event_info_dir, test_df_em_args_dir, False)
    test_nw_file_records = \
        read_file_info_records(test_nw_ere_dir, test_nw_entity_info_dir, test_nw_relation_info_dir,
                               test_nw_event_info_dir, test_nw_em_args_dir, False)
    # print df_file_records
    print 'Train set: DF files:', len(train_df_file_records), ' NW files:', len(train_nw_file_records)
    print 'Test set: DF files:', len(test_df_file_records), ' NW files:', len(test_nw_file_records)

    # 论坛或新闻
    if genre is True:
        train_files = train_df_file_records
        test_files = test_df_file_records
    else:
        train_files = train_df_file_records + train_nw_file_records
        test_files = test_nw_file_records

    # 标签
    print 'Sample extraction...'
    y_train = get_merged_labels(train_files)

    # tfidf和类别等特征
    import pandas as pd
    if genre is True:
        filepath = test_df_general_feature_filepath
    else:
        filepath = test_nw_general_feature_filepath
    trainlen = len(y_train)

    # x_all = gen_general_features(train_files + test_files)
    # x_train = x_all[:trainlen]
    # x_test = x_all[trainlen:]
    # x_all_df = pd.DataFrame(x_all)
    # x_all_df.to_csv(filepath, index=False)
    # print len(x_all), trainlen

    x_all_df = pd.read_csv(filepath)
    x_all = x_all_df.values.tolist()
    x_train = x_all[:trainlen]
    x_test = x_all[trainlen:]

    # 词向量特征
    # total_clip_length = 50
    # embeddings_index, dim = read_embedding_index(glove_6b_100d_path)  # 100 or 300?
    # x_train = gen_embeddings_vector_features(train_files, embeddings_index, dim, total_clip_length)
    # x_test = gen_embeddings_vector_features(test_files, embeddings_index, dim, total_clip_length)

    # doc2vec特征
    # print 'Load doc2vec...'
    # model = Doc2Vec.load(docmodel_path)
    # doc2vec_model = model.docvecs
    # x_train = []
    # for i in range(len(y_train)):
    #     x_train.append(doc2vec_model[i].tolist())
    # x_test = []
    # for i in range(len(y_test)):
    #     x_test.append(doc2vec_model[len(y_train)+i].tolist())

    # 特征拼接
    # for i in range(len(x_train)):
    #     x_train[i].extend(x_train1[i])
    # for i in range(len(x_test)):
    #     x_train[i].extend(x_test1[i])

    # 重采样
    print 'Resampling...'
    x_train, y_train = up_resampling_3classes(x_train, y_train)

    # 训练
    print 'Train...'
    model = network_fit(x_train, y_train, 3)  # 分三类
    # print model.total_loss  # 不对。。

    # print 'Grid search...'
    # param = grid_search_network(x_train, y_train, 3, 5)
    #
    # # 训练
    # print 'Train...'
    # model = network_fit(x_train, y_train, 3, drop_rate=param['drop_rate'], optimizer=param['optimizer'],
    #                              hidden_unit1=param['hidden_unit1'], hidden_unit2=param['hidden_unit2'],
    #                              activation=param['activation'], init_mode=param['init_mode'], epochs=param['epoch'])
    # 分三类
    # 交叉验证反而选出来的更差？？

    # 测试
    print 'Test...'
    probabilities = model.predict(x_test)
    y_predict = predict_by_proba(probabilities)

    # y_predict保存至csv
    if genre is True:
        test_y_predict_dir = test_df_y_predict
    else:
        test_y_predict_dir = test_nw_y_predict
    if os.path.exists(test_y_predict_dir) is False:
        os.makedirs(test_y_predict_dir)
    # 分类器预测的
    y_predict_df = pd.DataFrame(y_predict, columns=['y_predict'])
    y_predict_df.to_csv(test_y_predict_dir+'network_multi_classes_y_predict.csv', index=False)

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    if genre is True:
        # 寻找源
        print 'Find sources... '
        find_sources(test_files, test_df_source_dir, test_df_ere_dir)

        # 写入文件
        print 'Write into best files...'
        write_best_files(test_files, test_df_predict_dir)

    else:
        # 写入文件
        print 'Write into best files...'
        write_best_files(test_files, test_nw_predict_dir)


if __name__ == '__main__':
    network_multi_classes_test(True)
    # network_multi_classes_test(False)


