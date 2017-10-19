# coding=utf-8

from src.chinese_belief.models.network_fit import *
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

    # 提取特征及标签
    y_train = get_merged_labels(train_files)
    trainlen = len(y_train)
    x_all = gen_general_features(train_files+test_files)
    x_train = x_all[:trainlen]
    x_test = x_all[trainlen:]

    # 词向量特征
    # y_train = get_merged_labels(train_files)  # 0,1,2
    # total_clip_length = 36
    # embeddings_index, dim = read_embedding_index(word2vec_model_path)  # 100 or 300?
    # x_train = gen_embeddings_vector_features(train_files, embeddings_index, dim, total_clip_length)
    # x_test = gen_embeddings_vector_features(test_files, embeddings_index, dim, total_clip_length)

    # print 'Load doc2vec...'
    # model = Doc2Vec.load(docmodel_path)
    # doc2vec_model = model.docvecs
    # y_train_df = get_merged_labels(train_df_file_records)
    # y_train_nw = get_merged_labels(train_nw_file_records)
    # test_df_num = get_sample_num(test_df_file_records)
    # test_nw_num = get_sample_num(test_nw_file_records)
    # # print test_df_num, test_nw_num
    # if genre is True:
    #     y_train = y_train_df
    #     x_train = []
    #     for i in range(len(y_train)):
    #         x_train.append(doc2vec_model[i].tolist())
    #     x_test = []
    #     for i in range(test_df_num):
    #         x_test.append(doc2vec_model[len(y_train) + len(y_train_nw) + i].tolist())
    # else:
    #     y_train = y_train_df + y_train_nw
    #     x_train = []
    #     for i in range(len(y_train)):
    #         x_train.append(doc2vec_model[i].tolist())
    #     x_test = []
    #     for i in range(test_nw_num):
    #         x_test.append(doc2vec_model[len(y_train) + test_df_num + i].tolist())

    # 训练
    print 'Train...'
    model = network_fit(x_train, y_train, 4)  # 分三类
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


