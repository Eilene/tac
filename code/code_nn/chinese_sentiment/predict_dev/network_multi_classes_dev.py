# coding=utf-8

from src.chinese_sentiment.models.network_fit import *
from src.chinese_sentiment.utils.resampling import up_resampling_3classes
from src.chinese_sentiment.features.general_features import gen_general_features
from gensim.models.doc2vec import Doc2Vec


def network_multi_classes_dev(genre):
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

    # 标签
    print 'Sample extraction...'
    y_train = get_merged_labels(train_files)
    y_test = get_merged_labels(test_files)  # 0,1,2三类

    # tfidf和类别等特征
    import pandas as pd
    if genre is True:
        filepath = dev_df_general_feature_filepath
    else:
        filepath = dev_nw_general_feature_filepath
    trainlen = len(y_train)

    # x_all = gen_general_features(train_files + test_files)
    # x_train = x_all[:trainlen]
    # x_test = x_all[trainlen:]
    # x_all_df = pd.DataFrame(x_all)
    # x_all_df.to_csv(filepath, index=False)
    # print len(x_all), trainlen

    x_all_df = pd.read_csv(filepath)
    x_all = x_all_df.values.tolist()
    print len(x_all), len(y_train)+len(y_test), len(x_all[0])
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
    y_predict_df.to_csv(dev_y_predict_dir+'network_multi_classes_y_predict.csv', index=False)

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    if genre is True:
        # 寻找源
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
    network_multi_classes_dev(True)
    # network_multi_classes_dev(False)

