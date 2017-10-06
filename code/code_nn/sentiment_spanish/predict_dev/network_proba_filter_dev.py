# coding=utf-8

from src.sentiment_english.models.network_fit import *
from gensim.models.doc2vec import Doc2Vec

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
        trainnum = int(len(df_file_records) * portion)
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
    # print 'Read glove vectors...'
    # total_clip_length = 50
    # embeddings_index, dim = read_embedding_index(glove_100d_path)
    # print 'Samples extraction...'
    # without_none(train_files)  # 训练文件去掉none的样本
    # # 词向量特征
    # x_train = gen_embeddings_vector_features(train_files, embeddings_index, dim, total_clip_length)
    # x_test = gen_embeddings_vector_features(test_files, embeddings_index, dim, total_clip_length)
    # # # 标签
    # y_train = get_merged_labels(train_files)  # 1,2
    # y_train = [y-1 for y in y_train]  # 改为0,1
    # y_test = get_merged_labels(test_files)  # 0,1,2
    # tfidf和类别等特征
    # x_all = gen_general_features(train_files + test_files)
    # x_train = x_all[:len(y_train)]
    # x_test = x_all[len(y_train):]
    # # 拼起来
    # x_train = []
    # x_test = []
    # for i in range(len(x_train1)):
    #     x = x_train1[i] + x_train2[i]
    #     x_train.append(x)
    # for i in range(len(x_test1)):
    #     x = x_test1[i] + x_test2[i]
    #     x_test.append(x)

    # 外部数据
    # imdb_contexts, y_imdb = read_imdb_data(imdb_dir)
    # x_imdb = gen_external_embeddings_vector_features(imdb_contexts, embeddings_index, dim, 300)
    # # 不太好，imdb的长度较长，有的上千；不截断只平均也不太好，都一样了
    # # 会倾向于都预测成正的？？
    # x_train = x_imdb
    # y_train = y_imdb
    # x_train.extend(x_imdb)
    # y_train.extend(y_imdb)
    #
    # print len(y_train)

    # tfidf
    # contexts = get_contexts(train_files+test_files)
    # imdb_contexts, y_imdb = read_imdb_data(imdb_dir)
    # x_all = gen_all_general_features(imdb_contexts+contexts)
    # y_imdb.extend(y_train)
    # y_train = y_imdb
    # print y_train
    # trainnum = len(y_train)
    # x_train = x_all[:trainnum]
    # x_test = x_all[trainnum:]

    # 标签
    print 'Load doc2vec...'
    y_train = get_merged_labels(train_files)  # 0,1,2三类
    y_test = get_merged_labels(test_files)  # 0,1,2三类
    docmodel_path = st_output_prefix + 'doc2vec_model_200.txt'
    model = Doc2Vec.load(docmodel_path)
    doc2vec_model = model.docvecs
    x_train = []
    for i in range(len(y_train)):
        x_train.append(doc2vec_model[i].tolist())
    x_test = []
    for i in range(len(y_test)):
        x_test.append(doc2vec_model[len(y_train)+i].tolist())
    # 切分类似普通特征。可直接加入到各分类器代码中

    # 训练
    print 'Train...'
    model = network_fit(x_train, y_train, 2)  # 分正负

    # 搞一个调参
    # print 'Grid search...'
    # model = grid_search_network(x_train, y_train, 2, 5)

    # 测试
    print 'Test...'
    probabilities = model.predict(x_test)
    # y_predict_nn = probabilities
    print probabilities
    y_predict_nn = predict_by_proba_3classes_threshold(probabilities, 0.1, 0.3)  # 这个阈值该咋调呢。。
    # 测试文件根据打分过滤掉none的样本
    # y_predict_filter = filter_none(test_files)
    # y_predict = [y_predict_nn[i] if y_predict_filter[i] != 0 else 0 for i in range(len(y_predict_nn))]
    y_predict = y_predict_nn

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'CNN predict labels: ', y_predict_nn
    # print 'Filter labels:', y_predict_filter
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # y_predict保存至csv
    if os.path.exists(dev_y_predict_dir) is False:
        os.makedirs(dev_y_predict_dir)
    # 分类器预测的
    y_predict_df = pd.DataFrame(y_predict_nn, columns=['y_predict'])
    y_predict_df.to_csv(dev_y_predict_dir+'network_y_predict.csv', index=False)

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    find_sources(test_files, train_source_dir, train_ere_dir)
    # use_annotation_source(test_files)

    # 写入
    print 'Write into best files...'
    write_best_files(test_files, dev_predict_dir)

