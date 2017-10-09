# coding=utf-8

from sklearn import linear_model

from src.chinese_sentiment.features.general_features import *
from src.chinese_sentiment.features.embedding_vector_features import gen_embeddings_vector_features
from src.chinese_sentiment.utils.all_utils_package import *
from gensim.models.doc2vec import Doc2Vec


def regression_dev(genre):
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
    # print df_file_records
    print 'DF files:', len(df_file_records), ' NW files:', len(nw_file_records)

    # DF全部作为训练数据，NW分成训练和测试数据, 合并训练的NW和DF，即可用原来流程进行训练测试
    if genre is True:
        print 'Split into train and test dataset...'
        portion = 0.8
        trainnum = int(len(df_file_records) * portion)
        train_files = df_file_records[:trainnum]  # 这里train_files没有用
        test_files = df_file_records[trainnum:]
        # testnum = int(len(df_file_records) * (1-portion))
        # train_files = df_file_records[testnum:]  # 这里train_files没有用
        # test_files = df_file_records[:testnum]
    else:
        print 'Merge and split into train and test dataset...'
        portion = 0.2
        nw_trainnum = int(len(nw_file_records) * portion)
        train_files = df_file_records + nw_file_records[:nw_trainnum]
        test_files = nw_file_records[nw_trainnum:]

    # 提取特征及标签
    print "Samples extraction..."

    # 标签
    y_train = get_merged_labels(train_files)
    y_test = get_merged_labels(test_files)  # 0,1,2三类

    # tfidf和类别等特征
    # trainlen = len(y_train)
    # x_all = gen_general_features(train_files+test_files)
    # x_train1 = x_all[:trainlen]
    # x_test1 = x_all[trainlen:]

    # 词向量特征
    # total_clip_length = 50
    # embeddings_index, dim = read_embedding_index(glove_6b_100d_path)  # 100 or 300?
    # x_train2 = gen_embeddings_vector_features(train_files, embeddings_index, dim, total_clip_length)
    # x_test2 = gen_embeddings_vector_features(test_files, embeddings_index, dim, total_clip_length)

    # doc2vec特征
    print 'Load doc2vec...'
    model = Doc2Vec.load(docmodel_path)
    doc2vec_model = model.docvecs
    x_train = []
    for i in range(len(y_train)):
        x_train.append(doc2vec_model[i].tolist())
    x_test = []
    for i in range(len(y_test)):
        x_test.append(doc2vec_model[len(y_train)+i].tolist())

    # 特征拼接
    # x_train = x_train2
    # x_test = x_test2

    x_train, y_train = up_resampling_3classes(x_train, y_train)  # 重采样
    print 'Train data number:', len(y_train)
    print 'Test data number:', len(y_test)
    print 'Train labels:', y_train
    print 'Test labels:', y_test

    # 训练
    print 'Train...'
    # regr = linear_model.LinearRegression(normalize=True)  # 使用线性回归
    regr = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    regr.fit(x_train, y_train)

    # 测试
    print 'Test...'
    y_predict = regr.predict(X=x_test)  # 预测
    print y_predict
    for i in range(len(y_predict)):
        if y_predict[i] < 0.7:
            y_predict[i] = 0
        elif y_predict[i] < 1.7:
            y_predict[i] = 1
        else:
            y_predict[i] = 2
    y_predict = [int(y) for y in y_predict]

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    if genre is True:
        dev_y_predict_dir = dev_df_y_predict
    else:
        dev_y_predict_dir = dev_nw_y_predict
    # y_predict保存至csv
    if os.path.exists(dev_y_predict_dir) is False:
        os.makedirs(dev_y_predict_dir)
    # 分类器预测的
    y_predict_df = pd.DataFrame(y_predict, columns=['y_predict'])
    y_predict_df.to_csv(dev_y_predict_dir+'regression_y_predict.csv', index=False)

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
    regression_dev(True)
    regression_dev(False)
