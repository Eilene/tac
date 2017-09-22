# coding=utf-8

from gensim.models.doc2vec import Doc2Vec
from src.sentiment_english.utils.all_utils_package import *
from src.sentiment_english.models.network_fit import *


# def attach_doc2vec(df_file_records_dict, nw_file_records_dict, doc2vec_model):
#     num = 0
#     for i in range(len(df_file_records_dict)):
#         for name in ['entity', 'relation', 'event']:
#             if name in df_file_records_dict:
#                 for j in range(len(df_file_records[i][name])):
#                     df_file_records_dict[i][name][j]['doc2vec'] = doc2vec_model[num]
#                     num += 1
#     for i in range(len(nw_file_records_dict)):
#         for name in ['entity', 'relation', 'event']:
#             if name in nw_file_records_dict:
#                 for j in range(len(df_file_records[i][name])):
#                     df_file_records_dict[i][name][j]['doc2vec'] = doc2vec_model[num]
#                     num += 1

if __name__ == '__main__':
    mode = True  # True:DF,false:NW

    print 'Read data...'
    df_file_records, nw_file_records = \
        read_file_info_records(train_ere_dir, train_entity_info_dir, train_relation_info_dir, train_event_info_dir,
                               train_em_args_dir)
    file_records = df_file_records + nw_file_records
    print 'DF files:', len(df_file_records), ' NW files:', len(nw_file_records)

    # to_dict(df_file_records)
    # to_dict(file_records)
    # attach_doc2vec(df_file_records, nw_file_records, doc2vec_model)

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

    # 正常训练测试评价
    # 重采样
    print 'Resampling...'
    x_train, y_train = up_resampling_3classes(x_train, y_train)

    # 训练
    print 'Train...'
    model = network_fit(x_train, y_train, 3)  # 分三类

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
