# coding=utf-8

from cnn_fit import *
from svm_proba_filter import *
from regression import *
from lstm import *


def cnn_process(x_train, y_train, x_test, y_test):
    # 训练部分
    # 提取特征及标签
    total_clip_length = 56
    embeddings_index, dim = read_embedding_index(glove_100d_path)
    print 'Train samples extraction...'
    without_none(train_files)  # 训练文件去掉none的样本
    x_train = gen_matrix_features(train_files, embeddings_index, dim, total_clip_length)  # 提取特征
    y_train_cnn = [y - 1 for y in y_train]  # 改为0,1
    x_train, y_train_cnn = convert_samples(x_train, y_train_cnn)  # 转换为通道模式
    print 'Train...'
    model = cnn_fit(x_train, y_train_cnn, 2)  # 分正负
    # 测试部分
    # 提取特征及标签
    print 'Test samples extraction...'
    x_test = gen_matrix_features(test_files, embeddings_index, dim, total_clip_length)  # 提取特征
    x_test, y_test = convert_samples(x_test, y_test)
    y_test = y_test.tolist()
    # 测试
    print 'Test...'
    y_proba_cnn = model.predict(x_test)
    y_predict_cnn = predict_by_proba(y_proba_cnn, 0.1)

    return y_predict_cnn


def svm_process(x_train, y_train, x_test, y_test):
    # 训练
    print 'Train...'
    clf = svm.SVC(probability=True)
    clf.fit(x_train, y_train)
    joblib.dump(clf, 'svm_model.m')  # 保存训练模型
    # 测试
    print 'Test...'
    clf = joblib.load('svm_model.m')
    y_proba_svm = clf.predict_proba(x_test)
    y_predict_svm = predict_by_proba(y_proba_svm, 0.1)

    return y_predict_svm


def regression_process(x_train, y_train, x_test, y_test):
    # 训练
    print 'Train...'
    # regr = linear_model.LinearRegression(normalize=True)  # 使用线性回归
    regr = linear_model.RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0])
    regr.fit(x_train, y_train)

    # 测试
    print 'Test...'
    y_predict_regr = regr.predict(X=x_test)  # 预测
    for i in range(len(y_predict_regr)):
        if y_predict_regr[i] < 0.7:
            y_predict_regr[i] = 0
        elif y_predict_regr[i] < 1.7:
            y_predict_regr[i] = 1
        else:
            y_predict_regr[i] = 2
    y_predict_regr = [int(y) for y in y_predict_regr]

    return y_predict_regr


def lstm_process(x_train, y_train, x_test, y_test):
    x_all, embedding_matrix, nb_words = gen_lstm_features(file_records, embeddings_index)
    # 特征分割训练测试集
    trainlen = len(y_train)
    x_train = x_all[:trainlen]
    x_test = x_all[trainlen:]

    # 训练
    print 'Train...'
    model = lstm_fit(x_train, y_train, embedding_matrix, nb_words, 2)

    # 测试
    print 'Test...'
    probabilities = model.predict(x_test)
    y_predict_lstm = predict_by_proba(probabilities, 0.2)

    return y_predict_lstm


if __name__ == '__main__':
    mode = True

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

    # 训练文件去掉none的样本
    without_none(train_files)

    # 预处理和提取标签
    print 'Labels extraction...'
    y_train = get_merged_labels(train_files)  # 只有1,2两类
    y_test = get_merged_labels(test_files)  # 0,1,2三类
    print 'Train data number:', len(y_train)
    print 'Test data number:', len(y_test)
    print 'Train labels:', y_train
    print 'Test labels:', y_test

    # 普通特征
    print "Vector features extraction..."
    x_all = gen_general_features(train_files+test_files)  # 提取特征
    # 特征分割训练测试集
    trainlen = len(y_train)
    x_train = x_all[:trainlen]
    x_test = x_all[trainlen:]
    print 'SVM:'
    y_predict_svm = svm_process(x_train, y_train, x_test, y_test)
    print 'Regression:'
    y_predict_regression = regression_process(x_train, y_train, x_test, y_test)

    # 词向量特征
    embeddings_index, dim = read_embedding_index(glove_100d_path)
    print 'CNN:'
    y_predict_cnn = cnn_process(x_train, y_train, x_test, y_test)
    # print 'Lstm:'
    # y_predict_lstm = lstm_process()
    # y_predict_svm_2levels
    # y_predict_cnn_2levels
    # y_predict_cnn_3classes

    y_predict_filter = filter_none(test_files)

    # 投票，合并
    y_predict = [0] * len(y_test)
    for i in range(len(y_test)):
        y_candid = [y_predict_svm[i], y_predict_cnn[i], y_predict_regression[i], y_predict_filter[i]]
        y_candid = np.asarray(y_candid)
        counts = np.bincount(y_candid)
        print counts
        y = np.argmax(counts)
        print y
        y_predict[i] = y

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    # print 'Filter labels:', y_predict1
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    find_sources(test_files, source_dir, ere_dir)
    # use_annotation_source(test_files)

    # 写入文件
    print 'Write into best files...'
    write_best_files(test_files, predict_dir)




