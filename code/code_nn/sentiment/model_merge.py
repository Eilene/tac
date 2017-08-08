# coding=utf-8

# 都至test_file状态，投票;可将每种方法封装，返回test_files
# 还需一份合起来的y_predict和y_test，以做评价

# 先合cnn和svm
from cnn import *
from sklearn_svm import *


def cnn_process(train_files, test_files):
    # 训练部分
    # 提取特征，生成样本
    clip_length = 40
    embeddings_index, dim = read_embedding_index(glove_100d_path)
    print 'Train samples extraction...'
    x_train, y_train = get_train_samples(train_files, embeddings_index, dim, clip_length)
    print 'Train data number:', len(y_train)
    # cnn训练
    print 'Train...'
    model = cnn_fit(x_train, y_train)  # 分正负

    # 测试部分
    print 'Test...'
    test_files, y_test, y_predict = test_process(model, test_files, embeddings_index, dim, clip_length)

    return y_predict, y_test


def svm_process(train_files, test_files):
    # 训练文件去掉none的样本
    train_files = without_none(train_files)

    # 提取特征及标签
    print "Generate samples..."
    x_train, y_train, x_test, y_test = gen_samples(train_files, test_files)
    print 'Train data number:', len(y_train)
    print 'Test data number:', len(y_test)

    # 训练
    print 'Train...'
    # clf = MultinomialNB()  # 不接受负值
    clf = svm.SVC(probability=True)  # 总是全预测成负的
    print len(x_train), len(x_train[0])
    print len(y_train)
    clf.fit(x_train, y_train)
    # 保存训练模型
    joblib.dump(clf, 'svm_model.m')

    # 测试
    print 'Test...'
    clf = joblib.load('svm_model.m')
    y_predict = svm_predict(clf, x_test)

    return y_predict, y_test


def attach_predict_labels(test_files, y_predict):
    count = 0
    for i in range(len(test_files)):
        if 'entity' in test_files[i]:
            # 转字典
            test_files[i]['entity'] = test_files[i]['entity']
            # 加上label
            for j in range(len(test_files[i]['entity'])):
                if y_predict[count] == 0:
                    test_files[i]['entity'][j]['predict_polarity'] = 'none'
                elif y_predict[count] == 1:
                    test_files[i]['entity'][j]['predict_polarity'] = 'neg'
                else:
                    test_files[i]['entity'][j]['predict_polarity'] = 'pos'
                count += 1
        if 'relation' in test_files[i]:
            # 转字典
            test_files[i]['relation'] = test_files[i]['relation']
            # 加上label
            for j in range(len(test_files[i]['relation'])):
                if y_predict[count] == 0:
                    test_files[i]['relation'][j]['predict_polarity'] = 'none'
                elif y_predict[count] == 1:
                    test_files[i]['relation'][j]['predict_polarity'] = 'neg'
                else:
                    test_files[i]['relation'][j]['predict_polarity'] = 'pos'
                count += 1
        if 'event' in test_files[i]:
            # 转字典
            test_files[i]['event'] = test_files[i]['event']
            # 加上label
            for j in range(len(test_files[i]['event'])):
                if y_predict[count] == 0:
                    test_files[i]['event'][j]['predict_polarity'] = 'none'
                elif y_predict[count] == 1:
                    test_files[i]['event'][j]['predict_polarity'] = 'neg'
                else:
                    test_files[i]['event'][j]['predict_polarity'] = 'pos'
                count += 1
    return test_files


if __name__ == '__main__':
    # 读取各文件中间信息
    print 'Read data...'
    file_info_records = read_file_info_records(ere_dir, entity_info_dir, relation_info_dir, event_info_dir, em_args_dir)

    # 按文件划分训练和测试集
    print 'Split into train and test dataset...'
    portion = 0.8
    trainnum = int(len(file_info_records) * 0.8)
    train_files = file_info_records[:trainnum]
    test_files = file_info_records[trainnum:]

    # 预测
    # svm，cnn，过滤none，都返回合起来的predict
    y_predict_filter = filter_none(test_files)
    y_predict_svm, y_test_svm = svm_process(train_files, test_files)
    y_predict_cnn, y_test_cnn = cnn_process(train_files, test_files)
    print y_test_cnn == y_predict_svm
    y_test = y_test_cnn
    # 几种predict通过投票融合（怎样融合？？应该在前面概率融合或协同训练，加权？？这里先随便试下）
    y_predict = [0] * len(y_predict_cnn)
    for i in range(len(y_predict)):
        if y_predict_filter[i] != 0:
            if y_predict_cnn[i] == y_predict_svm[i]:
                y_predict[i] = y_predict_cnn[i]
            else:
                y_predict[i] = max(y_predict_cnn[i], y_predict_svm[i])  # 倾向于有，倾向于正，估计不好

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Filter labels:', y_predict_filter
    print 'cnn labels:', y_predict_cnn
    print 'svm labels:', y_predict_svm
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # 测试结果写入记录
    test_files = attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    test_files = find_sources(test_files, source_dir, ere_dir)

    # 写入文件
    print 'Write into best files...'
    write_best_files(test_files, predict_dir)



