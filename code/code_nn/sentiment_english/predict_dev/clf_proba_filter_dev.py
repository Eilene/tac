# coding=utf-8

from sklearn import svm
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression

from src.sentiment_english.features.general_features import *
from src.sentiment_english.utils.attach_predict_labels import attach_predict_labels
from src.sentiment_english.utils.constants import *
from src.sentiment_english.utils.evaluation import evaluation_3classes
from src.sentiment_english.utils.file_records_other_modification import without_none, to_dict
from src.sentiment_english.utils.filter_none_with_stdict import filter_none
from src.sentiment_english.utils.get_labels import get_merged_labels
from src.sentiment_english.utils.predict_by_proba import *
from src.sentiment_english.utils.read_file_info_records import *
from src.sentiment_english.utils.write_best import write_best_files
from src.sentiment_english.utils.find_source import find_sources
from src.sentiment_english.utils.read_imdb_data import read_imdb_data


if __name__ == '__main__':
    mode = True  # True:DF,false:NW
    clf_name = 'lr'  # 可选lr，svm等

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
    print "Samples extraction..."
    without_none(train_files)  # 训练文件去掉none的样本
    # x_all = gen_general_features(train_files+test_files)  # 提取特征
    y_train = get_merged_labels(train_files)  # 只有1,2两类
    y_test = get_merged_labels(test_files)  # 0,1,2三类
    # 特征分割训练测试集
    trainlen = len(y_train)
    # x_train = x_all[:trainlen]
    # x_test = x_all[trainlen:]
    print 'Train data number:', len(y_train)
    print 'Test data number:', len(y_test)
    print 'Train labels:', y_train
    print 'Test labels:', y_test

    # tfidf
    contexts = get_contexts(train_files+test_files)
    imdb_contexts, y_imdb = read_imdb_data(imdb_dir)
    x_all = gen_all_general_features(imdb_contexts+contexts)
    y_imdb.extend(y_train)
    y_train = y_imdb
    print y_train
    trainnum = len(y_train)
    x_train = x_all[:trainnum]
    x_test = x_all[trainnum:]

    # 训练
    print 'Train...'
    # clf = MultinomialNB()  # 不接受负值
    if clf_name == 'svm':
        clf = svm.SVC(probability=True)
    else:
        clf = LogisticRegression()
    clf.fit(x_train, y_train)
    # joblib.dump(clf, clf_name+'_model.m')  # 保存训练模型

    # 测试
    print 'Test...'
    # clf = joblib.load(clf_name+'_model.m')
    y_pred_proba = clf.predict_proba(x_test)
    y_predict_clf = predict_by_proba_3classes_threshold(y_pred_proba, 0.1, 0.4)
    # 测试文件根据打分过滤掉none的样本
    # y_predict_filter = filter_none(test_files)
    # y_predict = [y_predict_clf[i] if y_predict_filter[i] != 0 else 0 for i in range(len(y_predict_clf))]
    y_predict = y_predict_clf

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    # print 'Clf predict labels: ', y_predict_clf
    # print 'Filter labels:', y_predict_filter
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # y_predict保存至csv
    if os.path.exists(dev_y_predict_dir) is False:
        os.makedirs(dev_y_predict_dir)
    # 分类器预测的
    y_predict_df = pd.DataFrame(y_predict_clf, columns=['y_predict'])
    y_predict_df.to_csv(dev_y_predict_dir+clf_name+'_y_predict.csv', index=False)
    # 词典过滤的
    # y_predict1_df = pd.DataFrame(y_predict_filter, columns=['y_predict'])
    # y_predict1_df.to_csv(dev_y_predict_dir + 'filter_y_predict.csv', index=False)

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    find_sources(test_files, train_source_dir, train_ere_dir)
    # test_files = use_annotation_source(test_files)

    # 写入文件
    print 'Write into best files...'
    write_best_files(test_files, dev_predict_dir)



