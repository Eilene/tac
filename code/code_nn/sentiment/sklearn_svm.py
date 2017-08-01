# coding=utf-8
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn import svm
import nltk
from nltk.stem.wordnet import WordNetLemmatizer
from pattern.en import sentiment
import random

from constants import *
from read_file_info_records import *
from evaluation import *
from write_best import *
from find_source import *
from filter_none_with_stdict import *
from resampling import *

# tfidf, 词性、极性、主动性等分类特征
# 都需要全语料中抽


def get_train_labels(df):
    str_labels = df['label_polarity']
    datanum = len(str_labels)
    labels = [0] * datanum
    for i in range(datanum):
        if str_labels[i] == 'pos':
            labels[i] = 1
    return labels


def get_test_labels(df):
    str_labels = df['label_polarity']
    datanum = len(str_labels)
    # labels = [0] * datanum
    labels = []
    for i in range(datanum):
        if str_labels[i] == 'pos':
            labels.append(2)
        elif str_labels[i] == 'neg':
            labels.append(1)
        else:
            labels.append(0)
    return labels


def gen_train_samples(entity_df, relation_df, event_df):
    features = []
    labels = []

    # labels
    entity_labels = get_train_labels(entity_df)
    relation_labels = get_train_labels(relation_df)
    event_labels = get_train_labels(event_df)
    labels = entity_labels
    labels.extend(relation_labels)
    labels.extend(event_labels)

    # tfidf
    entity_contexts = entity_df['entity_mention_context']
    rel_arg1_contexts = relation_df['rel_arg1_context']
    rel_arg2_contexts = relation_df['rel_arg2_context']
    relation_contexts = []
    for i in range(len(rel_arg1_contexts)):
        context = rel_arg1_contexts[i] + ' ' + rel_arg2_contexts[2]
        relation_contexts.append(context)
    event_trigger_contexts = event_df['trigger_context']
    contexts = entity_contexts.tolist()
    contexts.extend(relation_contexts)
    contexts.extend(event_trigger_contexts.tolist())
    # print contexts
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words='english', max_features=300, binary=True)
    tfidf_features = vec.fit_transform(contexts).toarray()
    tfidf_features = tfidf_features.tolist()
    features = tfidf_features

    # 分类特征

    return features, labels, vec


def gen_test_samples(vec, contexts, df):
    labels = get_test_labels(df)
    features = vec.transform(contexts).toarray()
    features = features.tolist()

    return features, labels


def get_train_records(train_files):  # 拼接起来
    entity_list = []
    relation_list = []
    event_list = []
    for file_info in train_files:
        if 'entity' in file_info:
            entity_df = file_info['entity']
            entity_df = entity_df[entity_df.label_polarity != 'none'].reset_index()  # none的去掉
            entity_list.extend(entity_df.to_dict(orient='records'))
        if 'relation' in file_info:
            relation_df = file_info['relation']
            relation_df = relation_df[relation_df.label_polarity != 'none'].reset_index()  # none的去掉
            relation_list.extend(relation_df.to_dict(orient='records'))
        if 'event' in file_info:
            event_df = file_info['event']
            event_df = event_df[event_df.label_polarity != 'none'].reset_index()  # none的去掉
            event_list.extend(event_df.to_dict(orient='records'))

    all_entity_df = pd.DataFrame(entity_list)
    all_relation_df = pd.DataFrame(relation_list)
    all_event_df = pd.DataFrame(event_list)

    return all_entity_df, all_relation_df, all_event_df


def svm_predict(clf, x_test):
    y_pred_proba = clf.predict_proba(x_test)
    y_pred = []
    for p in y_pred_proba:
        if p[1] - p[0] > 0.1:
            y_pred.append(2)
        elif p[0] - p[1] > 0.1:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred


def test_process(vec, clf, test_files):
    y_test = []
    y_predict = []

    for file_info in test_files:
        if 'entity' in file_info:
            entity_df = file_info['entity']
            contexts = entity_df['entity_mention_context']

            x_entity_test, y_entity_test = gen_test_samples(vec, contexts, file_info['entity'])
            y_test.extend(y_entity_test)

            # 先用词典过滤none，再正负分类
            scores = context_scoring(contexts)
            print 'entity', scores
            y_entity_predict1 = predict_by_scores(scores)  # 过滤none
            y_entity_predict2 = svm_predict(clf, x_entity_test)
            y_entity_predict = [y_entity_predict2[i] if y_entity_predict1[i] != 0 else y_entity_predict1[i] for i in
                                range(len(y_entity_predict1))]
            # print file_info['filename'], y_entity_predict
            y_predict.extend(y_entity_predict)

            # 加入记录
            file_info['entity'] = entity_df.to_dict(orient='records')
            for i in range(len(file_info['entity'])):
                if y_entity_predict[i] == 2:
                    file_info['entity'][i]['predict_polarity'] = 'pos'
                elif y_entity_predict[i] == 1:
                    file_info['entity'][i]['predict_polarity'] = 'neg'
                else:
                    file_info['entity'][i]['predict_polarity'] = 'none'

        if 'relation' in file_info:
            relation_df = file_info['relation']
            rel_arg1_contexts = relation_df['rel_arg1_context']
            rel_arg2_contexts = relation_df['rel_arg2_context']
            contexts = []
            for i in range(len(rel_arg1_contexts)):
                context = rel_arg1_contexts[i] + ' ' + rel_arg2_contexts[i]
                contexts.append(context)

            x_relation_test, y_relation_test = gen_test_samples(vec, contexts, file_info['relation'])
            y_test.extend(y_relation_test)

            # 先用词典过滤none，再正负分类
            scores = context_scoring(contexts)
            print 'relation', scores
            y_relation_predict1 = predict_by_scores(scores)  # 过滤none
            y_relation_predict2 = svm_predict(clf, x_relation_test)
            # y_relation_predict2 = [[x + 1] for x in y_relation_predict2]  # 全加1
            y_relation_predict = [y_relation_predict2[i] if y_relation_predict1[i] != 0 else y_relation_predict1[i]
                                  for i in
                                  range(len(y_relation_predict1))]
            y_predict.extend(y_relation_predict)

            # 加入记录
            file_info['relation'] = relation_df.to_dict(orient='records')
            for i in range(len(file_info['relation'])):
                if y_relation_predict[i] == 2:
                    file_info['relation'][i]['predict_polarity'] = 'pos'
                elif y_relation_predict[i] == 1:
                    file_info['relation'][i]['predict_polarity'] = 'neg'
                else:
                    file_info['relation'][i]['predict_polarity'] = 'none'

        if 'event' in file_info:
            event_df = file_info['event']
            contexts = event_df['trigger_context']

            x_event_test, y_event_test = gen_test_samples(vec, contexts, file_info['event'])
            y_test.extend(y_event_test)

            # 先用词典过滤none，再正负分类
            scores = context_scoring(contexts)
            print 'event', scores
            y_event_predict1 = predict_by_scores(scores)
            y_event_predict2 = svm_predict(clf, x_event_test)
            # y_event_predict2 = [[x + 1] for x in y_event_predict2]  # 全加1
            y_event_predict = [y_event_predict2[i] if y_event_predict1[i] != 0 else y_event_predict1[i] for i in
                               range(len(y_event_predict1))]
            y_predict.extend(y_event_predict)

            # 加入记录
            file_info['event'] = event_df.to_dict(orient='records')
            for i in range(len(file_info['event'])):
                if y_event_predict[i] == 2:
                    file_info['event'][i]['predict_polarity'] = 'pos'
                elif y_event_predict[i] == 1:
                    file_info['event'][i]['predict_polarity'] = 'neg'
                else:
                    file_info['event'][i]['predict_polarity'] = 'none'

    return test_files, y_test, y_predict


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

    # 训练部分
    # 提取特征，生成样本
    print 'Train records extraction...'
    entity_info_df, relation_info_df, event_info_df = get_train_records(train_files)
    print 'Train samples extraction...'
    x_train, y_train, tfidf_model = gen_train_samples(entity_info_df, relation_info_df, event_info_df)
    # 保存tfidf模型
    joblib.dump(tfidf_model, 'tfidf_model.m')
    print 'Train data number:', len(y_train)
    # 训练
    print 'Train...'
    # clf = MultinomialNB()  # 不接受负值
    clf = svm.SVC(probability=True)  # 总是全预测成负的
    clf.fit(x_train, y_train)
    # 保存训练模型
    joblib.dump(clf, 'svm_model.m')

    # 测试部分
    print 'Test...'
    tfidf_model = joblib.load("tfidf_model.m")
    clf = joblib.load('svm_model.m')
    test_files, y_test, y_predict = test_process(tfidf_model, clf, test_files)

    # 测试评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # 寻找源
    print 'Find sources... '
    test_files = find_sources(test_files, source_dir, ere_dir)

    # 写入
    print 'Write into best files...'
    write_best_files(test_files, output_dir)
