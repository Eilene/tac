# coding=utf-8

import nltk

from pattern.en import lemma
from pattern.en import sentiment

from sklearn import svm
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from utils.evaluation import *
from utils.filter_none_with_stdict import *
from utils.find_source import *
from utils.read_file_info_records import *
from utils.resampling import *
from utils.write_best import *
from utils.constants import *


def get_train_labels(str_labels):
    datanum = len(str_labels)
    labels = [0] * datanum
    for i in range(datanum):
        if str_labels[i] == 'pos':
            labels[i] = 1
    return labels


def get_test_labels(str_labels):
    datanum = len(str_labels)
    labels = [0] * datanum
    for i in range(datanum):
        if str_labels[i] == 'pos':
            labels[i] = 2
        elif str_labels[i] == 'neg':
            labels[i] = 1
        else:
            labels[i] = 0
    return labels


def gen_samples(train_files, test_files):
    # 特征

    # 按文件顺序合并
    file_records = train_files + test_files
    contexts = []
    texts = []
    for file_info in file_records:
        if 'entity' in file_info:
            entity_df = file_info['entity']
            entity_contexts = entity_df['entity_mention_context']
            entity_texts = entity_df['entity_mention_text']
            contexts.extend(entity_contexts.tolist())
            texts.extend(entity_texts.tolist())
        if 'relation' in file_info:
            relation_df = file_info['relation']
            rel_arg1_contexts = relation_df['rel_arg1_context']
            rel_arg2_contexts = relation_df['rel_arg2_context']
            relation_contexts = []
            rel_arg1_texts = relation_df['rel_arg1_text']
            rel_arg2_texts = relation_df['rel_arg2_text']
            relation_texts = []
            for i in range(len(rel_arg1_contexts)):
                context = rel_arg1_contexts[i] + ' ' + rel_arg2_contexts[i]
                relation_contexts.append(context)
                text = rel_arg1_texts[i] + ' ' + rel_arg2_texts[i]
                relation_texts.append(text)
            contexts.extend(relation_contexts)
            texts.extend(relation_texts)
        if 'event' in file_info:
            event_df = file_info['event']
            event_contexts = event_df['trigger_context']
            event_texts = event_df['trigger_text']
            contexts.extend(event_contexts.tolist())
            texts.extend(event_texts.tolist())

    # tfidf
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words='english', max_features=300, binary=True)
    tfidf_features = vec.fit_transform(contexts).toarray()
    tfidf_features = tfidf_features.tolist()
    features = tfidf_features

    # 情感极性，主动性，词性
    features_cata = []  # 存放类别特征
    reserved_dim = 10  # 统一维数
    for i in range(len(texts)):
        # 词性
        pos = nltk.pos_tag(nltk.word_tokenize(texts[i]))
        length = len(pos)

        # 情感极性，主动性
        polarity = []
        for j in range(length):
            lemmed = lemma(pos[j][0])
            polarity.append(sentiment(lemmed)[0])  # 极性
            polarity.append(sentiment(lemmed)[1])  # 主动性
        # print polarity
        # 统一维数
        while length < reserved_dim:
            pos.append(('', ''))
            polarity.append(0.0)
            polarity.append(0.0)
            length = len(pos)
        if length > reserved_dim:
            pos = pos[:reserved_dim]
            polarity = polarity[:reserved_dim * 2]

        # 词性加入分类特征
        feature_cata = []
        for j in range(reserved_dim):
            feature_cata.append(pos[j][1])
        features_cata.append(feature_cata)

        features[i].extend(polarity)  # 极性用数值特征

    # 独热编码
    features_cata = pd.DataFrame(features_cata)
    features_cata = pd.get_dummies(features_cata)
    features_cata = features_cata.values
    print features_cata.shape
    features_cata = features_cata.tolist()

    # 合并
    for i in range(len(features)):
        features[i].extend(features_cata[i])
    print 'Features:', len(features), len(features[0])

    # 标签
    str_y_train = []
    str_y_test = []
    for file_info in train_files:
        if 'entity' in file_info:
            str_labels = file_info['entity']['label_polarity']
            str_y_train.extend(str_labels)
        if 'relation' in file_info:
            str_labels = file_info['relation']['label_polarity']
            str_y_train.extend(str_labels)
        if 'event' in file_info:
            str_labels = file_info['event']['label_polarity']
            str_y_train.extend(str_labels)
    for file_info in test_files:
        if 'entity' in file_info:
            str_labels = file_info['entity']['label_polarity']
            str_y_test.extend(str_labels)
        if 'relation' in file_info:
            str_labels = file_info['relation']['label_polarity']
            str_y_test.extend(str_labels)
        if 'event' in file_info:
            str_labels = file_info['event']['label_polarity']
            str_y_test.extend(str_labels)
    y_train = get_train_labels(str_y_train)
    y_test = get_test_labels(str_y_test)

    # 特征训练测试分开
    trainnum = len(y_train)
    x_train = features[:trainnum]
    x_test = features[trainnum:]

    # x_train = np.asarray(x_train)
    # x_test = np.asarray(x_test)
    # y_train = np.asarray(y_train)
    # y_test = np.asarray(y_test)

    return x_train, y_train, x_test, y_test


def without_none(file_records):
    for i in range(len(file_records)):
        if 'entity' in file_records[i]:
            file_records[i]['entity'] = file_records[i]['entity'][file_records[i]['entity'].label_polarity != 'none'].reset_index()
        if 'relation' in file_records[i]:
            file_records[i]['relation'] = file_records[i]['relation'][file_records[i]['relation'].label_polarity != 'none'].reset_index()
        if 'event' in file_records[i]:
            file_records[i]['event'] = file_records[i]['event'][file_records[i]['event'].label_polarity != 'none'].reset_index()
    return file_records


def filter_none(file_records):
    pred = []
    for i in range(len(file_records)):
        if 'entity' in file_records[i]:
            # 取上下文
            contexts = file_records[i]['entity']['entity_mention_context']
            # 打分
            scores = context_scoring(contexts)
            # 根据分给一份predict
            p = predict_by_scores(scores)
            pred.extend(p)
        if 'relation' in file_records[i]:
            # 取上下文
            rel_arg1_contexts = file_records[i]['relation']['rel_arg1_context']
            rel_arg2_contexts = file_records[i]['relation']['rel_arg2_context']
            contexts = []
            for j in range(len(rel_arg1_contexts)):
                context = rel_arg1_contexts[j] + ' ' + rel_arg2_contexts[j]
                contexts.append(context)
            # 打分
            scores = context_scoring(contexts)
            # 根据分给一份predict
            p = predict_by_scores(scores)
            pred.extend(p)
        if 'event' in file_records[i]:
            # 取上下文
            contexts = file_records[i]['event']['trigger_context']
            # 打分
            scores = context_scoring(contexts)
            # 根据分给一份predict
            p = predict_by_scores(scores)
            pred.extend(p)
    return pred


def attach_predict_labels(test_files, y_predict):
    count = 0
    for i in range(len(test_files)):
        if 'entity' in test_files[i]:
            # 转字典
            test_files[i]['entity'] = test_files[i]['entity'].to_dict(orient='records')
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
            test_files[i]['relation'] = test_files[i]['relation'].to_dict(orient='records')
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
            test_files[i]['event'] = test_files[i]['event'].to_dict(orient='records')
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


def svm_predict(clf, x_test):
    y_pred_proba = clf.predict_proba(x_test)
    y_pred = []
    for p in y_pred_proba:
        if p[1] - p[0] > 0.2:
            y_pred.append(2)
        elif p[0] - p[1] > 0.2:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred


if __name__ == '__main__':
    mode = True  # True:DF,false:NW

    # 读取各文件中间信息
    print 'Read data...'
    df_file_records, nw_file_records = \
        read_file_info_records(ere_dir, entity_info_dir, relation_info_dir, event_info_dir, em_args_dir)
    print len(df_file_records), len(nw_file_records)

    # DF全部作为训练数据，NW分成训练和测试数据, 合并训练的NW和DF，即可用原来流程进行训练测试
    if mode is True:
        print '**DF**'
        print 'Split into train and test dataset...'
        portion = 0.8
        trainnum = int(len(df_file_records) * 0.8)
        train_files = df_file_records[:trainnum]
        test_files = df_file_records[trainnum:]
    else:
        print '**NW**'
        print 'Merge and split into train and test dataset...'
        portion = 0.2
        nw_trainnum = int(len(nw_file_records) * portion)
        train_files = df_file_records + nw_file_records[:nw_trainnum]
        test_files = nw_file_records[nw_trainnum:]
        print nw_trainnum

    # 训练文件去掉none的样本
    train_files = without_none(train_files)

    # 提取特征及标签
    print "Generate samples..."
    x_train, y_train, x_test, y_test = gen_samples(train_files, test_files)
    print 'Train data number:', len(y_train)
    print 'Test data number:', len(y_test)
    print 'Test labels:', y_test

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
    # tfidf_model = joblib.load("tfidf_model.m")
    clf = joblib.load('svm_model.m')
    y_predict = svm_predict(clf, x_test)
    # 测试文件根据打分过滤掉none的样本
    y_predict1 = filter_none(test_files)
    y_predict = [y_predict[i] if y_predict1[i] != 0 else y_predict1[i] for i in range(len(y_predict))]

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    # print 'Filter labels:', y_predict1
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


