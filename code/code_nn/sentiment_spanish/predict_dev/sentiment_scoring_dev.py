# coding=utf-8

from src.sentiment_english.utils.read_file_info_records import read_file_info_records
from src.sentiment_english.utils.constants import *
from src.sentiment_english.utils.get_labels import get_merged_labels
from src.sentiment_english.utils.evaluation import evaluation_3classes
from src.sentiment_english.utils.file_records_other_modification import to_dict
from src.sentiment_english.utils.attach_predict_labels import *
from src.sentiment_english.utils.find_source import find_sources
from src.sentiment_english.utils.write_best import write_best_files

import nltk
from nltk import WordNetLemmatizer


# 读词典文件
def read_senti_dic(pos_filename, neg_filename, negation_filename):
    pos_senti_words = []
    pos_file = open(pos_filename)
    word = pos_file.readline()
    while word:
        if word != '\n':
            pos_senti_words.append(word.decode('utf-8'))  # 加了decode速度瞬间变快！！
        word = pos_file.readline()
    pos_file.close()

    neg_senti_words = []
    neg_file = open(neg_filename)
    word = neg_file.readline()
    while word:
        if word != '\n':
            neg_senti_words.append(word.decode('utf-8'))
        word = neg_file.readline()
    neg_file.close()

    negation_words = []
    negation_file = open(negation_filename)
    word = negation_file.readline()
    while word:
        if word != '\n':
            negation_words.append(word.decode('utf-8'))
        word = negation_file.readline()
    negation_file.close()

    return pos_senti_words, neg_senti_words, negation_words


# 打分
def scoring(text, pos_senti_words, neg_senti_words, negation_words):
    score = 0
    sencs = nltk.sent_tokenize(str(text).decode('utf-8'))
    lemm = WordNetLemmatizer()
    words = []
    for senc in sencs:
        words.extend(nltk.word_tokenize(senc))
    for word in words:  # 两个词的怎么办
        # lemmed = lemm.lemmatize(word)
        lemmed = word
        # print lemmed
        if lemmed in pos_senti_words:
            print 'pos', lemmed
            score += 1
        elif lemmed in neg_senti_words:
            print 'neg', lemmed
            score -= 1
        elif lemmed in negation_words:
            print 'negation', lemmed
            score = -score

    print score   # 没有，全0啊，怎么办

    return score


# 预测
def predict_by_score(score):
    pred = 0
    if score >= 1:
        pred = 2  # 正
    elif score <= -1:
        pred = 1  # 负
    return pred


def context_scoring(contexts, pos_senti_words, neg_senti_words, negation_words):
    scores = []
    for text in contexts:
        scores.append(scoring(text, pos_senti_words, neg_senti_words, negation_words))
    return scores


def predict_by_scores(scores):
    pred = []
    for score in scores:
        p = predict_by_score(score)
        pred.append(p)
    return pred


# 集成
def context_scoring_predict(contexts, pos_senti_words, neg_senti_words, negation_words):
    scores = context_scoring(contexts, pos_senti_words, neg_senti_words, negation_words)
    pred = predict_by_scores(scores)
    return pred


def predict_by_senti_dic(file_records, pos_senti_words, neg_senti_words, negation_words):
    pred = []
    for i in range(len(file_records)):
        if 'entity' in file_records[i]:
            # 取上下文
            contexts = file_records[i]['entity']['entity_mention_context7']
            # 打分并预测
            p = context_scoring_predict(contexts, pos_senti_words, neg_senti_words, negation_words)
            pred.extend(p)
        if 'relation' in file_records[i]:
            # 取上下文
            rel_arg1_contexts = file_records[i]['relation']['rel_arg1_context7']
            rel_arg2_contexts = file_records[i]['relation']['rel_arg2_context7']
            contexts = []
            for j in range(len(rel_arg1_contexts)):
                context = str(rel_arg1_contexts[j]) + ' ' + str(rel_arg2_contexts[j])
                contexts.append(context)
            # 打分并预测
            p = context_scoring_predict(contexts, pos_senti_words, neg_senti_words, negation_words)
            pred.extend(p)
        if 'event' in file_records[i]:
            # 取上下文
            contexts = file_records[i]['event']['trigger_context7']
            # 打分并预测
            p = context_scoring_predict(contexts, pos_senti_words, neg_senti_words, negation_words)
            pred.extend(p)
    return pred


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

    # 读取情感词典
    print 'Read sentiment words dicts...'
    pos_senti_words, neg_senti_words, negation_words = read_senti_dic(pos_word_path, neg_word_path, negation_word_path)

    # 只需要用测试数据
    print 'Scoring and predict...'
    y_test = get_merged_labels(test_files)
    y_predict = predict_by_senti_dic(test_files, pos_senti_words, neg_senti_words, negation_words)

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # # y_predict保存至csv
    # if os.path.exists(dev_y_predict_dir) is False:
    #     os.makedirs(dev_y_predict_dir)
    # # 分类器预测的
    # y_predict_df = pd.DataFrame(y_predict, columns=['y_predict'])
    # y_predict_df.to_csv(dev_y_predict_dir+'network_y_predict.csv', index=False)

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