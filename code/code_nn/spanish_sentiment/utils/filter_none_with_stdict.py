# coding=utf-8

import nltk
from nltk import WordNetLemmatizer
from pattern.en import sentiment


# 下面几个函数阈值怎么调合适

def scoring(text):  # 应是有情感词的就分高，不能正负相抵；意在过滤
    score = 0
    sencs = nltk.sent_tokenize(str(text).decode('utf-8'))
    lemm = WordNetLemmatizer()
    words = []
    for senc in sencs:
        words.extend(nltk.word_tokenize(senc))
    for word in words:
        lemmed = lemm.lemmatize(word)
        polarity = sentiment(lemmed)[0]
        if abs(polarity) >= 0.5:
            # print lemmed
            score += 1
    return score


def predict_by_score(score):
    pred = 0
    if score >= 1:
        pred = 1
    return pred


def context_scoring(contexts):
    scores = []
    for text in contexts:
        scores.append(scoring(text))
    return scores


def predict_by_scores(scores):
    pred = [0 if score < 1 else 1 for score in scores]
    return pred


def filter_none(file_records):
    pred = []
    for i in range(len(file_records)):
        if 'entity' in file_records[i]:
            # 取上下文
            contexts = file_records[i]['entity']['entity_mention_context7']
            # 打分
            scores = context_scoring(contexts)
            # 根据分给一份predict
            p = predict_by_scores(scores)
            pred.extend(p)
        if 'relation' in file_records[i]:
            # 取上下文
            rel_arg1_contexts = file_records[i]['relation']['rel_arg1_context7']
            rel_arg2_contexts = file_records[i]['relation']['rel_arg2_context7']
            contexts = []
            for j in range(len(rel_arg1_contexts)):
                context = str(rel_arg1_contexts[j]) + ' ' + str(rel_arg2_contexts[j])
                contexts.append(context)
            # 打分
            scores = context_scoring(contexts)
            # 根据分给一份predict
            p = predict_by_scores(scores)
            pred.extend(p)
        if 'event' in file_records[i]:
            # 取上下文
            contexts = file_records[i]['event']['trigger_context7']
            # 打分
            scores = context_scoring(contexts)
            # 根据分给一份predict
            p = predict_by_scores(scores)
            pred.extend(p)
    return pred


def filter_none_with_window_text(file_records):
    pred = []
    for i in range(len(file_records)):
        if 'entity' in file_records[i]:
            # 取上下文
            contexts = file_records[i]['entity']['window_text']
            # print contexts
            # 打分
            scores = context_scoring(contexts)
            # 根据分给一份predict
            p = predict_by_scores(scores)
            pred.extend(p)
        if 'relation' in file_records[i]:
            # 取上下文
            rel_arg1_contexts = file_records[i]['relation']['rel_arg1_window_text']
            rel_arg2_contexts = file_records[i]['relation']['rel_arg2_window_text']
            contexts = []
            for j in range(len(rel_arg1_contexts)):
                context = str(rel_arg1_contexts[j]) + ' ' + str(rel_arg2_contexts[j])
                print rel_arg1_contexts[j]
                contexts.append(context)
            # print contexts
            # 打分
            scores = context_scoring(contexts)
            # 根据分给一份predict
            p = predict_by_scores(scores)
            pred.extend(p)
        if 'event' in file_records[i]:
            # 取上下文
            contexts = file_records[i]['event']['trigger_window_text']
            # print contexts
            # 打分
            scores = context_scoring(contexts)
            # 根据分给一份predict
            p = predict_by_scores(scores)
            pred.extend(p)
    return pred
