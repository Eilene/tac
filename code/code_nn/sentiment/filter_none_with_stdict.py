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

