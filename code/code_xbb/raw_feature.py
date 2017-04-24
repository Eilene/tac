# -*- coding:utf-8 -*-
from word_count import dictionary,embeddings_index
from pattern.en import sentiment
from pattern.en import lemma
import nltk
import numpy as np

#获取一个句子的feature，是一个矩阵（单词个数＊feature_num),包括每个词的词性，情感极性，主动性，词频，出现词序，词向量
def sentence_feature(sentence,mention_length):
    word = {}
    feature = []
    feature_cata = []
    #分词及词性
    word_token = nltk.pos_tag(nltk.word_tokenize(sentence))
    #词干还原作为key存在word中
    for i in word_token:
        word[lemma(i[0])] = [i[1]]

    #计算词的情感极性和主动性#计算词频和出现的次序
    for i in word:
        word[i].append(mention_length)
        word[i].append(sentiment(i)[0])
        word[i].append(sentiment(i)[1])
        if(i in dictionary):
            word[i].append(dictionary[i][0])
            word[i].append(dictionary[i][1])
        else:
            word[i].append(0)
            word[i].append(0)
        #暂时不用词性，都是数字，先不用做离散化编码
        word[i] = word[i]
        if(i in embeddings_index):
            word[i] += list(embeddings_index[i])
        else:
            word[i] += [0.01]*100

    #返回sentence 对应的feature向量
    for i in word_token:
        feature.append(word[lemma(i[0])][1:])#数值型特征
        feature_cata.append(word[lemma(i[0])][0])
    return [feature,feature_cata]

