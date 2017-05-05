# -*- coding:UTF-8 -*-
from Constant import *
import pandas as pd
import numpy as np
# from word_count import dictionary,embeddings_index
from word_count import dictionary,embeddings_index
from pattern.en import sentiment
from pattern.en import lemma
import nltk

#先不考虑类别信息，只用词向量
#获取一个句子的feature，是一个矩阵（单词个数＊feature_num),包括每个词的词性，情感极性，主动性，词频，出现词序，词向量
def sentence_feature(sentence):
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
        # word[i].append(sentiment(i)[0])
        # word[i].append(sentiment(i)[1])
        # if(i in dictionary):
        #     word[i].append(dictionary[i][0])
        #     word[i].append(dictionary[i][1])
        # else:
        #     word[i].append(0)
        #     word[i].append(0)

        if(i in embeddings_index):
            word[i] += list(embeddings_index[i])
        else:
            word[i] += [0.01]*100

    #返回sentence 对应的feature向量
    for i in word_token:
        feature.append(word[lemma(i[0])][1:])#数值型特征
        feature_cata.append(word[lemma(i[0])][0])
    #截断补齐
    if(len(feature) < Sentence_length):
        feature += [[0.1]*(100)]*(Sentence_length - len(feature))
    if(len(feature) > Sentence_length):
        feature = feature[:Sentence_length]
    return [feature,feature_cata]

def get_feature(mode,path):
    df = pd.read_csv(path)
    Train_X = []
    #TrainX_cata = []
    Train_Y = []
    count1 = 0
    count2 = 0
    for i in df.values:
        if(mode==2):
            if(i[-1] == 'neg'):
                count1 += 1
                if(count1 > 1000):
                    continue
            if(i[-1] == 'pos'):
                count2 += 1
                if(count2 > 1000):
                    continue
        if(mode == 2 and (i[-1]=='neg' or i[-1] == 'pos')):
            Train_X.append(sentence_feature(i[0])[0])
        # temp_cata = sentence_feature(i[0])[1]
        # if (len(temp_cata) < Sentence_length):
        #     temp_cata = temp_cata + ['None'] * (Sentence_length - len(temp_cata))
        # for j in range(0, Sentence_length):
        #     TrainX_cata.append([i[3], i[4], i[7]] + [temp_cata[j]])
        if(mode == 1):
            if (i[-1] == 'None'):
                Train_Y.append([0])
            else:
                Train_Y.append([1])
        else:
            if (i[-1] == 'neg'):
                Train_Y.append([0])
            if (i[-1] == 'pos'):
                Train_Y.append([1])

    #对类别特征独热编码
    # TrainX_cata = pd.DataFrame(TrainX_cata)
    # TrainX_cata = pd.get_dummies(TrainX_cata)
    # TrainX_cata = TrainX_cata.values
    # cata_feature_num = TrainX_cata.shape[1]

    # count = 0
    # for i in range(0, len(Train_X)):
    #     for j in range(0, len(Train_X[i])):
    #         Train_X[i][j] = Train_X[i][j] + list(TrainX_cata[count])+[0] *(54- cata_feature_num)
    #         count += 1

    #类别个数不一致，应该先所有数据提特征，不能部分文件提，不然one-hotencoding后数据维数不一致,现在先用58


    #转化成单通道输入方式
    Train_Y = np.array(Train_Y)
    print np.array(Train_X).shape,Train_Y.shape
    data = np.empty((Train_Y.shape[0], Sentence_length, 100, 1), dtype='float32')
    #Train_X = sequence.pad_sequences(Train_X, maxlen=Sentence_length)
    cnt = 0
    for i in Train_X:
        data[cnt, :, :, 0] = i
        cnt += 1
    Train_X = data
    return Train_X,Train_Y