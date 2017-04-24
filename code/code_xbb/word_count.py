# -*- coding:utf-8 -*-
#求整个语料集合上的词频，返回dic，key 是还原后的词，val = [频数，出现排名]
import nltk
import numpy as np
from pattern.en import lemma
import os
from Constant import *

dir = os.listdir(Source_path)
dictionary = {}
for f in dir:
    #先只处理论坛数据
    if(f[-1] != 't'):
        continue
    doc = open(Source_path+f, 'r').read().decode('utf-8')

    # 先做分词和词性还原
    word = nltk.word_tokenize(doc)
    for i in range(0,len(word)):
        word[i] = lemma(word[i])
    fredist = nltk.FreqDist(word)
    for i in fredist.keys():
        if i in dictionary.keys():
            dictionary[i] += fredist[i]
        else:
            dictionary[i] = fredist[i]

word_list = (sorted(dictionary.items(), reverse=True, key = lambda x:x[1])) # 根据词频字典值排序，并打印
dictionary = {}

for i in enumerate(word_list):
    dictionary[i[1][0]] = [i[0],i[1][1]]

embeddings_index = {}
f = open(Wordvec_path)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
