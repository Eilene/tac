# -*- coding:UTF-8 -*-
from Constant import *
import pandas as pd
from raw_feature import sentence_feature
import numpy as np

def get_feature():

    raw_df = pd.read_csv(Write_path)
    raw_df = raw_df.loc[:, ['entity_type', 'entity_specificity', 'mention_type', 'mention_length',
                            'pre_sentence', 'now_sentence', 'next_sentence', 'label_polarity', 'file']]
    TrainX = []
    TrainX_cata = []
    TrainY = []

    for i in raw_df.values:
        # 计算数值特征
        # 截断补齐,不是单个句子截断，而是三个句子拼在一起，把含有target的句子放在最前面
        temp = sentence_feature(i[4],int(i[3]))[0] + sentence_feature(i[5],int(i[3]))[0] + sentence_feature(i[6],int(i[3]))[0]
        if (len(temp) > Sentence_length):
            temp = temp[:Sentence_length]
        else:
            temp = temp + ([[0.01] * num_feature_num]) * (Sentence_length - len(temp))

        #计算类别特征
        temp_cata = sentence_feature(i[4],int(i[3]))[1] + sentence_feature(i[5],int(i[3]))[1] + sentence_feature(i[6],int(i[3]))[1]
        if(len(temp_cata) < Sentence_length):
            temp_cata = temp_cata + ['None']*(Sentence_length-len(temp_cata))
        for j in range(0,Sentence_length):
            TrainX_cata.append([i[0],i[1],i[2]]+[temp_cata[j]])

        # 先不考虑类别特征和窗口，只靠三个句子带来的信息
        TrainX.append(temp)
        if (i[7] == "None"):
            TrainY.append([[1, 0, 0]])
        elif (i[7] == "pos"):
            TrainY.append([[0, 1, 0]])
        else:
            TrainY.append([[0, 0, 1]])

    # 对类别特征独热编码
    TrainX_cata = pd.DataFrame(TrainX_cata)
    TrainX_cata = pd.get_dummies(TrainX_cata)
    TrainX_cata = TrainX_cata.values
    cata_feature_num = TrainX_cata.shape[1]

    count = 0
    for i in range(0, len(TrainX)):
        for j in range(0, len(TrainX[i])):
            TrainX[i][j] = TrainX[i][j] + list(TrainX_cata[count])
            count += 1

    TrainX = np.array(TrainX)
    TrainY = np.array(TrainY)

    return TrainX,TrainY,cata_feature_num