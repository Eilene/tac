# coding=utf-8

import random


def resampling_3classes(x, y):
    x_new = []
    y_new = []

    # 三类样本分开
    pos_index = []
    neg_index = []
    none_index = []
    datanum = len(y)
    for i in range(datanum):
        if y[i] == 2:
            pos_index.append(i)
        elif y[i] == 1:
            neg_index.append(i)
        else:
            none_index.append(i)

    # 均衡采样，有放回采样
    pos = 0
    neg = 0
    none = 0
    pos_num = len(pos_index)
    neg_num = len(neg_index)
    none_num = len(none_index)
    print pos_num, neg_num, none_num

    samplenum = int(neg_num * 3)  # 这个看着调

    for i in range(samplenum):
        flag = random.randint(1, 7)
        if flag == 2 or flag == 3:
            index = random.randint(0, pos_num-1)
            x_new.append(x[pos_index[index]])
            y_new.append(y[pos_index[index]])
            pos += 1
        elif flag == 1 or flag == 4 or flag == 5:
            index = random.randint(0, neg_num-1)
            x_new.append(x[neg_index[index]])
            y_new.append(y[neg_index[index]])
            neg += 1
        else:
            index = random.randint(0, none_num-1)
            x_new.append(x[none_index[index]])
            y_new.append(y[none_index[index]])
            none += 1
    print 'pos vs neg vs none', pos, neg, none

    return x_new, y_new



# 二类，采样
def resampling(x, y):
    x_new = []
    y_new = []

    # 正负样本分开
    pos_index = []
    neg_index = []
    datanum = len(y)
    for i in range(datanum):
        if y[i] == 1:
            pos_index.append(i)
        else:
            neg_index.append(i)

    # 正负样本均衡采样，有放回采样
    pos = 0
    neg = 0
    pos_num = len(pos_index)
    neg_num = len(neg_index)
    samplenum = int(pos_num * 4)
    for i in range(samplenum):
        flag = random.randint(1, 2)
        if flag == 1:
            index = random.randint(0, pos_num-1)
            x_new.append(x[pos_index[index]])
            y_new.append(y[pos_index[index]])
            pos += 1
        else:
            index = random.randint(0, neg_num-1)
            x_new.append(x[neg_index[index]])
            y_new.append(y[neg_index[index]])
            neg += 1
    print 'pos vs neg', pos, neg

    return x_new, y_new