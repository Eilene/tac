# coding=utf-8

import random


# 重采样
# 0:none，1:neg，2:pos
# neg采样至与none一样多，pos与neg保持比例不变
# x为特征列表，y为标签列表
def up_resampling_3classes(x, y):
    samples = []

    # 三类样本分开
    pos_index = []
    neg_index = []
    datanum = len(y)
    for i in range(datanum):
        if y[i] == 2:
            pos_index.append(i)
        elif y[i] == 1:
            neg_index.append(i)
        else:  # none样本直接放入
            samples.append([x[i], 0])

    # 有放回采样
    pos_num = len(pos_index)
    neg_num = len(neg_index)
    none_num = len(samples)
    print 'Before resampling pos vs neg vs none:', pos_num, neg_num, none_num
    neg_samplenum = none_num
    # pos_samplenum = none_num
    pos_samplenum = int(none_num*(float(pos_num)/float(neg_num)))
    print 'After resampling pos vs neg vs none:', pos_samplenum, neg_samplenum, none_num
    # 正样本
    for i in range(pos_samplenum):
        index = random.randint(0, pos_num - 1)
        samples.append([x[pos_index[index]],y[pos_index[index]]])
    # 负样本
    for i in range(neg_samplenum):
        index = random.randint(0, neg_num - 1)
        samples.append([x[neg_index[index]], y[neg_index[index]]])

    # 打乱样本顺序，使其均匀
    random.shuffle(samples)

    # 拆分特征和标签
    x_new = []
    y_new = []
    for sample in samples:
        x_new.append(sample[0])
        y_new.append(sample[1])

    return x_new, y_new


# 标签0,1,2
def resampling_3classes(x, y, pos_samplenum=-1, neg_samplenum=-1, none_samplenum=-1):
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
    print 'Before resampling pos vs neg vs none:', pos_num, neg_num, none_num

    if pos_samplenum == -1:  # 使用默认值
        pos_samplenum = neg_num
    if neg_samplenum == -1:
        neg_samplenum = pos_samplenum
    if none_samplenum == -1:
        none_samplenum = pos_samplenum

    samplenum = pos_samplenum + neg_samplenum + none_samplenum
    for i in range(samplenum):
        flag = random.randint(1, samplenum)
        if flag <= pos_samplenum:
            index = random.randint(0, pos_num-1)
            x_new.append(x[pos_index[index]])
            y_new.append(y[pos_index[index]])
            pos += 1
        elif flag <= pos_samplenum + neg_samplenum:
            index = random.randint(0, neg_num-1)
            x_new.append(x[neg_index[index]])
            y_new.append(y[neg_index[index]])
            neg += 1
        else:
            index = random.randint(0, none_num-1)
            x_new.append(x[none_index[index]])
            y_new.append(y[none_index[index]])
            none += 1
    print 'After resampling pos vs neg vs none', pos, neg, none

    return x_new, y_new


# 二类，采样，标签需1,0
def resampling_2classes(x, y, pos_samplenum=-1, neg_samplenum=-1):
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
    print 'Before resampling 1 vs 0', pos_num, neg_num

    if pos_samplenum == -1:  # 使用默认值
        pos_samplenum = int((pos_num + neg_num)/2)
    if neg_samplenum == -1:  # 使用默认值
        neg_samplenum = pos_samplenum

    samplenum = pos_samplenum + neg_samplenum
    for i in range(samplenum):
        flag = random.randint(1, samplenum)
        if flag <= pos_samplenum:
            index = random.randint(0, pos_num-1)
            x_new.append(x[pos_index[index]])
            y_new.append(y[pos_index[index]])
            pos += 1
        else:
            index = random.randint(0, neg_num-1)
            x_new.append(x[neg_index[index]])
            y_new.append(y[neg_index[index]])
            neg += 1
    print 'After resampling 1 vs 0', pos, neg

    return x_new, y_new
