# coding=utf-8

import random


# 重采样
# 0:none，1:neg，2:pos
# neg采样至与none一样多，pos与neg保持比例不变
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
