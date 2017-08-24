# coding=utf-8

# imdb 外部数据 + 训练集数据，训练doc2vec
# 再分类器训练
# 算特征，也许应放features里？？
import logging

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from src.sentiment.utils.attach_predict_labels import attach_predict_labels
from src.sentiment.utils.constants import *
from src.sentiment.utils.evaluation import evaluation_3classes
from src.sentiment.utils.file_records_other_modification import to_dict
from src.sentiment.utils.file_records_other_modification import without_none
from src.sentiment.utils.get_labels import get_merged_labels
from src.sentiment.utils.predict_by_proba import *
from src.sentiment.utils.read_file_info_records import *
from src.sentiment.utils.write_best import write_best_files
from src.sentiment.utils.find_source import find_sources

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def clean_text(corpus):
    punctuation = """.,?!:;(){}[]"""
    corpus = [z.lower().replace('\n','') for z in corpus]
    corpus = [z.replace('<br />', ' ') for z in corpus]
    for c in punctuation:
        corpus = [z.replace(c, ' %s '%c) for z in corpus]
    corpus = [z.split() for z in corpus]
    return corpus


def labelize_reviews(reviews, label_type):
    labelized = []
    for i, v in enumerate(reviews):
        label = '%s_%s'  % (label_type, i)
        labelized.append(TaggedDocument(v, [label]))
    return labelized


def get_dataset(train_files, test_files):
    x_train = get_merged_context(train_files)
    x_test = get_merged_context(test_files)
    x_train = clean_text(x_train)
    x_test = clean_text(x_test)
    x_train = labelize_reviews(x_train, 'TRAIN')
    x_test = labelize_reviews(x_test, 'TEST')
    print x_train

    y_train = get_merged_labels(train_files)
    y_test = get_merged_labels(test_files)

    return x_train, x_test, y_train, y_test


def get_merged_context(file_records):
    contexts = []

    for fr in file_records:
        if 'entity' in fr:
            entity_contexts = fr['entity']['entity_mention_context'].values.tolist()
            contexts.extend(entity_contexts)
        if 'relation' in fr:
            rel_arg1_contexts = fr['relation']['rel_arg1_context']
            rel_arg2_contexts = fr['relation']['rel_arg2_context']
            relation_contexts = []
            for j in range(len(rel_arg1_contexts)):
                context = rel_arg1_contexts[j] + ' ' + rel_arg2_contexts[j]
                relation_contexts.append(context)
            contexts.extend(relation_contexts)
        if 'event' in fr:
            event_contexts = fr['event']['trigger_context'].values.tolist()
            contexts.extend(event_contexts)

    return contexts


# 读取向量
def get_vecs(model, corpus, size):
    vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)


# 对数据进行训练
def train(x_train, x_test, size=400, epoch_num=10):
    # 实例DM和DBOW模型
    model_dm = Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, workers=3)
    model_dbow = Doc2Vec(min_count=1, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

    # 使用所有的数据建立词典
    model_dm.build_vocab(np.concatenate((x_train, x_test)))
    model_dbow.build_vocab(np.concatenate((x_train, x_test)))

    # 进行多次重复训练，每一次都需要对训练数据重新打乱，以提高精度
    all_train_reviews = x_train
    for epoch in range(epoch_num):
        perm = np.random.permutation(all_train_reviews.shape[0])
        model_dm.train(all_train_reviews[perm])
        model_dbow.train(all_train_reviews[perm])

    # 训练测试数据集
    x_test = np.array(x_test)
    for epoch in range(epoch_num):
        perm = np.random.permutation(x_test.shape[0])
        model_dm.train(x_test[perm])
        model_dbow.train(x_test[perm])

    return model_dm, model_dbow


# 将训练完成的数据转换为vectors
def get_vectors(model_dm, model_dbow, x_train, x_test):

    # 获取训练数据集的文档向量
    train_vecs_dm = get_vecs(model_dm, x_train, size)
    train_vecs_dbow = get_vecs(model_dbow, x_train, size)
    train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
    # 获取测试数据集的文档向量
    test_vecs_dm = get_vecs(model_dm, x_test, size)
    test_vecs_dbow = get_vecs(model_dbow, x_test, size)
    test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

    return train_vecs,test_vecs


# 使用分类器对文本向量进行分类训练
def classifier(train_vecs, y_train, test_vecs, y_test):
    # 使用sklearn的SGD分类器
    from sklearn.linear_model import SGDClassifier

    lr = SGDClassifier(loss='log', penalty='l1')
    lr.fit(train_vecs, y_train)

    print 'Test Accuracy: %.2f' % lr.score(test_vecs, y_test)

    return lr


if __name__ == '__main__':
    mode = True  # True:DF,false:NW

    # 读取各文件中间信息
    print 'Read data...'
    df_file_records, nw_file_records = \
        read_file_info_records(ere_dir, entity_info_dir, relation_info_dir, event_info_dir, em_args_dir)
    print 'DF files:', len(df_file_records), ' NW files:', len(nw_file_records)

    # DF全部作为训练数据，NW分成训练和测试数据, 合并训练的NW和DF，即可用原来流程进行训练测试
    if mode is True:
        print '*** DF ***'
        print 'Split into train and test dataset...'
        portion = 0.8
        trainnum = int(len(df_file_records) * 0.8)
        train_files = df_file_records[:trainnum]  # 这里train_files没有用
        test_files = df_file_records[trainnum:]
    else:
        print '*** NW ***'
        print 'Merge and split into train and test dataset...'
        portion = 0.2
        nw_trainnum = int(len(nw_file_records) * portion)
        train_files = df_file_records + nw_file_records[:nw_trainnum]
        test_files = nw_file_records[nw_trainnum:]

    # 训练数据只保留正负样本
    without_none(train_files)

    # 数据合并并生成doc2vec所需格式
    text_train, text_test, y_train, y_test = get_dataset(train_files, test_files)
    # 设置向量维度和训练次数
    size, epoch_num = 400, 10
    # 对数据进行训练，获得模型
    model_dm, model_dbow = train(text_train, text_test, size, epoch_num)
    # 从模型中抽取文档相应的向量
    train_vecs, test_vecs = get_vectors(model_dm, model_dbow, text_train, text_test)

    # 训练
    lr = classifier(train_vecs, y_train, test_vecs, y_test)

    # 测试
    pred_probas = lr.predict_proba(test_vecs)
    print pred_probas
    y_predict = predict_by_proba(pred_probas, 0.2)

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    # print 'Filter labels:', y_predict1
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # y_predict保存至csv
    if os.path.exists(y_predict_dir) is False:
        os.makedirs(y_predict_dir)
    # 分类器预测的
    y_predict_df = pd.DataFrame(y_predict, columns=['y_predict'])
    y_predict_df.to_csv(y_predict_dir + 'doc2vec_y_predict.csv', index=False)
    # 词典过滤的
    # y_predict1_df = pd.DataFrame(y_predict1, columns=['y_predict'])
    # y_predict1_df.to_csv(y_predict_dir + 'filter_y_predict.csv', index=False)

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    find_sources(test_files, source_dir, ere_dir)
    # test_files = use_annotation_source(test_files)

    # 写入文件
    print 'Write into best files...'
    write_best_files(test_files, predict_dir)





