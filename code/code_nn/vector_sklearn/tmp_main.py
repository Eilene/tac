# coding=utf-8

import os
import os.path
import xml.dom.minidom

import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from src.vector_sklearn.gen_samples import *


# 合并各文件，划分成训练和测试数据
def gen_whole_data():
    # 从GloVe文件中解析出每个词和它所对应的词向量，并用字典的方式存储
    embeddings_index = {}
    dim = 100
    embedding_vectors_fp = open('../../data/glove.6B/glove.6B.100d.txt')
    for line in embedding_vectors_fp:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    embedding_vectors_fp.close()
    # print('Found %s word vectors.' % len(embeddings_index))

    # 遍历文件夹，生成x,y，分成train和test
    best_data_dir = '../../data/2016E27_V2/data/'
    source_rootdir = best_data_dir + "source/"
    ere_dir = best_data_dir + "ere/"
    annotation_dir = best_data_dir + "annotation/"
    file_num = 0  # 先统计文件数，确定分割比例
    for parent, dirnames, source_filenames in os.walk(source_rootdir):
        for source_filename in source_filenames:  # 输出文件信息
            if source_filename[-3:] == "xml":  # 先不处理新闻数据
                continue
            file_num += 1
    train_file_num = int(file_num * 0.8)
    # print file_num, train_file_num
    file_count = 0
    train_features = []
    train_labels = []
    results = []  # 测试数据的一切信息，即要输出到各个best文件的信息
    for parent, dirnames, source_filenames in os.walk(source_rootdir):
        for source_filename in source_filenames:  # 输出文件信息
            if source_filename[-3:] == "xml":  # 先不处理新闻数据
                continue
            part_name = source_filename[:-8]
            # if len(features) == 0:
            #     print source_filename
            file_count += 1
            if file_count <= train_file_num:
                raw_records = gen_train_records(parent + source_filename,
                                                ere_dir + part_name + ".rich_ere.xml",
                                                annotation_dir + part_name + ".best.xml")
                features, labels = gen_train_sample(raw_records, embeddings_index, dim)
                train_features.extend(features)
                train_labels.extend(labels)
            else:
                raw_records = gen_test_records(parent + source_filename, ere_dir + part_name + ".rich_ere.xml")
                features = gen_test_sample(raw_records, embeddings_index, dim)
                best_filename = '../../data/output/E27_V2_Results/' + part_name + ".best.xml"
                rs = {'filename': best_filename, 'records': raw_records, 'features': features}
                results.append(rs)
    # print "result:"
    # print len(train_labels), len(test_labels)
    # print train_features
    # print train_labels
    # print test_features
    # print test_labels
    return train_features, train_labels, results


# lr训练与测试
def lr(train_features, train_labels, results):
    # 训练
    lr_clf = LogisticRegression()
    lr_clf.fit(train_features, train_labels)

    # 预测并填入结果
    for i in range(len(results)):
        probabilities = lr_clf.predict_proba(results[i]['features'])
        print probabilities  # 依然全部都是负类
        for j in range(len(probabilities)):
            if probabilities[j][0] < probabilities[j][1]:
                results[i]['records'][j]['sentiment']['polarity'] = 'pos'
            else:
                results[i]['records'][j]['sentiment']['polarity'] = 'neg'

    return results


# svm训练与测试
def svm(train_features, train_labels, results):
    # 训练
    svm_clf = svm.SVC()
    svm_clf.fit(train_features, train_labels)

    # 预测并填入结果
    for i in range(len(results)):
        probabilities = svm_clf.predict_proba(results[i]['features'])
        print probabilities  # 的确全部都是负类
        for j in range(len(probabilities)):
            if probabilities[j][0] < probabilities[j][1]:
                results[i]['records'][j]['sentiment']['polarity'] = 'pos'
            else:
                results[i]['records'][j]['sentiment']['polarity'] = 'neg'

    return results


# 输出best.xml
def write_best(filename, records, no):
    doc = xml.dom.minidom.Document()

    best = doc.createElement('belief_sentiment_doc')
    best.setAttribute('id', "tree-56acee9a00000000000000"+str(no))
    doc.appendChild(best)
    st = doc.createElement('sentiment_annotations')
    best.appendChild(st)
    entities = doc.createElement('entities')
    st.appendChild(entities)

    for i in range(len(records)):
        entity = doc.createElement('entity')
        entity.setAttribute('ere_id', records[i]['target']['ere_id'])
        entity.setAttribute('offset', str(records[i]['target']['offset']))
        entity.setAttribute('length', str(records[i]['target']['length']))
        entities.appendChild(entity)

        text = doc.createElement('text')
        text_text = doc.createTextNode(records[i]['target']['text'])
        text.appendChild(text_text)
        entity.appendChild(text)

        sentiments = doc.createElement('sentiments')
        entity.appendChild(sentiments)

        sentiment = doc.createElement('sentiment')
        sentiment.setAttribute('polarity', records[i]['sentiment']['polarity'])
        sentiment.setAttribute('sarcasm', records[i]['sentiment']['sarcasm'])
        sentiments.appendChild(sentiment)

        if records[i]['sentiment']['polarity'] != 'none':
            source = doc.createElement('source')
            source.setAttribute('ere_id', str(records[i]['source']['ere_id']))
            source.setAttribute('offset', str(records[i]['source']['offset']))
            source.setAttribute('length', str(records[i]['source']['length']))
            sentiment.appendChild(source)

    f = open(filename, 'w')
    f.write(doc.toprettyxml(indent='\t', encoding='utf-8'))
    f.close()


def write_best_files(results):
    sentiment = {'polarity': 'pos', 'sarcasm': 'no'}
    for i in range(len(results)):
        for j in range(len(results[i]['records'])):
            results[i]['records'][j]['sentiment'] = sentiment
        write_best(results[i]['filename'], results[i]['records'], i)


if __name__ == '__main__':
    train_x, train_y, output_results = gen_whole_data()
    final_results = lr(train_x, train_y, output_results)  # 修改了output_results
    write_best_files(final_results)




