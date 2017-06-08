# coding=utf-8

import xml.dom.minidom
import os
import os.path
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras

from gen_samples import *


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


# 训练CNN并测试
def cnn(train_features, train_labels, results):
    # 转化成单通道输入方式
    train_labels = np.array(train_labels)
    # print np.array(train_features).shape, train_labels.shape
    data = np.empty((train_labels.shape[0], 51, 100, 1), dtype='float32')  # 向量行数50+1
    cnt = 0
    for i in range(len(train_features)):
        data[cnt, :, :, 0] = train_features[i]
        cnt += 1
    train_features = data
    input_shape = train_features.shape[1:]  # 与samples个数无关
    train_labels = keras.utils.to_categorical(train_labels, 2)

    # 开始建立CNN模型
    batch_size = 128
    epochs = 3
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    # sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss=keras.losses.binary_crossentropy,optimizer='Adam',metrics=['accuracy'])
    model.summary()
    # 训练
    model.fit(train_features, train_labels, batch_size=batch_size, epochs=epochs,verbose=1)

    # 预测并填入结果
    for i in range(len(results)):
        data = np.empty((len(results[i]['features']), 51, 100, 1), dtype='float32')  # 向量行数50+1
        cnt = 0
        for j in range(len(results[i]['features'])):
            data[cnt, :, :, 0] = results[i]['features'][j]
            cnt += 1
        results[i]['features'] = data
        probabilities = model.predict(results[i]['features'])
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
    final_results = cnn(train_x, train_y, output_results)  # 修改了output_results
    write_best_files(final_results)




