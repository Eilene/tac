# coding=utf-8

from constants import *
from read_file_info_records import *
from evaluation import *
from write_best import *
from find_source import *
from filter_none_with_stdict import *

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM
from read_embedding_index import *
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences



MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 1000


def get_train_labels(str_labels):
    datanum = len(str_labels)
    labels = [0] * datanum
    for i in range(datanum):
        if str_labels[i] == 'pos':
            labels[i] = 1
    return labels


def get_test_labels(str_labels):
    datanum = len(str_labels)
    labels = [0] * datanum
    for i in range(datanum):
        if str_labels[i] == 'pos':
            labels[i] = 2
        elif str_labels[i] == 'neg':
            labels[i] = 1
        else:
            labels[i] = 0
    return labels


def gen_features(contexts, embeddings_index):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(contexts)
    sequences = tokenizer.texts_to_sequences(contexts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    nb_words = min(MAX_NB_WORDS, len(word_index))  # 20000
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # print(embedding_matrix.shape)

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return data, embedding_matrix, nb_words


def gen_samples(train_files, test_files, embeddings_index):
    # 特征
    # 按文件顺序合并
    file_records = train_files + test_files
    contexts = []
    for file_info in file_records:
        if 'entity' in file_info:
            entity_df = file_info['entity']
            entity_contexts = entity_df['entity_mention_context']
            contexts.extend(entity_contexts.tolist())

        if 'relation' in file_info:
            relation_df = file_info['relation']
            rel_arg1_contexts = relation_df['rel_arg1_context']
            rel_arg2_contexts = relation_df['rel_arg2_context']
            relation_contexts = []
            for i in range(len(rel_arg1_contexts)):
                context = rel_arg1_contexts[i] + ' ' + rel_arg2_contexts[i]
                relation_contexts.append(context)
            contexts.extend(relation_contexts)
        if 'event' in file_info:
            event_df = file_info['event']
            event_contexts = event_df['trigger_context']
            contexts.extend(event_contexts.tolist())

    # 用上下文提取特征
    features, embedding_matrix, nb_words = gen_features(contexts, embeddings_index)

    # 标签
    str_y_train = []
    str_y_test = []
    for file_info in train_files:
        if 'entity' in file_info:
            str_labels = file_info['entity']['label_polarity']
            str_y_train.extend(str_labels)
        if 'relation' in file_info:
            str_labels = file_info['relation']['label_polarity']
            str_y_train.extend(str_labels)
        if 'event' in file_info:
            str_labels = file_info['event']['label_polarity']
            str_y_train.extend(str_labels)
    for file_info in test_files:
        if 'entity' in file_info:
            str_labels = file_info['entity']['label_polarity']
            str_y_test.extend(str_labels)
        if 'relation' in file_info:
            str_labels = file_info['relation']['label_polarity']
            str_y_test.extend(str_labels)
        if 'event' in file_info:
            str_labels = file_info['event']['label_polarity']
            str_y_test.extend(str_labels)
    y_train = get_train_labels(str_y_train)
    y_test = get_test_labels(str_y_test)

    # 特征训练测试分开
    trainnum = len(y_train)
    x_train = features[:trainnum]
    x_test = features[trainnum:]

    return x_train, y_train, x_test, y_test, embedding_matrix, nb_words


def lstm_fit(x_train, y_train, embedding_matrix, nb_words):
    # 训练集进一步划分开发集
    split_at = len(x_train) - len(x_train) // 10  # 这是多少？
    # print split_at
    y_train = np.asarray(y_train)
    (x_train_new, x_dev) = x_train[:split_at], x_train[split_at:]
    (y_train_new, y_dev) = y_train[:split_at], y_train[split_at:]

    # 转换标签格式
    y_train_new = keras.utils.to_categorical(y_train_new, 2)
    y_dev = keras.utils.to_categorical(y_dev, 2)
    # print x_train, y_train

    print('Shape of train data tensor:', x_train_new.shape)
    print('Shape of train label tensor:', y_train_new.shape)

    batch_size = 32
    embedding_layer = Embedding(nb_words + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH)
    print('Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.add(Dense(2, activation='softmax'))
    model.layers[1].trainable = False

    # try using different optimizers and different optimizer configs
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    model.compile(loss=keras.losses.binary_crossentropy, optimizer='Adam', metrics=['accuracy'])
    model.summary()

    print('Train...')
    model.fit(x_train_new, y_train_new,
              batch_size=batch_size,
              epochs=15,
              validation_data=(x_dev, y_dev))
    score, acc = model.evaluate(x_dev, y_dev,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return model


def lstm_predict(model, x_test, y_test):
    probabilities = model.predict(x_test)
    y_predict = []
    # print probabilities
    for i in probabilities:
        if i[1] - i[0] > 0.2:  # 可调阈值，过滤0
            y_predict.append(2)
        elif i[0] - i[1] > 0.2:
            y_predict.append(1)
        else:
            y_predict.append(0)
    accuracy = np.mean(y_predict == y_test)
    print("Prediction Accuracy: %.2f%%" % (accuracy * 100))
    return y_predict


def without_none(file_records):
    for i in range(len(file_records)):
        if 'entity' in file_records[i]:
            file_records[i]['entity'] = file_records[i]['entity'][file_records[i]['entity'].label_polarity != 'none'].reset_index()
        if 'relation' in file_records[i]:
            file_records[i]['relation'] = file_records[i]['relation'][file_records[i]['relation'].label_polarity != 'none'].reset_index()
        if 'event' in file_records[i]:
            file_records[i]['event'] = file_records[i]['event'][file_records[i]['event'].label_polarity != 'none'].reset_index()
    return file_records


def filter_none(file_records):
    pred = []
    for i in range(len(file_records)):
        if 'entity' in file_records[i]:
            # 取上下文
            contexts = file_records[i]['entity']['entity_mention_context']
            # 打分
            scores = context_scoring(contexts)
            # 根据分给一份predict
            p = predict_by_scores(scores)
            pred.extend(p)
        if 'relation' in file_records[i]:
            # 取上下文
            rel_arg1_contexts = file_records[i]['relation']['rel_arg1_context']
            rel_arg2_contexts = file_records[i]['relation']['rel_arg2_context']
            contexts = []
            for j in range(len(rel_arg1_contexts)):
                context = rel_arg1_contexts[j] + ' ' + rel_arg2_contexts[j]
                contexts.append(context)
            # 打分
            scores = context_scoring(contexts)
            # 根据分给一份predict
            p = predict_by_scores(scores)
            pred.extend(p)
        if 'event' in file_records[i]:
            # 取上下文
            contexts = file_records[i]['event']['trigger_context']
            # 打分
            scores = context_scoring(contexts)
            # 根据分给一份predict
            p = predict_by_scores(scores)
            pred.extend(p)
    return pred


def attach_predict_labels(test_files, y_predict):
    count = 0
    for i in range(len(test_files)):
        if 'entity' in test_files[i]:
            # 转字典
            test_files[i]['entity'] = test_files[i]['entity'].to_dict(orient='records')
            # 加上label
            for j in range(len(test_files[i]['entity'])):
                if y_predict[count] == 0:
                    test_files[i]['entity'][j]['predict_polarity'] = 'none'
                elif y_predict[count] == 1:
                    test_files[i]['entity'][j]['predict_polarity'] = 'neg'
                else:
                    test_files[i]['entity'][j]['predict_polarity'] = 'pos'
                count += 1
        if 'relation' in test_files[i]:
            # 转字典
            test_files[i]['relation'] = test_files[i]['relation'].to_dict(orient='records')
            # 加上label
            for j in range(len(test_files[i]['relation'])):
                if y_predict[count] == 0:
                    test_files[i]['relation'][j]['predict_polarity'] = 'none'
                elif y_predict[count] == 1:
                    test_files[i]['relation'][j]['predict_polarity'] = 'neg'
                else:
                    test_files[i]['relation'][j]['predict_polarity'] = 'pos'
                count += 1
        if 'event' in test_files[i]:
            # 转字典
            test_files[i]['event'] = test_files[i]['event'].to_dict(orient='records')
            # 加上label
            for j in range(len(test_files[i]['event'])):
                if y_predict[count] == 0:
                    test_files[i]['event'][j]['predict_polarity'] = 'none'
                elif y_predict[count] == 1:
                    test_files[i]['event'][j]['predict_polarity'] = 'neg'
                else:
                    test_files[i]['event'][j]['predict_polarity'] = 'pos'
                count += 1
    return test_files


if __name__ == '__main__':
    # 读取各文件中间信息
    print 'Read data...'
    file_info_records = read_file_info_records(ere_dir, entity_info_dir, relation_info_dir, event_info_dir, em_args_dir)

    # 按文件划分训练和测试集
    print 'Split into train and test dataset...'
    portion = 0.8
    trainnum = int(len(file_info_records) * 0.8)
    train_files = file_info_records[:trainnum]
    test_files = file_info_records[trainnum:]

    # 训练文件去掉none的样本
    train_files = without_none(train_files)

    # 提取特征及标签
    print "Generate samples..."
    embeddings_index, dim = read_embedding_index(glove_100d_path)
    x_train, y_train, x_test, y_test, embedding_matrix, nb_words = gen_samples(train_files, test_files, embeddings_index)
    print 'Train data number:', len(y_train)
    print 'Test data number:', len(y_test)

    # 训练
    print 'Train...'
    model = lstm_fit(x_train, y_train, embedding_matrix, nb_words)

    # 测试
    print 'Test...'
    y_predict = lstm_predict(model, x_test, y_test)
    # 测试文件根据打分过滤掉none的样本
    y_predict1 = filter_none(test_files)
    y_predict = [y_predict[i] if y_predict1[i] != 0 else y_predict1[i] for i in range(len(y_predict))]

    # 评价
    print 'Evalution: '
    print 'Test labels: ', y_test
    print 'Predict labels: ', y_predict
    evaluation_3classes(y_test, y_predict)  # 3类的测试评价

    # 测试结果写入记录
    test_files = attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    test_files = find_sources(test_files, source_dir, ere_dir)

    # 写入文件
    write_best_files(test_files, predict_dir)

