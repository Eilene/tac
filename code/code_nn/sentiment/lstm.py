# coding=utf-8

import keras

from keras.layers import Dense, Dropout, Activation, Embedding
from keras.layers import LSTM
from keras.models import Sequential

from utils.evaluation import evaluation_3classes
from utils.filter_none_with_stdict import *
from utils.find_source import find_sources
from utils.read_embedding_index import *
from utils.read_file_info_records import *
# from utils.resampling import *
from utils.write_best import write_best_files
from utils.constants import *
from utils.file_records_other_modification import without_none, to_dict
from utils.get_labels import get_merged_labels
from utils.predict_by_proba import *
from utils.attach_predict_labels import *

from features.lstm_features import *


def lstm_fit(x_train, y_train, embedding_matrix, nb_words, classnum):
    # 训练集进一步划分开发集
    split_at = len(x_train) - len(x_train) // 10  # 这是多少？
    # print split_at
    y_train = np.asarray(y_train)
    (x_train_new, x_dev) = x_train[:split_at], x_train[split_at:]
    (y_train_new, y_dev) = y_train[:split_at], y_train[split_at:]

    # 转换标签格式
    y_train_new = keras.utils.to_categorical(y_train_new, classnum)
    y_dev = keras.utils.to_categorical(y_dev, classnum)
    # print x_train, y_train

    print('Shape of train data tensor:', x_train_new.shape)
    print('Shape of train label tensor:', y_train_new.shape)

    batch_size = 32
    embedding_layer = Embedding(nb_words + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    print('Build model...')
    model = Sequential()
    model.add(embedding_layer)
    model.add(Dropout(0.2))
    model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    model.add(Dense(classnum, activation='sigmoid'))
    # model.layers[1].trainable = False

    # try using different optimizers and different optimizer configs
    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])
    model.compile(loss=keras.losses.binary_crossentropy, optimizer='Adam', metrics=['accuracy'])
    model.summary()

    print('Train...')
    model.fit(x_train_new, y_train_new,
              batch_size=batch_size,
              epochs=1,
              validation_data=(x_dev, y_dev))
    score, acc = model.evaluate(x_dev, y_dev,
                                batch_size=batch_size)
    print('Test score:', score)
    print('Test accuracy:', acc)

    return model


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
        train_files = df_file_records[:trainnum]
        test_files = df_file_records[trainnum:]
    else:
        print '*** NW ***'
        print 'Merge and split into train and test dataset...'
        portion = 0.2
        nw_trainnum = int(len(nw_file_records) * portion)
        train_files = df_file_records + nw_file_records[:nw_trainnum]
        test_files = nw_file_records[nw_trainnum:]

    # 提取特征及标签
    print "Generate samples..."
    embeddings_index, dim = read_embedding_index(glove_100d_path)
    without_none(train_files)  # 训练文件去掉none的样本
    file_records = train_files + test_files
    x_all, embedding_matrix, nb_words = gen_lstm_features(file_records, embeddings_index)
    y_train = get_merged_labels(train_files)  # 只有1,2两类
    y_train = [y-1 for y in y_train]  # 改为0,1
    y_test = get_merged_labels(test_files)  # 0,1,2三类
    # 特征分割训练测试集
    trainlen = len(y_train)
    x_train = x_all[:trainlen]
    x_test = x_all[trainlen:]
    print 'Train data number:', len(y_train)
    print 'Test data number:', len(y_test)

    # 训练
    print 'Train...'
    model = lstm_fit(x_train, y_train, embedding_matrix, nb_words, 2)

    # 测试
    print 'Test...'
    probabilities = model.predict(x_test)
    y_predict = predict_by_proba(probabilities, 0.2)
    # 测试文件根据打分过滤掉none的样本
    # y_predict1 = filter_none(test_files)
    # y_predict = [y_predict[i] if y_predict1[i] != 0 else y_predict1[i] for i in range(len(y_predict))]

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
    y_predict_df.to_csv(y_predict_dir+'lstm_y_predict.csv', index=False)

    # 测试结果写入记录
    to_dict(test_files)
    attach_predict_labels(test_files, y_predict)

    # 寻找源
    print 'Find sources... '
    find_sources(test_files, source_dir, ere_dir)

    # 写入文件
    write_best_files(test_files, predict_dir)

