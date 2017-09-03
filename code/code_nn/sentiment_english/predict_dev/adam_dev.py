# coding=utf-8

from src.sentiment_english.utils.constants import *
from src.sentiment_english.utils.read_file_info_records import *
from src.sentiment_english.utils.get_labels import get_labels
from src.sentiment_english.utils.evaluation import evaluation_3classes
from src.sentiment_english.utils.file_records_other_modification import to_dict
from src.sentiment_english.utils.attach_predict_labels import *
from src.sentiment_english.utils.find_source import find_sources
from src.sentiment_english.utils.write_best import write_best_files
from src.sentiment_english.utils.resampling import up_resampling_3classes


def get_adam_dataform(file_records):
    contexts = []
    texts = []
    labels = []

    for file_record in file_records:
        if 'entity' in file_record:
            entity_contexts =  file_record['entity']['entity_mention_sentence'].tolist()  # 先用1句试试
            entity_texts = file_record['entity']['entity_mention_text'].tolist()
            # 根据offset找到，位置，相应长度的换成$T$
            sentence_offsets = file_record['entity']['entity_mention_sentence_offset'].tolist()
            target_offsets = file_record['entity']['entity_mention_offset'].tolist()
            lengths = file_record['entity']['entity_mention_length'].tolist()
            for i in range(len(entity_contexts)):
                # if str(entity_contexts[i]) == 'nan':   # 先不考虑，如果测试集有这种的话到时直接赋个label？不然没法训练
                sen_offset = int(sentence_offsets[i])
                offset = int(target_offsets[i])
                length = int(lengths[i])
                # # 怎么调都有错的。。。
                # if entity_contexts[i][offset-sen_offset+1:offset-sen_offset+length+1] != entity_texts[i]:
                #     print '|'+entity_contexts[i][offset-sen_offset+1:offset-sen_offset+length+1]+'|'+entity_texts[i]+'|'
                #     print len(entity_contexts[i][offset-sen_offset+1:offset-sen_offset+length+1]), len(entity_texts[i])
                # entity_contexts[i] = entity_contexts[i][:offset-sen_offset+1] + '$T$' + \
                #                      entity_contexts[i][offset-sen_offset+length+1:]
                # 用匹配试一下
                start = offset-sen_offset-2
                if start < 0:
                    start = 0
                t_offset = entity_contexts[i][start:offset-sen_offset+length+2].find(entity_texts[i])
                # 一定范围内
                # print t_offset+start, offset-sen_offset
                entity_contexts[i] = entity_contexts[i][:t_offset+start] + '$T$' + \
                                     entity_contexts[i][t_offset+start+length:]
                entity_contexts[i] = entity_contexts[i].replace("\n", "")
                entity_contexts[i] = entity_contexts[i].replace("\r", "")
            contexts.extend(entity_contexts)
            texts.extend(entity_texts)

            str_labels = file_record['entity']['label_polarity']
            entity_labels = get_labels(str_labels)
            labels.extend(entity_labels)

        if 'relation' in file_record:
            del file_record['relation']
        if 'event' in file_record:
            del file_record['event']

    labels = [y-1 for y in labels]  # 变0,1，-1，即-1为none，以此类推

    return contexts, texts, labels


def merge_lines(contexts, texts, labels):
    data_list = []
    for i in range(len(labels)):
        data_list.append(contexts[i]+'\n')
        data_list.append(texts[i]+'\n')
        data_list.append(str(labels[i])+'\n')
    return data_list


if __name__ == '__main__':
    mode = True  # True:DF,false:NW
    in_out = False  # True:in, false:out
    adam_dev_dir = st_output_prefix+'adam_dev/'

    print 'Read data...'
    df_file_records, nw_file_records = \
        read_file_info_records(train_ere_dir, train_entity_info_dir, train_relation_info_dir, train_event_info_dir,
                               train_em_args_dir)

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

    # 生成数据格式
    train_contexts, train_texts, train_labels = get_adam_dataform(train_files)
    test_contexts, test_texts, test_labels = get_adam_dataform(test_files)

    if in_out is True:
        # 重采样
        print len(train_labels)
        train_samples = []
        for i in range(len(train_labels)):
            train_samples.append([train_contexts[i], train_texts[i]])
        train_labels = [y+1 for y in train_labels]
        train_samples, train_labels = up_resampling_3classes(train_samples, train_labels)
        train_labels = [y-1 for y in train_labels]  # 变0,1，-1，即-1为none，以此类推
        train_contexts = []
        train_texts = []
        for i in range(len(train_labels)):
            train_contexts.append(train_samples[i][0])
            train_texts.append(train_samples[i][1])
        print len(train_labels)

        # 合并数据
        train_data_list = merge_lines(train_contexts, train_texts, train_labels)
        test_data_list = merge_lines(test_contexts, test_texts, test_labels)

        # 写入文件
        train_file = file(adam_dev_dir+"best_train.xml.seg", "w+")
        train_file.writelines(train_data_list)
        train_file.close()

        test_file = file(adam_dev_dir+"best_test.xml.seg", "w+")
        test_file.writelines(test_data_list)
        test_file.close()

    else:
        # 读取预测结果
        filename = adam_dev_dir+ 'y_predict.csv'
        y_pred_df = pd.read_csv(filename)
        y_predict = y_pred_df.values.T[0].tolist()
        y_predict = [-1 if y == 2 else y for y in y_predict]
        y_predict = [int(y + 1) for y in y_predict]

        # 评价
        y_test = [y+1 for y in test_labels]
        print 'Evalution: '
        print 'Test labels: ', y_test
        # print 'Filter labels:', y_predict1
        print 'Predict labels: ', y_predict
        evaluation_3classes(y_test, y_predict)  # 3类的测试评价

        # 测试结果写入记录
        to_dict(test_files)
        attach_predict_labels(test_files, y_predict)

        # 寻找源
        print 'Find sources... '
        find_sources(test_files, train_source_dir, train_ere_dir)
        # use_annotation_source(test_files)

        # 写入文件
        print 'Write into best files...'
        write_best_files(test_files, dev_predict_dir)

    # 最终输出是-1->2，其余不变
