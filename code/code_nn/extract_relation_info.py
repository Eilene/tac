# coding=utf-8

import os
import xml.dom.minidom
import pandas as pd
# import numpy as np
from split_sentences import split_sentences, find_context


# 将上下文字典中的文本按序拼接成字符串
def context_dict_to_string(context_dict, above, below):
    keys = [k for k in range(-above, below + 1)]
    context = ""
    for key in keys:
        if (context_dict is not None) and ('text' in context_dict[key]):
            # 如果出现None可能是源，一般不出现
            context += context_dict[key]['text']
            context = context.replace('\n', '')  # 去掉换行
    if context == "":
        context = "None"
    return context


# rel_arg的所属entity信息
def rel_arg_entity_info(entity_list, rel_arg_id, rel_arg_mention_id, rel_arg_text, sentences):
    # 根据id找所在entity和mention
    for k in range((len(entity_list))):
        entity_id = entity_list[k].getAttribute('id')
        if rel_arg_id == entity_id:
            rel_arg_entity_type = entity_list[k].getAttribute('type')
            rel_arg_entity_specificity = entity_list[k].getAttribute('specificity')
            entity_mention_list = entity_list[k].getElementsByTagName('entity_mention')
            for m in range((len(entity_mention_list))):
                entity_mention_id = entity_mention_list[m].getAttribute('id')
                if rel_arg_mention_id == entity_mention_id:
                    rel_arg_mention_noun_type = entity_mention_list[m].getAttribute('noun_type')
                    rel_arg_mention_offset = int(entity_mention_list[m].getAttribute('offset'))
                    rel_arg_mention_length = int(entity_mention_list[m].getAttribute('length'))
                    # 上下文信息
                    above = 3
                    below = 3
                    rel_arg_context_dict = find_context(rel_arg_mention_offset, sentences,
                                                         rel_arg_text, above, below)
                    # 拼成一个字符串
                    rel_arg_context = context_dict_to_string(rel_arg_context_dict, above, below)
                    # if rel_arg_context == "None":
                    #     print rel_arg_text, rel_arg_id, part_name, rel_arg_context
                    return rel_arg_entity_type, rel_arg_entity_specificity, rel_arg_mention_noun_type, \
                           rel_arg_mention_offset, rel_arg_mention_length, rel_arg_context


# rel_arg的所属filler信息
def rel_arg_filler_info(filler_list, rel_arg_id, rel_arg_text, sentences):
    # 根据id找所在filler和mention
    for k in range((len(filler_list))):
        filler_id = filler_list[k].getAttribute('id')
        if rel_arg_id == filler_id:
            rel_arg_filler_type = filler_list[k].getAttribute('type')
            rel_arg_mention_offset = int(filler_list[k].getAttribute('offset'))
            rel_arg_mention_length = int(filler_list[k].getAttribute('length'))
            # 上下文信息
            above = 3
            below = 3
            rel_arg_context_dict = find_context(rel_arg_mention_offset, sentences,
                                                 rel_arg_text, above, below)
            # 拼成一个字符串
            rel_arg_context = context_dict_to_string(rel_arg_context_dict, above, below)
            # if rel_arg_context == "None":
            #     print rel_arg_text, rel_arg_id, part_name, rel_arg_context
            return rel_arg_filler_type, rel_arg_mention_offset, rel_arg_mention_length, rel_arg_context


def extract_relation_each_file(source_filepath, ere_filepath, annotation_filepath, part_name):
    relation_records_each_file = []

    source_fp = open(source_filepath)
    all_source_text = source_fp.read().decode("utf-8")  # 注意编码
    sentences = split_sentences(all_source_text)  # 分句

    ere_file = xml.dom.minidom.parse(ere_filepath)
    ere_root = ere_file.documentElement
    relation_list = ere_root.getElementsByTagName('relation')
    entity_list = ere_root.getElementsByTagName('entity')
    filler_list = ere_root.getElementsByTagName('filler')

    annotation_file = xml.dom.minidom.parse(annotation_filepath)
    annotation_root = annotation_file.documentElement
    annotation_sentiment_list = annotation_root.getElementsByTagName('sentiment_annotations')
    annotation_sentiment_list = annotation_sentiment_list[0]
    annotation_relation_list = annotation_sentiment_list.getElementsByTagName('relation')  # 实际上是relation_mention

    for i in range(len(relation_list)):
        # relation信息
        relation_id = relation_list[i].getAttribute('id')
        relation_type = relation_list[i].getAttribute('type')
        relation_subtype = relation_list[i].getAttribute('subtype')
        relation_mention_list = relation_list[i].getElementsByTagName('relation_mention')

        for j in range(len(relation_mention_list)):
            # relation mention信息
            relation_mention_id = relation_mention_list[j].getAttribute('id')
            relation_mention_realis = relation_mention_list[j].getAttribute('realis')

            # polarity
            for k in range((len(annotation_relation_list))):
                annotation_relation_id = annotation_relation_list[k].getAttribute('ere_id')
                if annotation_relation_id == relation_mention_id:
                    st_em = annotation_relation_list[k].getElementsByTagName('sentiment')
                    if len(st_em) == 0:
                        print part_name, annotation_relation_id, relation_mention_id
                    label_polarity = st_em[0].getAttribute('polarity')
            if label_polarity == 'none':
                break  # 如果为none则丢弃该样本

            # rel_arg是entity
            # 基本信息
            rel_arg1 = relation_mention_list[j].getElementsByTagName('rel_arg1')
            rel_arg1 = rel_arg1[0]
            rel_arg1_id = rel_arg1.getAttribute('entity_id')
            rel_arg1_mention_id = rel_arg1.getAttribute('entity_mention_id')
            rel_arg1_role = rel_arg1.getAttribute('role')
            rel_arg1_text = rel_arg1.firstChild.data
            # 所属entity及entity mention信息
            rel_arg1_entity_type, rel_arg1_entity_specificity, rel_arg1_mention_noun_type, rel_arg1_mention_offset, \
            rel_arg1_mention_length, rel_arg1_context = rel_arg_entity_info(entity_list, rel_arg1_id,
                                                                            rel_arg1_mention_id, rel_arg1_text,
                                                                            sentences)

            # rel_arg，同上
            rel_arg2 = relation_mention_list[j].getElementsByTagName('rel_arg2')
            rel_arg2 = rel_arg2[0]
            rel_arg2_role = rel_arg2.getAttribute('role')
            rel_arg2_text = rel_arg2.firstChild.data
            rel_arg2_id = rel_arg2.getAttribute('entity_id')
            if rel_arg2_id != "":
                rel_arg2_mention_id = rel_arg2.getAttribute('entity_mention_id')
                # 所属entity及entity mention信息
                rel_arg2_entity_type, rel_arg2_entity_specificity, rel_arg2_mention_noun_type, rel_arg2_mention_offset, \
                rel_arg2_mention_length, rel_arg2_context = rel_arg_entity_info(entity_list, rel_arg2_id,
                                                                            rel_arg2_mention_id, rel_arg2_text,
                                                                            sentences)
                rel_arg2_is_filler = 0
            else:  # rel_arg2有的不是entity是filler，先简单处理
                rel_arg2_is_filler = 1
                rel_arg2_id = rel_arg2.getAttribute('filler_id')
                if rel_arg2_id == "":
                    print part_name, relation_mention_id
                rel_arg2_entity_type, rel_arg2_mention_offset, rel_arg2_mention_length, rel_arg2_context = \
                    rel_arg_filler_info(filler_list, rel_arg2_id, rel_arg2_text, sentences)
                rel_arg2_mention_id = "None"
                rel_arg2_entity_specificity = "None"
                rel_arg2_mention_noun_type = "None"

            # trigger
            trigger = relation_list[i].getElementsByTagName('trigger')
            if len(trigger) == 0:
                trigger_offset = "None"
                trigger_length = "None"
                trigger_text = "None"
                trigger_context = "None"
            else:
                trigger = trigger[0]
                trigger_offset = int(trigger.getAttribute('offset'))
                trigger_length = int(trigger.getAttribute('length'))
                trigger_text = trigger.firstChild.data
                # 上下文信息
                above = 0
                below = 0  # 可调，考虑trigger中at等词较多，似乎不宜太长上下文，这里先只提取当前句子
                trigger_context_dict = find_context(trigger_offset, sentences, trigger_text, above, below)
                # 拼成一个字符串
                trigger_context = context_dict_to_string(trigger_context_dict, above, below)

            relation_record = {'file': part_name,
                      'relation_id': relation_id, 'relation_type': relation_type, 'relation_subtype': relation_subtype,
                      'relation_mention_id': relation_mention_id, 'relation_mention_realis': relation_mention_realis,
                      'rel_arg1_id': rel_arg1_id, 'rel_arg1_mention_id': rel_arg1_mention_id,
                      'rel_arg1_role': rel_arg1_role, 'rel_arg1_text': rel_arg1_text,
                      'rel_arg1_entity_type': rel_arg1_entity_type,
                      'rel_arg1_entity_specificity': rel_arg1_entity_specificity,
                      'rel_arg1_mention_noun_type': rel_arg1_mention_noun_type,
                      'rel_arg1_mention_offset': rel_arg1_mention_offset,
                      'rel_arg1_mention_length': rel_arg1_mention_length, 'rel_arg1_context': rel_arg1_context,
                      'rel_arg2_id': rel_arg2_id, 'rel_arg2_mention_id': rel_arg2_mention_id, 
                      'rel_arg2_role': rel_arg2_role, 'rel_arg2_text': rel_arg2_text, 
                      'rel_arg2_entity_type': rel_arg2_entity_type,
                      'rel_arg2_entity_specificity': rel_arg2_entity_specificity,
                      'rel_arg2_mention_noun_type': rel_arg2_mention_noun_type,
                      'rel_arg2_mention_offset': rel_arg2_mention_offset,
                      'rel_arg2_mention_length': rel_arg2_mention_length, 'rel_arg2_context': rel_arg2_context,
                      'rel_arg2_is_filler': rel_arg2_is_filler,
                      'trigger_offset': trigger_offset, 'trigger_length': trigger_length, 'trigger_text': trigger_text,
                      'trigger_context': trigger_context,
                      'label_polarity': label_polarity
                      }

            relation_records_each_file.append(relation_record)

    return relation_records_each_file


def extract_relation(source_dir, ere_dir, annotation_dir):
    relation_records = []

    ere_suffix = ".rich.ere.xml"
    ere_suffix_length = len(ere_suffix)
    for parent, dirnames, ere_filenames in os.walk(ere_dir):
        for ere_filename in ere_filenames:  # 输出文件信息
            part_name = ere_filename[:-ere_suffix_length]
            source_filepath = source_dir + part_name + ".cmp.txt"
            if os.path.exists(source_filepath) is False: # 不存在，则可能是新闻，xml，先跳过，后续考虑处理
                # source_filepath = source_dir + part_name + ".xml"
                continue
            ere_filepath = ere_dir + ere_filename
            annotation_filepath = annotation_dir + part_name + ".best.xml"
            records = extract_relation_each_file(source_filepath, ere_filepath, annotation_filepath, part_name)
            relation_records.extend(records)

    return relation_records


def write_to_csv(records, filename):
    df = pd.DataFrame(records)  # 好像会重复多次？看看
    print len(records)
    print df.shape
    df.to_csv(filename, encoding="utf-8", index=None)


if __name__ == '__main__':

    data_dirpath = "../data/2016E27_V2/data/"
    source_dirpath = data_dirpath + "source/"
    ere_dirpath = data_dirpath + "ere/"
    annotation_dirpath = data_dirpath + "annotation/"

    relation_pos_neg_info = extract_relation(source_dirpath, ere_dirpath, annotation_dirpath)
    write_to_csv(relation_pos_neg_info, 'relation_pos_neg_info.csv')


# 后续工作：
# arg是filler的情况:要不要分开
# source是xml的情况：暂时未提取这部分数据，后续是否分开提取
# 抽取event