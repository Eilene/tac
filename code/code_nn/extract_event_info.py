# coding=utf-8

import os
import xml.dom.minidom
import pandas as pd
# import numpy as np
from split_sentences import split_sentences, find_context
from constants import *


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


# em_arg的所属entity信息
def em_arg_entity_info(entity_list, em_arg_id, em_arg_mention_id, em_arg_text, sentences):
    # 根据id找所在entity和mention
    for k in range((len(entity_list))):
        entity_id = entity_list[k].getAttribute('id')
        if em_arg_id == entity_id:
            em_arg_entity_type = entity_list[k].getAttribute('type')
            em_arg_entity_specificity = entity_list[k].getAttribute('specificity')
            entity_mention_list = entity_list[k].getElementsByTagName('entity_mention')
            for m in range((len(entity_mention_list))):
                entity_mention_id = entity_mention_list[m].getAttribute('id')
                if em_arg_mention_id == entity_mention_id:
                    em_arg_mention_noun_type = entity_mention_list[m].getAttribute('noun_type')
                    em_arg_mention_offset = int(entity_mention_list[m].getAttribute('offset'))
                    em_arg_mention_length = int(entity_mention_list[m].getAttribute('length'))
                    # 上下文信息
                    above = 3
                    below = 3
                    em_arg_context_dict = find_context(em_arg_mention_offset, sentences,
                                                         em_arg_text, above, below)
                    # 拼成一个字符串
                    em_arg_context = context_dict_to_string(em_arg_context_dict, above, below)
                    # if em_arg_context == "None":
                    #     print em_arg_text, em_arg_id, part_name, em_arg_context
                    return em_arg_entity_type, em_arg_entity_specificity, em_arg_mention_noun_type, \
                           em_arg_mention_offset, em_arg_mention_length, em_arg_context


# em_arg的所属filler信息
def em_arg_filler_info(filler_list, em_arg_id, em_arg_text, sentences):
    # 根据id找所在filler和mention
    for k in range((len(filler_list))):
        filler_id = filler_list[k].getAttribute('id')
        if em_arg_id == filler_id:
            em_arg_filler_type = filler_list[k].getAttribute('type')
            em_arg_mention_offset = int(filler_list[k].getAttribute('offset'))
            em_arg_mention_length = int(filler_list[k].getAttribute('length'))
            # 上下文信息
            above = 3
            below = 3
            em_arg_context_dict = find_context(em_arg_mention_offset, sentences,
                                                 em_arg_text, above, below)
            # 拼成一个字符串
            em_arg_context = context_dict_to_string(em_arg_context_dict, above, below)
            # if em_arg_context == "None":
            #     print em_arg_text, em_arg_id, part_name, em_arg_context
            return em_arg_filler_type, em_arg_mention_offset, em_arg_mention_length, em_arg_context


def extract_event_each_file(source_filepath, ere_filepath, annotation_filepath, part_name):
    event_records_each_file = []
    em_args_each_file = []

    source_fp = open(source_filepath)
    all_source_text = source_fp.read().decode("utf-8")  # 注意编码
    sentences = split_sentences(all_source_text)  # 分句

    ere_file = xml.dom.minidom.parse(ere_filepath)
    ere_root = ere_file.documentElement
    hopper_list = ere_root.getElementsByTagName('hopper')
    entity_list = ere_root.getElementsByTagName('entity')
    filler_list = ere_root.getElementsByTagName('filler')

    annotation_file = xml.dom.minidom.parse(annotation_filepath)
    annotation_root = annotation_file.documentElement
    annotation_sentiment_list = annotation_root.getElementsByTagName('sentiment_annotations')
    annotation_sentiment_list = annotation_sentiment_list[0]
    annotation_event_list = annotation_sentiment_list.getElementsByTagName('event')

    for i in range(len(hopper_list)):
        # hopper信息
        hopper_id = hopper_list[i].getAttribute('id')
        event_mention_list = hopper_list[i].getElementsByTagName('event_mention')

        for j in range(len(event_mention_list)):
            # event信息
            event_mention_id = event_mention_list[j].getAttribute('id')
            event_mention_type = event_mention_list[j].getAttribute('type')
            event_mention_subtype = event_mention_list[j].getAttribute('subtype')
            event_mention_realis = event_mention_list[j].getAttribute('realis')
            event_mention_ways = event_mention_list[j].getAttribute('ways')

            # polarity
            for k in range((len(annotation_event_list))):
                annotation_event_id = annotation_event_list[k].getAttribute('ere_id')
                if annotation_event_id == event_mention_id:
                    st_em = annotation_event_list[k].getElementsByTagName('sentiment')
                    # if len(st_em) == 0:
                    #     logger.info("错误：无情感标签。" + " " + part_name + " " + annotation_event_id +
                    #                 " " + event_mention_id)
                    label_polarity = st_em[0].getAttribute('polarity')
            if label_polarity == 'none':
                break  # 如果为none则丢弃该样本

            # trigger
            trigger = event_mention_list[j].getElementsByTagName('trigger')
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

            # em_arg
            em_args = event_mention_list[j].getElementsByTagName('em_arg')
            em_arg_num = len(em_args)
            # print em_arg_num  # 一般不超过4个
            for em_arg in em_args:
                em_arg_role = em_arg.getAttribute('role')
                em_arg_text = em_arg.firstChild.data
                em_arg_id = em_arg.getAttribute('entity_id')
                if em_arg_id != "":  # 是entity
                    em_arg_mention_id = em_arg.getAttribute('entity_mention_id')
                    # 所属entity及entity mention信息
                    em_arg_entity_type, em_arg_entity_specificity, em_arg_mention_noun_type, em_arg_mention_offset, \
                    em_arg_mention_length, em_arg_context = em_arg_entity_info(entity_list, em_arg_id,
                                                                                    em_arg_mention_id, em_arg_text,
                                                                                    sentences)
                    em_arg_is_filler = 0  # 否
                else:
                    em_arg_id = em_arg.getAttribute('filler_id')
                    if em_arg_id == "":
                        logger.info("错误：参数不是entity或filler。" + " " + part_name + " " + event_mention_id)
                    em_arg_entity_type, em_arg_mention_offset, em_arg_mention_length, em_arg_context = \
                        em_arg_filler_info(filler_list, em_arg_id, em_arg_text, sentences)
                    em_arg_mention_id = "None"
                    em_arg_entity_specificity = "None "
                    em_arg_mention_noun_type = "None"
                    em_arg_is_filler = 1
                em_arg_record = {
                    'file': part_name, 'hopper_id': hopper_id, 'event_mention_id': event_mention_id,
                    'em_arg_id': em_arg_id, 'em_arg_mention_id': em_arg_mention_id,
                    'em_arg_role': em_arg_role, 'em_arg_text': em_arg_text,
                    'em_arg_entity_type': em_arg_entity_type,
                    'em_arg_entity_specificity': em_arg_entity_specificity,
                    'em_arg_mention_noun_type': em_arg_mention_noun_type,
                    'em_arg_mention_offset': em_arg_mention_offset,
                    'em_arg_mention_length': em_arg_mention_length, 'em_arg_context': em_arg_context,
                    'em_arg_is_filler': em_arg_is_filler
                }
                em_args_each_file.append(em_arg_record)

            relation_record = {
                'file': part_name, 'hopper_id': hopper_id, 'event_mention_id': event_mention_id,
                'event_mention_type': event_mention_type, 'event_mention_subtype': event_mention_subtype,
                'event_mention_realis': event_mention_realis, 'event_mention_ways': event_mention_ways,
                'trigger_offset': trigger_offset, 'trigger_length': trigger_length,
                'trigger_text': trigger_text, 'trigger_context': trigger_context,
                'em_arg_num': em_arg_num,
                'label_polarity': label_polarity
            }

            event_records_each_file.append(relation_record)

    return event_records_each_file, em_args_each_file


def extract_event(source_dir, ere_dir, annotation_dir):
    event_records = []
    event_em_arg_records = []

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
            records, em_records = extract_event_each_file(source_filepath, ere_filepath, annotation_filepath, part_name)
            event_records.extend(records)
            event_em_arg_records.extend(em_records)

    return event_records, event_em_arg_records


def write_to_csv(records, filename):
    df = pd.DataFrame(records)  # 好像会重复多次？看看
    logger.debug('记录条数：%d', len(records))
    logger.debug('记录维数：(%d, %d)', df.shape[0], df.shape[1])
    df.to_csv(filename, encoding="utf-8", index=None)


if __name__ == '__main__':
    event_pos_neg_info, event_pos_neg_info_em_args = extract_event(source_dirpath, ere_dirpath, annotation_dirpath)
    write_to_csv(event_pos_neg_info, 'event_pos_neg_info.csv')
    write_to_csv(event_pos_neg_info_em_args, 'event_pos_neg_info_em_args.csv')  # em_arg另分了一个文件，是否合适