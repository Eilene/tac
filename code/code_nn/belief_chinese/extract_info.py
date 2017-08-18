# coding=utf-8
import os
import shutil
import xml.dom.minidom

import pandas as pd

from utils.constants import *
from utils.split_sentences import *

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger()


def get_window_text(window_length, sentence, sen_offset, target_text, target_offset):
    window_text = ''
    offset_in_sentence = target_offset - sen_offset
    target_words = list(jieba.cut(target_text))
    # res_data = json.dumps(target_words, ensure_ascii=False, encoding="gb2312")
    # print res_data
    # 共window_length，从target两端延，若target长，直接截target，从头截
    target_len = len(target_words)
    if target_len >= window_length:
        for k in range(window_length):
            window_text += target_words[k]
    else:
        # 分词，得到词和在句中偏移量
        words = word_segmentation(sentence)
        index = 0
        for k in range(len(words)):
            if offset_in_sentence == words[k]['offset']:
                index = k
                # m = k
                # for tw in target_words:  # 试下等不等，若无意外可去掉
                #     m += 1
                #     if m > len(words) or tw != words[m]['text']:
                #         print 'error:', tw, words[m]['text']
                #         return window_text
        # 需截的左右界
        remained_length = target_len - window_length
        left = max(0, index - int(remained_length / 2))
        right = min(len(words), left + window_length)
        for k in range(left, right):
            window_text += words[k]['text'] + ' '
        return window_text


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
                    # if rel_arg_context == '':
                    #     print rel_arg_text, rel_arg_id, rel_arg_mention_offset
                    # 从上下文中进一步提取窗口词
                    window_length = 10
                    # print rel_arg_context_dict
                    if rel_arg_context_dict is not None:  # 有可能，是源
                        sen = rel_arg_context_dict[0]['text']
                        sen_offset = rel_arg_context_dict[0]['offset']
                    else:
                        sen = ''
                        sen_offset = 0
                    window_text = get_window_text(window_length, sen, sen_offset, rel_arg_text, rel_arg_mention_offset)
                    return rel_arg_entity_type, rel_arg_entity_specificity, rel_arg_mention_noun_type, \
                           rel_arg_mention_offset, rel_arg_mention_length, rel_arg_context, window_text, sen, sen_offset


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
            # if rel_arg_context == '':
            #     print rel_arg_text, rel_arg_id, part_name, rel_arg_context
            # 从上下文中进一步提取窗口词
            if rel_arg_context_dict is not None:
                window_length = 10
                sen = rel_arg_context_dict[0]['text']
                sen_offset = rel_arg_context_dict[0]['offset']
                window_text = get_window_text(window_length, sen, sen_offset, rel_arg_text, rel_arg_mention_offset)
            else:  # 会出现？？
                window_text = ''
                sen = ''
                sen_offset = 0
            return rel_arg_filler_type, rel_arg_mention_offset, rel_arg_mention_length, rel_arg_context, window_text, \
                   sen, sen_offset


def extract_relation_each_file(source_filepath, ere_filepath, annotation_filepath, part_name, with_none):
    relation_records_each_file = []

    source_fp = open(source_filepath)
    all_source_text = source_fp.read().decode("utf-8")  # 注意编码
    source_fp.close()
    sentences = split_sentences(all_source_text)  # 分句

    ere_file = xml.dom.minidom.parse(ere_filepath)
    ere_root = ere_file.documentElement
    relation_list = ere_root.getElementsByTagName('relation')
    entity_list = ere_root.getElementsByTagName('entity')
    filler_list = ere_root.getElementsByTagName('filler')

    annotation_file = xml.dom.minidom.parse(annotation_filepath)
    annotation_root = annotation_file.documentElement
    annotation_sentiment_list = annotation_root.getElementsByTagName('belief_annotations')
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

            # belief type
            for k in range((len(annotation_relation_list))):
                annotation_relation_id = annotation_relation_list[k].getAttribute('ere_id')
                if annotation_relation_id == relation_mention_id:
                    be_em = annotation_relation_list[k].getElementsByTagName('belief')
                    if len(be_em) == 0:
                        # logger.info("错误：无情感标签。" + " " + part_name + " " + annotation_relation_id +
                        #             " " + relation_mention_id)
                        print part_name, relation_mention_id
                    label_type = be_em[0].getAttribute('type')
                    break
            # if with_none is False and label_type == 'na':  # 感觉判断也没必要，na少数
            #     break  # 如果为none则丢弃该样本

            # rel_arg是entity
            # 基本信息
            rel_arg1 = relation_mention_list[j].getElementsByTagName('rel_arg1')
            rel_arg1 = rel_arg1[0]
            rel_arg1_id = rel_arg1.getAttribute('entity_id')
            rel_arg1_mention_id = rel_arg1.getAttribute('entity_mention_id')
            rel_arg1_role = rel_arg1.getAttribute('role')
            rel_arg1_text = rel_arg1.firstChild.data
            # 所属entity及entity mention信息
            # print part_name
            rel_arg1_entity_type, rel_arg1_entity_specificity, rel_arg1_mention_noun_type, rel_arg1_mention_offset, \
            rel_arg1_mention_length, rel_arg1_context, rel_arg1_window_text, rel_arg1_sentence, rel_arg1_sentence_offset\
                = rel_arg_entity_info(entity_list, rel_arg1_id, rel_arg1_mention_id, rel_arg1_text, sentences)

            # rel_arg，同上
            rel_arg2 = relation_mention_list[j].getElementsByTagName('rel_arg2')
            rel_arg2 = rel_arg2[0]
            rel_arg2_role = rel_arg2.getAttribute('role')
            rel_arg2_text = rel_arg2.firstChild.data
            rel_arg2_id = rel_arg2.getAttribute('entity_id')
            if rel_arg2_id != '':
                rel_arg2_mention_id = rel_arg2.getAttribute('entity_mention_id')
                # 所属entity及entity mention信息
                rel_arg2_entity_type, rel_arg2_entity_specificity, rel_arg2_mention_noun_type, rel_arg2_mention_offset, \
                rel_arg2_mention_length, rel_arg2_context, rel_arg2_window_text, rel_arg2_sentence, rel_arg2_sentence_offset = \
                    rel_arg_entity_info(entity_list, rel_arg2_id, rel_arg2_mention_id, rel_arg2_text, sentences)
                rel_arg2_is_filler = 0
            else:  # rel_arg2有的不是entity是filler，先简单处理
                rel_arg2_is_filler = 1
                rel_arg2_id = rel_arg2.getAttribute('filler_id')
                # if rel_arg2_id == '':
                #     logger.info("错误：参数不是entity或filler。" + " " + part_name + " " + relation_mention_id)
                rel_arg2_entity_type, rel_arg2_mention_offset, rel_arg2_mention_length, rel_arg2_context, \
                rel_arg2_window_text, rel_arg2_sentence, rel_arg2_sentence_offset = \
                    rel_arg_filler_info(filler_list, rel_arg2_id, rel_arg2_text, sentences)
                rel_arg2_mention_id = ''
                rel_arg2_entity_specificity = ''
                rel_arg2_mention_noun_type = ''

            # trigger
            trigger = relation_mention_list[j].getElementsByTagName('trigger')  # ？待查
            if len(trigger) == 0:
                trigger_offset = 0
                trigger_length = 0
                trigger_text = ""
                trigger_context = ""
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

            # actual source
            source = be_em[0].getElementsByTagName('source')
            if label_type == 'none' or len(source) == 0:
                source_id = ''
                source_offset = 0
                source_length = 0
                source_text = ''
            else:
                source = source[0]
                source_id = source.getAttribute('ere_id')
                source_offset = int(source.getAttribute('offset'))
                source_length = int(source.getAttribute('length'))
                source_text = source.firstChild.data

            relation_record = {
                'file': part_name,
                'relation_id': relation_id, 'relation_type': relation_type, 'relation_subtype': relation_subtype,
                'relation_mention_id': relation_mention_id, 'relation_mention_realis': relation_mention_realis,
                'rel_arg1_id': rel_arg1_id, 'rel_arg1_mention_id': rel_arg1_mention_id,
                'rel_arg1_role': rel_arg1_role, 'rel_arg1_text': rel_arg1_text,
                'rel_arg1_entity_type': rel_arg1_entity_type,
                'rel_arg1_entity_specificity': rel_arg1_entity_specificity,
                'rel_arg1_mention_noun_type': rel_arg1_mention_noun_type,
                'rel_arg1_mention_offset': rel_arg1_mention_offset,
                'rel_arg1_mention_length': rel_arg1_mention_length, 'rel_arg1_context': rel_arg1_context,
                'rel_arg1_window_text': rel_arg1_window_text, 'rel_arg1_sentence': rel_arg1_sentence,
                'rel_arg1_sentence_offset': rel_arg1_sentence_offset,
                'rel_arg2_id': rel_arg2_id, 'rel_arg2_mention_id': rel_arg2_mention_id,
                'rel_arg2_role': rel_arg2_role, 'rel_arg2_text': rel_arg2_text,
                'rel_arg2_entity_type': rel_arg2_entity_type,
                'rel_arg2_entity_specificity': rel_arg2_entity_specificity,
                'rel_arg2_mention_noun_type': rel_arg2_mention_noun_type,
                'rel_arg2_mention_offset': rel_arg2_mention_offset,
                'rel_arg2_mention_length': rel_arg2_mention_length, 'rel_arg2_context': rel_arg2_context,
                'rel_arg2_is_filler': rel_arg2_is_filler, 'rel_arg2_window_text': rel_arg2_window_text,
                'rel_arg2_sentence': rel_arg2_sentence,  'rel_arg2_sentence_offset': rel_arg2_sentence_offset,
                'trigger_offset': trigger_offset, 'trigger_length': trigger_length, 'trigger_text': trigger_text,
                'trigger_context': trigger_context,
                'label_type': label_type,
                'source_id': source_id, 'source_offset': source_offset, 'source_length': source_length,
                'source_text': source_text
            }

            relation_records_each_file.append(relation_record)

    return relation_records_each_file


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
                    # if em_arg_context == "''":
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
            # if em_arg_context == "''":
            #     print em_arg_text, em_arg_id, part_name, em_arg_context
            return em_arg_filler_type, em_arg_mention_offset, em_arg_mention_length, em_arg_context


def extract_event_each_file(source_filepath, ere_filepath, annotation_filepath, part_name, with_none):
    event_records_each_file = []
    em_args_each_file = []

    source_fp = open(source_filepath)
    all_source_text = source_fp.read().decode("utf-8")  # 注意编码
    source_fp.close()
    sentences = split_sentences(all_source_text)  # 分句

    ere_file = xml.dom.minidom.parse(ere_filepath)
    ere_root = ere_file.documentElement
    hopper_list = ere_root.getElementsByTagName('hopper')
    entity_list = ere_root.getElementsByTagName('entity')
    filler_list = ere_root.getElementsByTagName('filler')

    annotation_file = xml.dom.minidom.parse(annotation_filepath)
    annotation_root = annotation_file.documentElement
    annotation_sentiment_list = annotation_root.getElementsByTagName('belief_annotations')
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
                    be_em = annotation_event_list[k].getElementsByTagName('belief')
                    # if len(st_em) == 0:
                    #     logger.info("错误：无情感标签。" + " " + part_name + " " + annotation_event_id +
                    #                 " " + event_mention_id)
                    label_type = be_em[0].getAttribute('type')
                    break
            # if with_none is False and label_type == 'none':
            #     break  # 如果为none则丢弃该样本

            # trigger
            trigger = event_mention_list[j].getElementsByTagName('trigger')
            trigger = trigger[0]
            trigger_offset = int(trigger.getAttribute('offset'))
            trigger_length = int(trigger.getAttribute('length'))
            trigger_text = trigger.firstChild.data
            # 上下文信息
            above = 3
            below = 3  # 可调
            trigger_context_dict = find_context(trigger_offset, sentences, trigger_text, above, below)
            # 拼成一个字符串
            trigger_context = context_dict_to_string(trigger_context_dict, above, below)
            # 从上下文中进一步提取窗口词
            window_length = 10
            sen = trigger_context_dict[0]['text']
            sen_offset = trigger_context_dict[0]['offset']
            window_text = get_window_text(window_length, sen, sen_offset, trigger_text, trigger_offset)

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
                    # if em_arg_id == "":
                    #     logger.info("错误：参数不是entity或filler。" + " " + part_name + " " + event_mention_id)
                    em_arg_entity_type, em_arg_mention_offset, em_arg_mention_length, em_arg_context = \
                        em_arg_filler_info(filler_list, em_arg_id, em_arg_text, sentences)
                    em_arg_mention_id = ""
                    em_arg_entity_specificity = ""
                    em_arg_mention_noun_type = ""
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

            # actual source
            source = be_em[0].getElementsByTagName('source')
            if label_type == 'none' or len(source) == 0:
                source_id = ''
                source_offset = 0
                source_length = 0
                source_text = ''
            else:
                source = source[0]
                source_id = source.getAttribute('ere_id')
                source_offset = int(source.getAttribute('offset'))
                source_length = int(source.getAttribute('length'))
                source_text = source.firstChild.data

            event_record = {
                'file': part_name, 'hopper_id': hopper_id, 'event_mention_id': event_mention_id,
                'event_mention_type': event_mention_type, 'event_mention_subtype': event_mention_subtype,
                'event_mention_realis': event_mention_realis, 'event_mention_ways': event_mention_ways,
                'trigger_offset': trigger_offset, 'trigger_length': trigger_length,
                'trigger_text': trigger_text, 'trigger_context': trigger_context,
                'trigger_window_text': window_text, 'trigger_sentence': sen, 'trigger_sentence_offset': sen_offset,
                'em_arg_num': em_arg_num,
                'label_type': label_type,
                'source_id': source_id, 'source_offset': source_offset, 'source_length': source_length,
                'source_text': source_text
            }

            event_records_each_file.append(event_record)

    return event_records_each_file, em_args_each_file


def write_to_csv(records, filename):
    df = pd.DataFrame(records)
    # logger.debug('记录条数：%d', len(records))
    # logger.debug('记录维数：(%d, %d)', df.shape[0], df.shape[1])
    df.to_csv(filename, encoding="utf-8", index=None)


def traverse_and_write_mid_files(source_dir, ere_dir, annotation_dir,
                                 entity_info_dir, relation_info_dir, event_info_dir, em_args_dir, with_none):
    # 创建中间数据文件夹，若已有，删除重建
    if os.path.exists(relation_info_dir):
        shutil.rmtree(relation_info_dir)
    if os.path.exists(event_info_dir):
        shutil.rmtree(event_info_dir)
    if os.path.exists(em_args_dir):
        shutil.rmtree(em_args_dir)
    os.makedirs(relation_info_dir)
    os.makedirs(event_info_dir)
    os.makedirs(em_args_dir)

    # 遍历源数据集文件夹，生成中间数据文件
    ere_suffix = ".rich_ere.xml"
    ere_suffix_length = len(ere_suffix)
    for parent, dirnames, ere_filenames in os.walk(ere_dir):
        for ere_filename in ere_filenames:  # 有的source有多个ere和annotation，所以有后缀，一对多
            part_name = ere_filename[:-ere_suffix_length]
            source_filepath = source_dir + part_name + ".cmp.txt"
            ere_filepath = ere_dir + ere_filename
            annotation_filepath = annotation_dir + part_name + ".best.xml"
            if os.path.exists(source_filepath) is False:  # 不存在，则是xml
                # continue
                prefix_length = len('CMN_DF_000020_20150228_F000000CW')  # 由于给的数据命名不统一，需要这样做
                source_filepath = source_dir + part_name[:prefix_length] + ".xml"  # 论坛和新闻xml
                print prefix_length, source_filepath
                ere_filepath = ere_dir + ere_filename
                annotation_filepath = annotation_dir + part_name + ".best.xml"
            print part_name
            # relation
            relation_records = extract_relation_each_file(source_filepath, ere_filepath, annotation_filepath,
                                                          part_name, with_none)
            if len(relation_records) != 0:
                write_to_csv(relation_records, relation_info_dir + part_name + '.csv')
            # event
            event_records, em_args = extract_event_each_file(source_filepath, ere_filepath, annotation_filepath,
                                                             part_name, with_none)
            if len(event_records) != 0:
                write_to_csv(event_records, event_info_dir + part_name + '.csv')
                if len(em_args) != 0:
                    write_to_csv(em_args, em_args_dir + part_name + '.csv')

    # source文件实际可以在外面处理好，这里重复了3次，以后看看优化下

if __name__ == '__main__':
    traverse_and_write_mid_files(source_dir, ere_dir, annotation_dir, entity_info_dir,
                                 relation_info_dir, event_info_dir, em_args_dir, True)