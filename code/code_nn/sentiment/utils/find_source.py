# coding=utf-8
import re
import xml.dom.minidom
import os

import sys
reload(sys)
sys.setdefaultencoding("utf8")


def find_sources(test_files, source_dir, ere_dir):
    for i in range(len(test_files)):
        # source
        source_filepath = source_dir + test_files[i]['filename'] + ".cmp.txt"
        if os.path.exists(source_filepath) is False:
            # prefix_length = len('ENG_DF_000183_20150408_F0000009B')
            # source_filepath = source_dir + test_files[i]['filename'][:prefix_length] + ".xml"  # 论坛xml
            # print source_filepath
            continue  # 没调好，先放放
        source_fp = open(source_filepath)
        all_source_text = source_fp.read().decode("utf-8")  # 注意编码
        source_fp.close()
        # if all_source_text is None:
        #     print source_filepath
        # print source_filepath, all_source_text
        
        # ere, entities
        ere_filepath = ere_dir + test_files[i]['filename'] + '.rich_ere.xml'
        ere_file = xml.dom.minidom.parse(ere_filepath)
        ere_root = ere_file.documentElement
        entity_mention_list = ere_root.getElementsByTagName('entity_mention')
        entity_mentions = []
        for j in range(len(entity_mention_list)):
            ere_id = entity_mention_list[j].getAttribute('id')
            offset = int(entity_mention_list[j].getAttribute('offset'))
            length = int(entity_mention_list[j].getAttribute('length'))
            text = entity_mention_list[j].getElementsByTagName('mention_text')
            text = text[0]
            text_text = text.firstChild.data
            entity_mention = {'ere_id': ere_id, 'offset': offset, 'length': length, 'text': text_text}
            entity_mentions.append(entity_mention)

        # if source_filepath[-3:] == 'xml':
        #     print ere_filepath, len(entity_mentions)
            
        if 'entity' in test_files[i]:
            for j in range(len(test_files[i]['entity'])):  # 其实只需非none的找即可
                ere_id = test_files[i]['entity'][j]['entity_mention_id']
                offset = test_files[i]['entity'][j]['entity_mention_offset']
                length = test_files[i]['entity'][j]['entity_mention_length']
                predict_source = find_source(offset, length, ere_id, all_source_text, entity_mentions)
                if predict_source is not None:
                    test_files[i]['entity'][j]['predict_source_id'] = predict_source['ere_id']
                    test_files[i]['entity'][j]['predict_source_offset'] = predict_source['offset']
                    test_files[i]['entity'][j]['predict_source_length'] = predict_source['length']
                    test_files[i]['entity'][j]['predict_source_text'] = predict_source['text'].decode("utf-8")

        if 'relation' in test_files[i]:
            for j in range(len(test_files[i]['relation'])):
                ere_id = test_files[i]['relation'][j]['rel_arg1_mention_id']
                offset = test_files[i]['relation'][j]['rel_arg1_mention_offset']
                length = test_files[i]['relation'][j]['rel_arg1_mention_length']
                predict_source = find_source(offset, length, ere_id, all_source_text, entity_mentions)
                if predict_source is not None:
                    test_files[i]['relation'][j]['predict_source_id'] = predict_source['ere_id']
                    test_files[i]['relation'][j]['predict_source_offset'] = predict_source['offset']
                    test_files[i]['relation'][j]['predict_source_length'] = predict_source['length']
                    test_files[i]['relation'][j]['predict_source_text'] = predict_source['text'].decode("utf-8")

        if 'event' in test_files[i]:
            for j in range(len(test_files[i]['event'])):
                ere_id = 'noid'
                offset = test_files[i]['event'][j]['trigger_offset']
                length = test_files[i]['event'][j]['trigger_length']
                predict_source = find_source(offset, length, ere_id, all_source_text, entity_mentions)
                if predict_source is not None:
                    test_files[i]['event'][j]['predict_source_id'] = predict_source['ere_id']
                    test_files[i]['event'][j]['predict_source_offset'] = predict_source['offset']
                    test_files[i]['event'][j]['predict_source_length'] = predict_source['length']
                    test_files[i]['event'][j]['predict_source_text'] = predict_source['text'].decode("utf-8")

    return test_files


def find_source(offset, length, id, all_source_text, entity_mentions):
    predict_text_offset, predict_source_text = match(all_source_text, offset, length, id)
    predict_source = find_source_id(predict_text_offset, entity_mentions)
    return predict_source


def match(text, offset, length, id):
    if text[offset-len('<quote orig_author="'):offset] == '<quote orig_author="':
        # print id, '   entity in xml: <quote orig_author="', text[offset:offset+length]
        return None, None
    if text[offset-len('<post author="'):offset] == '<post author="':
        # print id, '   entity in xml: <post author="', text[offset:offset+length]
        return None, None
    regex_author = re.compile(r' author="(.*?)"')
    regex_orig = re.compile(r' orig_author="(.*?)"')
    index = offset; stack = 0
    while index >= 5:
        # '<quote'
        if index >= 6 and text[index-6:index] == "<quote":
            if stack > 0:
                stack = stack - 1
            else:
                if text[index] == ">":
                    # print id, "   <quote>", text[offset:offset+length]
                    return None, None
                # print "<quote> : ", regex_orig.search(text[index:index+100]).group()[14:-1]
                return index+14, regex_orig.search(text[index:index+100]).group()[14:-1]
        # '</quote>'
        if index >= 8 and text[index-8:index] == "</quote>":
            stack = stack + 1
        # '<post'
        if text[index-5:index] == "<post":
            # print "<post> : ", regex_author.search(text[index:index+100]).group()[9:-1]
            return index+9, regex_author.search(text[index:index+100]).group()[9:-1]
        index = index - 1


def find_source_id(offset, entity_mentions):
    for i in range(len(entity_mentions)):  # 若在这里，从这里删去，加入sources
        if entity_mentions[i]['offset'] == offset:
            return entity_mentions[i]
    return
