# coding=utf-8

import nltk
import nltk.data
import xml.dom.minidom
import re


# 分句
# 传入source文件全部文本，返回句子及每句起始偏移量
def split_sentences(whole_text):
    # 去除文本中的<post>等标签部分
    dr = re.compile(r'<[^>]+>', re.S)
    whole_text_without_tag = dr.sub('', whole_text)
    # 分句
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    texts = tokenizer.tokenize(whole_text_without_tag)
    # 查找每个句子起始偏移量
    start = 0
    sencs = []
    for text in texts:
        index = whole_text.find(text, start)
        sen = {'offset': index, 'text': text}
        sencs.append(sen)
        start = index + len(text)
    return sencs


# 查找所在句子、前句、后句，由于有序，使用折半查找
# 传入词offset和用split_sentences得到的分句结果，返回对应词的上下文句子
def find_context(word_offset, sencs):
    cx = {}
    begin = 0
    last = len(sencs) - 1
    if begin > last:
        return
    while begin <= last:
        mid = (begin + last) / 2
        if (mid == len(sencs)-1 and sencs[mid]['offset'] <= word_offset)\
                or (mid < len(sencs)-1 and sencs[mid]['offset'] <= word_offset < sencs[mid+1]['offset']):
            cx[0] = sencs[mid]['text']
            if mid > 0:
                cx[-1] = sencs[mid-1]['text']
            else:
                cx[-1] = ''
            if mid < len(sencs)-1:
                cx[1] = sencs[mid+1]['text']
            else:
                cx[1] = ''
            return cx
        elif word_offset >= sencs[mid]['offset']:
            begin = mid + 1
        elif word_offset < sencs[mid+1]['offset']:
            last = mid - 1


# 传入source和ere文件，返回ere文件中所有entity_mention的id和对应句子
def get_contexts(source_filename, ere_filename):
    source_fp = open(source_filename)
    all_source_text = source_fp.read().decode("utf-8")  # 注意编码
    sentences = split_sentences(all_source_text)  # 分句

    ere_file = xml.dom.minidom.parse(ere_filename)
    root = ere_file.documentElement
    entity_mention_list = root.getElementsByTagName('entity_mention')
    ids_with_contexts = {}
    for i in range(len(entity_mention_list)):
        ere_id = entity_mention_list[i].getAttribute('id')
        offset = int(entity_mention_list[i].getAttribute('offset'))  # 注意转整型
        ids_with_contexts[ere_id] = find_context(offset, sentences)  # 查找target所在上下文句子
        # -1：上一个句子；0：当前句子；1：下一个句子

        # text_em = entity_mention_list[i].getElementsByTagName('mention_text')
        # text = text_em[0].firstChild.data
        # print ere_id, text, offset, ids_with_contexts[ere_id]

    return ids_with_contexts  # key为id，value为上下文句子


if __name__ == '__main__':
    id_sencs = get_contexts("0a421343005f3241376fa01e1cb3c6fb.cmp.txt", "0a421343005f3241376fa01e1cb3c6fb.rich_ere.xml")
    print id_sencs


