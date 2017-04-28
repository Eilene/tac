# coding=utf-8

import nltk
import nltk.data
import xml.dom.minidom
import re
import sys


# 分句
# 传入source文件全部文本，返回句子及每句起始偏移量
def split_sentences(whole_text):
    # 先把各标签中的段落抽出来
    paras = re.findall(r'>([\s\S]*?)<', whole_text)
    paras.remove('\n')
    # 再分句
    # 中间\n好像都未分，是不是本来就不该分，会有强行换行存在
    # 有些符号没分，如“...”，好像不太合理
    texts = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for para in paras:
        sub_texts = tokenizer.tokenize(para)
        texts.extend(sub_texts)
    # 查找偏移量
    sencs = []
    start = 0
    for text in texts:
        index = whole_text.find(text, start)
        if index == -1:
            print "error: cannot find sentence in source text"
            sys.exit()
        start = index + len(text)
        sen = {'offset': index, 'text': text.strip('\n')}
        sencs.append(sen)
    # print len(sencs)
    return sencs


# 查找所在句子、前句、后句，由于有序，使用折半查找
# 传入词offset和用split_sentences得到的分句结果，返回对应词的上下文句子
def find_context(word_offset, sencs, whole_text):
    cx = {}
    begin = 0
    last = len(sencs) - 1
    if begin > last:
        return
    while begin <= last:
        mid = (begin + last) / 2
        if (mid == len(sencs)-1 and sencs[mid]['offset'] <= word_offset)\
                or (mid < len(sencs)-1 and sencs[mid]['offset'] <= word_offset < sencs[mid+1]['offset']):
            cx[0] = sencs[mid]
            if mid > 0:
                # 判断和和上句之间有没有html标签
                # 由于strip，截取长度不太精确，但应该不影响
                sub_string = whole_text[sencs[mid-1]['offset']+len(sencs[mid-1]['text']): sencs[mid]['offset']]
                if re.search(r'<[^>]+>', sub_string):  # 如果有，不要上句
                    cx[-1] = ''
                else:
                    cx[-1] = sencs[mid-1]
            else:
                cx[-1] = ''
            if mid < len(sencs)-1:
                # 判断和和下句之间有没有html标签
                sub_string = whole_text[sencs[mid]['offset']+len(sencs[mid]['text']): sencs[mid+1]['offset']]
                if re.search(r'<[^>]+>', sub_string):  # 如果有，不要下句
                    cx[1] = ''
                else:
                    cx[1] = sencs[mid+1]
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
        context = find_context(offset, sentences, all_source_text)  # 查找target所在上下文句子
        if context is not None:  # 如果是None，则说明是源，出现在标签中的，没有上下文
            ids_with_contexts[ere_id] = context
            # -1：上一个句子；0：当前句子；1：下一个句子
            # text_em = entity_mention_list[i].getElementsByTagName('mention_text')
            # text = text_em[0].firstChild.data
            # print ere_id, text, offset, ids_with_contexts[ere_id]

    return ids_with_contexts  # key为id，value为上下文句子


if __name__ == '__main__':
    id_sencs = get_contexts("data\\source\\0ba982819aaf9f5b94a7cebd48ac6018.cmp.txt",
                            "data\\ere\\0ba982819aaf9f5b94a7cebd48ac6018.rich_ere.xml")
    print id_sencs


