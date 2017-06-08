# coding=utf-8

import re
import sys
import nltk
import nltk.data


# 分句
# 参数：source文件全部文本，返回值：句子及每句起始偏移量
def split_sentences(whole_text):
    # 先把各标签中的段落抽出来
    paras = re.findall(r'>([\s\S]*?)<', whole_text)  # 是否会有非html标签的><，注意看具体文件
    # while '\n' in paras:
    #     paras.remove('\n')  # 去掉空段落
    # print paras
    # 再分句
    # 中间\n好像都未分，是不是本来就不该分，会有强行换行存在
    # 有些符号没分，如“...”，好像不太合理
    texts = []
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for para in paras:
        if para != '\n':  # 会不会有多个回车的？如果有可能影响后面index，必须处理掉
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
        sen = {'offset': index, 'text': text}
        sencs.append(sen)
    # print sencs
    return sencs


# 查找所在句子、前句、后句，由于有序，使用折半查找
# 参数：词offset和用split_sentences得到的分句结果，返回值：对应词的上下文句子
def find_context(word_offset, sencs, whole_text):
    cx = {}
    begin = 0
    last = len(sencs) - 1
    if sencs[0]['offset'] > word_offset:  # 比第一句还前的肯定是源，就不找了
        return
    if begin > last:
        return  # 没找到，一般该词是源
    while begin <= last:
        mid = (begin + last) / 2
        if (mid == len(sencs)-1 and sencs[mid]['offset'] <= word_offset) \
                or (mid < len(sencs)-1 and sencs[mid]['offset'] <= word_offset < sencs[mid+1]['offset']):
            if sencs[mid]['offset']+len(sencs[mid]['text']) < word_offset:
                return  # 词offset大于该句子范围，说明在html标签内，是源
            cx[0] = sencs[mid]
            if mid > 0:
                # 判断和上句之间有没有html标签
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
