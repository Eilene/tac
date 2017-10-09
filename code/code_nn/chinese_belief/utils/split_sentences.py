# coding=utf-8

import xml.dom.minidom
import re
import sys
import jieba

import json


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


# 分词
# 参数：一个句子，返回值：词及每个词在句中的偏移量
def word_segmentation(sentence):
    # 分词
    words = jieba.cut(sentence)
    # 去标点
    chinese_punctuations = ['，', '。', '：', '；', '？', '（', '）', '【', '】', '&', '！', '*', '@', '#', '$', '%',
                            '“', '”', '’', '‘']  # 但是标点符号也能表达感情，不删，怎么用？不过这里只是window文本
    word_filtered = [word for word in words if not word in chinese_punctuations]
    # 要不要去停用词？再看
    # 查找偏移量
    words_offset = []  # key：偏移量；value：词
    start = 0
    for word in word_filtered:
        index = sentence.find(word, start)
        if index == -1:
            print "error: cannot find word in sentence"
            sys.exit()
        start = index + len(word)
        word_offset = {'offset': index, 'text': word}
        words_offset.append(word_offset)
    return words_offset


# 检查某字符是否分句标志符号的函数；如果是，返回True，否则返回False
def find_token(cutlist, char):
    if char in cutlist:
        return True
    else:
        return False


def cut_sentences(cutlist, lines):
    l = []
    line = []
    for i in lines:
        if find_token(cutlist, i):
            line.append(i)
            l.append("".join(line))
            # l.append(i)
            line = []
        else:
            line.append(i)
    if len(line) != 0:
        l.append("".join(line))
    return l


# 分句
# 参数：source文件全部文本，返回值：句子及每句起始偏移量
def split_sentences(whole_text):
    # 先把各标签中的段落抽出来
    if re.search(r'</a>', whole_text) is not None:
        paras = re.findall(r'>([\s\S]*?)<[^a^i][^a]', whole_text)
        # [^a^i][^a]一个字符的标签不行，这里具体看没有这种情况才这么写
    else:
        paras = re.findall(r'>([\s\S]*?)<', whole_text)  # 一般是新闻文件，有<P>
    # 再分句
    cutlist = "。！？\n".decode('utf-8')  # 看情况加,，要不要按逗号加；可试试不同符号，不同级别分句效果
    texts = []
    for para in paras:
        if para != u'\n' and para != '':
            # print para
            sub_texts = cut_sentences(cutlist, para)
            texts.extend(sub_texts)
    texts = [text for text in texts if (text != u'\n' and text != '')]  # 过滤掉
    # 查找偏移量
    sencs = []
    start = 0
    for text in texts:
        index = whole_text.find(text, start)
        if index == -1:
            print "error: cannot find sentence in source text"
            sys.exit()
        start = index + len(text)
        sen = {'offset': index, 'text': text}  # 为了后面定位准确，这里没有去除前后的\n
        sencs.append(sen)
    # res_data = json.dumps(sencs, ensure_ascii=False, encoding="gb2312")
    # print res_data
    return sencs


# 查找所在句子、前句、后句，由于有序，使用折半查找
# 参数：词offset和用split_sentences得到的分句结果，需要的上文句子数，下文句子数；返回值：对应词的上下文句子
def find_context(word_offset, sencs, whole_text, above, below):
    cx = {}
    for i in range(-above, below+1):  # 初始化（根据需求，也可不做，则没有的就不含该key）
        cx[i] = ''
    begin = 0
    slen = len(sencs)
    last = slen - 1
    if sencs[0]['offset'] > word_offset:  # 比第一句还前的肯定是源，就不找了
        # print sencs[0]['offset'], word_offset
        return
    if begin > last:
        return  # 没找到，一般该词是源
    while begin <= last:
        mid = (begin + last) / 2
        if (mid == slen-1 and sencs[mid]['offset'] <= word_offset) \
                or (mid < len(sencs)-1 and sencs[mid]['offset'] <= word_offset < sencs[mid+1]['offset']):
            if sencs[mid]['offset']+len(sencs[mid]['text']) < word_offset:
                return  # 词offset大于该句子范围，说明在html标签内，是源
            # 当前句子
            cx[0] = sencs[mid]
            # 上文
            for i in range(min(mid, above)):
                # 判断和上句之间有没有html标签
                sub_string = whole_text[sencs[mid-i-1]['offset']+len(sencs[mid-1]['text']): sencs[mid-i]['offset']]
                if re.search(r'<[^>]+>', sub_string) is None:  # 如果有，不要上句
                    cx[-i-1] = sencs[mid-i-1]
                else:
                    break  # 上文没有，则再上文都没有
            # 下文
            for i in range(min(slen-1-mid, below+1)):
                # 判断和下句之间有没有html标签
                sub_string = whole_text[sencs[mid+i]['offset']+len(sencs[mid]['text']): sencs[mid+i+1]['offset']]
                if re.search(r'<[^>]+>', sub_string) is None:  # 如果有，不要下句
                    cx[i+1] = sencs[mid+i+1]
                else:
                    break  # 上文没有，则再上文都没有
            # 返回
            return cx
        elif word_offset >= sencs[mid]['offset']:
            begin = mid + 1
        elif word_offset < sencs[mid+1]['offset']:
            last = mid - 1


# 参数：source和ere文件名，所需上文、下文句子数；返回值：ere文件中所有entity_mention的id和对应句子
def get_contexts(source_filename, ere_filename, above, below):
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
        context = find_context(offset, sentences, all_source_text, above, below)  # 查找target所在上下文句子
        if context is not None:  # 如果是None，则说明是源，出现在标签中的，没有上下文
            ids_with_contexts[ere_id] = context
            # -1：上一个句子；0：当前句子；1：下一个句子
            # text_em = entity_mention_list[i].getElementsByTagName('mention_text')
            # text = text_em[0].firstChild.data
            # res_data = json.dumps(ids_with_contexts[ere_id], ensure_ascii=False, encoding="gb2312")
            # print ere_id, text, offset, res_data

    return ids_with_contexts  # key为id，value为上下文句子


# 将上下文字典中的文本按序拼接成字符串
def context_dict_to_string(context_dict, above, below):
    keys = [k for k in range(-above, below + 1)]
    context = ''
    for key in keys:
        if (context_dict is not None) and ('text' in context_dict[key]):
            # 如果出现None可能是源，一般不出现
            context += context_dict[key]['text']
            context = context.replace('\n', '')  # 去掉换行
    # if context == '':
    #     context = "None"

    # 先在这里去掉链接等标签？？去掉的话target在句子中的具体offset没法找了，不用则可去；用则先不去，在处理特征时再去
    re_h = re.compile('</?\w+[^>]*>')  # HTML标签
    # print context
    context = re_h.sub("", context)
    # print context

    return context


if __name__ == '__main__':
    # 测试
    id_sencs = get_contexts("../../../data/2016E61/data/source/0a4aba6cda0dc08f8288ce5655529f93.cmp.txt",
                            "../../../data/2016E61/data/ere/0a4aba6cda0dc08f8288ce5655529f93.rich_ere.xml", 3, 3)
    # 3,3，表示需要上文3个句子，下文3个句子，加上当前句，共7个句子
    # res_data = json.dumps(id_sencs, ensure_ascii=False, encoding="gb2312")
    # print res_data


