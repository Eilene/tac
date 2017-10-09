# coding=utf-8

import nltk
import nltk.data
import xml.dom.minidom
import re
import sys


def get_window_text(window_length, sentence, sen_offset, target_text, target_offset):
    window_text = ''
    offset_in_sentence = target_offset - sen_offset
    target_words = nltk.word_tokenize(target_text)
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
    words = nltk.WordPunctTokenizer().tokenize(sentence)
    # 去标点
    spanish_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '"', "'"]
    word_filtered = [word for word in words if not word in spanish_punctuations]
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


# 抽取段落，即post、quote等段落标签间的
# def extract_paras(whole_text):
#     a = ['<post', '<quote', '<headline>', '<P>', '</quote>', '</post>', '</P>']
#     b = ['<img', '<a', '</a>']

# 分句
# 参数：source文件全部文本，返回值：句子及每句起始偏移量
def split_sentences(whole_text):
    # 先把各标签中的段落抽出来
    if re.search(r'</a>', whole_text) is not None:
        paras = re.findall(r'>([\s\S]*?)<[^a^i][^a]', whole_text)
        # [^a^i][^a]一个字符的标签不行，这里具体看没有这种情况才这么写
    else:
        paras = re.findall(r'>([\s\S]*?)<', whole_text)  # 一般是新闻文件，有<P>
    # print paras
    # 再分句
    # 中间\n好像都未分，是不是本来就不该分，会有强行换行存在
    # 有些符号没分，如“...”，好像不太合理
    texts = []
    tokenizer = nltk.data.load('tokenizers/punkt/spanish.pickle')
    for para in paras:
        if para != '\n':  # 会不会有多个回车的？如果有可能影响后面index，必须处理掉
            sub_texts = tokenizer.tokenize(para)
            # print sub_texts
            texts.extend(sub_texts)
    # 查找偏移量
    sencs = []
    start = 0
    for text in texts:
        index = whole_text.find(text, start)  # 有问题了，有的标签中和外面一样的，怎么办；把标签offset范围找好？
        if index == -1:
            print "error: cannot find sentence in source text"
            sys.exit()
        start = index + len(text)
        sen = {'offset': index, 'text': text}  # 为了后面定位准确，这里没有去除前后的\n
        sencs.append(sen)
    # print sencs

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
                # print sencs[mid]['offset'], len(sencs[mid]['text']), word_offset  # 都在这
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
                # 判断和和下句之间有没有html标签
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
            context = context_dict_to_string(context, above, below)
            # -1：上一个句子；0：当前句子；1：下一个句子
            # text_em = entity_mention_list[i].getElementsByTagName('mention_text')
            # text = text_em[0].firstChild.data
            # print ere_id, text, offset, ids_with_contexts[ere_id]
        # else:
        #     print ere_id, offset

    return ids_with_contexts  # key为id，value为上下文句子


# 将上下文字典中的文本按序拼接成字符串
def context_dict_to_string(context_dict, above, below):
    keys = [k for k in range(-above, below + 1)]
    context = ''
    for key in keys:
        if (context_dict is not None) and ('text' in context_dict[key]):
            # 如果出现None可能是源，一般不出现
            context += context_dict[key]['text']
            context = context.replace('\n', ' ')  # 去掉换行
    # if context == '':
    #     context = "None"

    # 先在这里去掉链接等标签？？去掉的话target在句子中的具体offset没法找了，不用则可去；用则先不去，在处理特征时再去
    re_h = re.compile('</?\w+[^>]*>')  # HTML标签
    # print context
    context = re_h.sub("", context)
    # print context

    return context


if __name__ == '__main__':
    # id_sencs = get_contexts("../data/2016E27_V2/data/source/0ba982819aaf9f5b94a7cebd48ac6018.cmp.txt",
    #                         "../data/2016E27_V2/data/ere/0ba982819aaf9f5b94a7cebd48ac6018.rich_ere.xml", 3, 3)
    # print id_sencs

    # 测试
    id_sencs = get_contexts("../../../data/spa/source/ENG_DF_001471_20131112_G00A0FOVI.xml",  # 这几个文件单抽算了
                            "../../../data/spa/ere/ENG_DF_001471_20131112_G00A0FOVI.rich_ere.xml", 0, 0)
    # 还是存在不少同一句中多个target的，所以抽出很多相同句；怎么处理呢？改抽小句？
    # 若3,3，表示需要上文3个句子，下文3个句子，加上当前句，共7个句子
    # print id_sencs


