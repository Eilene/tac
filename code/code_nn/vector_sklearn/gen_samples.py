# coding=utf-8

import xml.dom.minidom
from nltk.tokenize import WordPunctTokenizer

from split_sentences import *


# 训练语料的特征生成方法，都从annotation中出，测试语料不应访问到annotation，所以分开

# 初步处理
def gen_train_records(source_filename, ere_filename, annotation_filename):
    source_fp = open(source_filename)
    all_source_text = source_fp.read().decode("utf-8")  # 注意编码
    sentences = split_sentences(all_source_text)

    records = [] # records的每个元素是一个字典，包括目标，情感，前后中三个句子
    annotation_file = xml.dom.minidom.parse(annotation_filename)
    root = annotation_file.documentElement
    entity_list = root.getElementsByTagName('entity')  # 只有sentiment才有，所以不必区分sentiment和belief
    for i in range(len(entity_list)):
        # 要不要源，如果日后特征需要的话则要，涉及到情感none时找源问题和是否保留该条问题
        # 目标
        # 目前少一个去ere文件中找entity_id过程
        ere_id = entity_list[i].getAttribute('ere_id')
        offset = int(entity_list[i].getAttribute('offset'))
        length = int(entity_list[i].getAttribute('length'))
        text_em = entity_list[i].getElementsByTagName('text')
        text = text_em[0].firstChild.data
        target = {'ere_id': ere_id, 'offset': offset, 'length': length, 'text': text}
        # 情感
        st_em = entity_list[i].getElementsByTagName('sentiment')
        polarity = st_em[0].getAttribute('polarity')
        sarcasm = st_em[0].getAttribute('sarcasm')
        sentiment = {'polarity': polarity, 'sarcasm': sarcasm}  # sarcasm不需要，要不要直接存数
        # 上下文，包括前一个、当前、后一个句子
        context = find_context(offset, sentences, all_source_text)
        if context is None:  # 说明是在标签中出现的源，一般annotation中不会出现该情况
            # print 0
            continue
        # 生成一条记录加入
        rec = {'target': target, 'sentiment': sentiment, 'context': context}
        records.append(rec)

    # print records
    return records
# 初步处理后也可生成中间文件


# 生成训练X,Y
def gen_train_sample(records, embeddings_index, dim):
    x_samples = []
    y_labels = []

    for rec in records:
        # 标签（情感）
        if rec['sentiment']['polarity'] == 'pos':
            y_labels.append(1)
        elif rec['sentiment']['polarity'] == 'neg':
            y_labels.append(-1)
        else:  # 先3分类 (e27的数据里好像没有none的，e27_v2有）
            # y_labels.append(0)
            continue

        # 特征
        # 分词并去除标点符号
        english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        words_filtered = []
        for index in [0]:
            if rec['context'][index] != '':
                words = WordPunctTokenizer().tokenize(rec['context'][index]['text'])
                words_filtered.extend([word for word in words if not word in english_punctuations])  # 去除标点符号
        # 用词向量求句子向量
        senc_vector = [0.0] * dim
        word_count = 0
        for word in words_filtered:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                for i in range(dim):
                    word_count += 1
                    senc_vector[i] += embedding_vector[i]
        if word_count != 0:
            for i in range(dim):
                senc_vector[i] /= word_count
        # 词向量矩阵加入样本中
        x_samples.append(senc_vector)

    # print x_samples
    # print len(y_labels)
    return x_samples, y_labels


# 测试语料特征生成方法，不能访问annotation

# 初步处理
def gen_test_records(source_filename, ere_filename):
    # 读文本并分句
    source_fp = open(source_filename)
    all_source_text = source_fp.read().decode("utf-8")  # 注意编码
    sentences = split_sentences(all_source_text)  # 分句

    # 得到记录，包括目标、上下文；并找出源对应的实体id等信息
    records = []
    sources = []
    ere_file = xml.dom.minidom.parse(ere_filename)
    root = ere_file.documentElement
    entity_mention_list = root.getElementsByTagName('entity_mention')
    # print source_filename
    for i in range(len(entity_mention_list)):
        # 目标(目前只有实体，上同)
        ere_id = entity_mention_list[i].getAttribute('id')
        offset = int(entity_mention_list[i].getAttribute('offset'))  # 注意转整型
        length = int(entity_mention_list[i].getAttribute('length'))
        text_em = entity_mention_list[i].getElementsByTagName('mention_text')
        text = text_em[0].firstChild.data
        entity_mention = {'ere_id': ere_id, 'offset': offset, 'length': length, 'text': text}
        # 上下文
        context = find_context(offset, sentences, all_source_text)  # 查找target所在上下文句子
        # print entity_mention, context
        if context is None:  # 说明是在标签中出现的源，一般annotation中不会出现该情况
            sources.append(entity_mention)
            continue
        # 情感，先赋默认值
        sentiment = {'polarity': 'none', 'sarcasm': 'no'}
        # 生成一条记录加入
        rec = {'source': None, 'target': entity_mention, 'sentiment': sentiment, 'context': context}
        records.append(rec)

    # 匹配目标和源
    # 对每个源，向上文找到标签名称post/quote等，向下文找到匹配的</post></quote>，确定管辖范围
    # 如果中间遇到别的<post><quote>，计数，1开始，<>+1,</>-1,归0则范围划定
    # 每个目标，找到落在范围内的源，会有多个，最小区间为其源
    # 先简单写一版
    # 考虑调用写好的那个函数
    sources.sort(key=lambda x: x['offset'])
    # print len(sources)
    for i in range(len(records)):
        for j in range(len(sources)-1):
            if sources[j]['offset'] <= records[i]['target']['offset'] < sources[j+1]['offset']:
                records[i]['source'] = sources[j]
                break
        if records[i]['target']['offset'] >= sources[-1]['offset']:
            records[i]['source'] = sources[-1]

    # print records
    return records


# 生成测试X
def gen_test_sample(records, embeddings_index, dim):
    x_samples = []

    for rec in records:
        # 分词并去除标点符号
        english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
        words_filtered = []
        for index in [0]:
            if rec['context'][index] != '':
                words = WordPunctTokenizer().tokenize(rec['context'][index]['text'])
                words_filtered.extend([word for word in words if not word in english_punctuations])  # 去除标点符号
        # 用词向量求句子向量
        senc_vector = [0.0] * dim
        word_count = 0
        for word in words_filtered:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                for i in range(dim):
                    word_count += 1
                    senc_vector[i] += embedding_vector[i]
        if word_count != 0:
            for i in range(dim):
                senc_vector[i] /= word_count
        # 词向量矩阵加入样本中
        x_samples.append(senc_vector)

    # print len(x_samples)
    return x_samples
