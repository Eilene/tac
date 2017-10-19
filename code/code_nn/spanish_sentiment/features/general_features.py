# coding=utf-8

import nltk
from pattern.en import lemma
from pattern.en import sentiment
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# def gen_all_general_features(contexts):
#     vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words='spanish', max_features=300, binary=True)
#     tfidf_features = vec.fit_transform(contexts).toarray()
#     print tfidf_features.shape
#     tfidf_features = tfidf_features.tolist()
#     return tfidf_features
#
#
# def get_contexts(file_records):
#     contexts = []
#     for file_info in file_records:
#         if 'entity' in file_info:
#             # 文本
#             entity_df = file_info['entity']
#             # print len(entity_df)  # 0也没关系，就正好是没加
#             entity_contexts = entity_df['entity_mention_context3']
#             contexts.extend(entity_contexts.tolist())
#         if 'relation' in file_info:
#             # 文本
#             relation_df = file_info['relation']
#             rel_arg1_contexts = relation_df['rel_arg1_context3']
#             rel_arg2_contexts = relation_df['rel_arg2_context3']
#             relation_contexts = []
#             for i in range(len(rel_arg1_contexts)):
#                 # if rel_arg1_contexts[i] == np.nan:  # 寻么填充和这个都不管用。。
#                 #     rel_arg1_contexts[i] = ''
#                 # if rel_arg2_contexts[i] == np.nan:
#                 #     rel_arg2_contexts[i] = ''
#                 context = str(rel_arg1_contexts[i]) + ' ' + str(rel_arg2_contexts[i])
#                 relation_contexts.append(context)
#             contexts.extend(relation_contexts)
#         if 'event' in file_info:
#             # 文本
#             event_df = file_info['event']
#             event_contexts = event_df['trigger_context3']
#             event_texts = event_df['trigger_text']
#             contexts.extend(event_contexts.tolist())
#     return contexts


def gen_general_features(file_records):
    # 按文件顺序合并
    contexts = []
    texts = []
    types = pd.DataFrame(columns=['entity_type', 'entity_mention_noun_type', 'relation_type', 'relation_subtype',
                                  'event_mention_type', 'event_mention_subtype', 'rel_arg1_entity_type',
                                  'rel_arg1_mention_noun_type', 'rel_arg2_entity_type', 'rel_arg2_mention_noun_type'])

    # 未独热前：entity_type, entity_mention_noun_type, relation_type, relation_subtype, event_mention_type,
    # event_mention_subtype
    # rel_arg1_entity_type, rel_arg1_mention_noun_type, rel_arg2_entity_type, rel_arg2_mention_noun_type
    # 这几个如何与entity的共用？
    # 可所有的都加进来
    
    # (暂时只有target、sentence(context)层次，post和file层次待补，可能需中间文件补)

    for file_info in file_records:
        if 'entity' in file_info:
            # 文本
            entity_df = file_info['entity']
            # print len(entity_df)  # 0也没关系，就正好是没加
            entity_contexts = entity_df['entity_mention_context3']
            entity_texts = entity_df['entity_mention_text']
            contexts.extend(entity_contexts.tolist())
            texts.extend(entity_texts.tolist())
            # 类型
            entity_types = entity_df[['entity_type', 'entity_mention_noun_type']].copy()
            entity_types['relation_type'] = ''
            entity_types['relation_subtype'] = ''
            entity_types['event_mention_type'] = ''
            entity_types['event_mention_subtype'] = ''
            entity_types['rel_arg1_entity_type'] = ''
            entity_types['rel_arg1_mention_noun_type'] = ''
            entity_types['rel_arg2_entity_type'] = ''
            entity_types['rel_arg2_mention_noun_type'] = ''
            types = pd.concat([types, entity_types])
        if 'relation' in file_info:
            # 文本
            relation_df = file_info['relation']
            rel_arg1_contexts = relation_df['rel_arg1_context3']
            rel_arg2_contexts = relation_df['rel_arg2_context3']
            relation_contexts = []
            rel_arg1_texts = relation_df['rel_arg1_text']
            rel_arg2_texts = relation_df['rel_arg2_text']
            relation_texts = []
            for i in range(len(rel_arg1_contexts)):
                # if rel_arg1_contexts[i] == np.nan:  # 寻么填充和这个都不管用。。
                #     rel_arg1_contexts[i] = ''
                # if rel_arg2_contexts[i] == np.nan:
                #     rel_arg2_contexts[i] = ''
                context = str(rel_arg1_contexts[i]) + ' ' + str(rel_arg2_contexts[i])
                relation_contexts.append(context)
                text = rel_arg1_texts[i] + ' ' + rel_arg2_texts[i]
                relation_texts.append(text)
            contexts.extend(relation_contexts)
            texts.extend(relation_texts)
            # 类型
            relation_types = relation_df[['relation_type', 'relation_subtype', 'rel_arg1_entity_type',
                                         'rel_arg1_mention_noun_type', 'rel_arg2_mention_noun_type']].copy()
            relation_types['entity_type'] = ''
            relation_types['entity_mention_noun_type'] = ''
            relation_types['event_mention_type'] = ''
            relation_types['event_mention_subtype'] = ''
            types = pd.concat([types, relation_types])
        if 'event' in file_info:
            # 文本
            event_df = file_info['event']
            event_contexts = event_df['trigger_context3']
            event_texts = event_df['trigger_text']
            contexts.extend(event_contexts.tolist())
            texts.extend(event_texts.tolist())
            # 类型
            event_types = event_df[['event_mention_type', 'event_mention_subtype']].copy()
            event_types['entity_type'] = ''
            event_types['entity_mention_noun_type'] = ''
            event_types['relation_type'] = ''
            event_types['relation_subtype'] = ''
            event_types['rel_arg1_entity_type'] = ''
            event_types['rel_arg1_mention_noun_type'] = ''
            event_types['rel_arg2_entity_type'] = ''
            event_types['rel_arg2_mention_noun_type'] = ''
            types = pd.concat([types, event_types])
    # print types
    # print types.values.shape
    types = types.values.tolist()

    # context层次
    # tfidf
    vec = TfidfVectorizer(min_df=1, ngram_range=(1, 2), stop_words='english', max_features=300, binary=True)
    tfidf_features = vec.fit_transform(contexts).toarray()
    tfidf_features = tfidf_features.tolist()
    # 词性计数，情感词计数
    # context_pos_count_list = pos_senti_count(contexts)
    # 词性、情感具体值列表
    # context_pos_senti = pos_senti_list(contexts)

    # target层次
    # 实体类型特征
    type_one_hot = one_hot(types)
    # 词性计数，情感词计数
    # text_pos_count_list = pos_senti_count(texts)
    # 词性、情感具体值列表
    # text_pos_senti = pos_senti_list(texts)

    # 合并所有特征
    features = tfidf_features
    for i in range(len(features)):
        # features[i].extend(context_pos_count_list[i])
        # features[i].extend(text_pos_count_list[i])
        features[i].extend(type_one_hot[i])
        # features[i].extend(context_pos_senti[i])
        # features[i].extend(text_pos_senti[i])

    print len(features[0])

    return features


# 词性计数，情感词计数
def pos_senti_count(texts):
    pos_name = []
    pos_count_list = []
    pos_senti_count_list = []
    neg_senti_count_list = []
    for i in range(len(texts)):
        # 词性
        pos = nltk.pos_tag(nltk.word_tokenize(texts[i]))
        # print pos
        # 一边生成词典，一边生成特征
        pos_count = [0] * len(pos_name)
        pos_senti_count = 0
        neg_senti_count = 0
        for p in pos:
            if p[1] in pos_name:
                pos_count[pos_name.index(p[1])] += 1
            else:
                pos_name.append(p[1])
                pos_count.append(1)
                for j in range(len(pos_count_list)):
                    pos_count_list[j].append(0)
            lemmed = lemma(p[0])
            polarity = sentiment(lemmed)[0]
            # print lemmed, polarity
            # if polarity >= 0.1:  # 这个阈值？？  # 或者直接累值？？
            #     pos_senti_count += 1
            # elif polarity <= -0.1:
            #     neg_senti_count += 1
            if polarity > 0:
                pos_senti_count += polarity
            elif polarity < 0:
                neg_senti_count += polarity
        pos_count_list.append(pos_count)
        pos_senti_count_list.append(pos_senti_count)
        neg_senti_count_list.append(neg_senti_count)
    print len(pos_name), pos_name
    # 加入特征
    for i in range(len(pos_count_list)):
        pos_count_list[i].append(pos_senti_count_list[i])
        pos_count_list[i].append(neg_senti_count_list[i])
        # print pos_senti_count_list[i], neg_senti_count_list[i]

    return pos_count_list


# 词性，情感词直接堆砌
def pos_senti_list(texts):
    # 情感极性，主动性，词性
    pos_catas = []  # 存放类别特征
    senti_list = []  # 情感数值
    reserved_dim = 10  # 统一维数
    for i in range(len(texts)):
        # 词性
        pos = nltk.pos_tag(nltk.word_tokenize(texts[i]))
        length = len(pos)

        # 情感极性，主动性
        polarity = []
        for j in range(length):
            lemmed = lemma(pos[j][0])
            polarity.append(sentiment(lemmed)[0])  # 极性
            polarity.append(sentiment(lemmed)[1])  # 主动性
        # print polarity
        # 统一维数
        while length < reserved_dim:
            pos.append(('', ''))
            polarity.append(0.0)
            polarity.append(0.0)
            length = len(pos)
        if length > reserved_dim:
            pos = pos[:reserved_dim]
            polarity = polarity[:reserved_dim * 2]

        # 数值特征
        senti_list.append(polarity)

        # 词性加入分类特征
        pos_cata = []
        for j in range(reserved_dim):
            pos_cata.append(pos[j][1])
        pos_catas.append(pos_cata)
        # 这里的独热方式好像不太合适，所有不如上面计数方式；但好像也需

    # 独热编码
    pos_one_hot = one_hot(pos_catas)

    # 特征合并
    pos_senti = pos_one_hot
    for i in range(len(pos_senti)):
        pos_senti[i].extend(senti_list[i])

    print len(pos_senti[0])

    return pos_senti


# 独热编码
def one_hot(features_cata):
    features_cata_df = pd.DataFrame(features_cata)
    features_cata_one_hot_df = pd.get_dummies(features_cata_df)
    features_cata_one_hot = features_cata_one_hot_df.values
    # print features_cata_one_hot.shape
    features_cata_one_hot = features_cata_one_hot.tolist()
    return features_cata_one_hot