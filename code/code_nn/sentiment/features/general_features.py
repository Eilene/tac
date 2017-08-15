# coding=utf-8

# 矩阵特征：词向量矩阵，其他
# 向量特征：tfidf，词性等类别特征（论文指出似乎不太有用），词向量拼接，情感词计数，目标的各种类型特征
# 在目标、句子、文件等不同层次上提取


# def embedding_matrix_features():
#     return
#
#
# def tfidf_features():
#     return
#
# def embedding_vector_features():
#     return
#
# def sentiment_features():
#     return
#
# def cata_features():  # 包括各方面的需独热的类别特征
#     return

import nltk
from pattern.en import lemma
from pattern.en import sentiment
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


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
            entity_df = file_info['entity']
            # print len(entity_df)  # 0也没关系，就正好是没加
            entity_contexts = entity_df['entity_mention_context']
            entity_texts = entity_df['entity_mention_text']
            contexts.extend(entity_contexts.tolist())
            texts.extend(entity_texts.tolist())

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
            relation_df = file_info['relation']
            rel_arg1_contexts = relation_df['rel_arg1_context']
            rel_arg2_contexts = relation_df['rel_arg2_context']
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

            relation_types = relation_df[['relation_type', 'relation_subtype', 'rel_arg1_entity_type',
                                         'rel_arg1_mention_noun_type', 'rel_arg2_mention_noun_type']].copy()
            relation_types['entity_type'] = ''
            relation_types['entity_mention_noun_type'] = ''
            relation_types['event_mention_type'] = ''
            relation_types['event_mention_subtype'] = ''
            types = pd.concat([types, relation_types])

        if 'event' in file_info:
            event_df = file_info['event']
            event_contexts = event_df['trigger_context']
            event_texts = event_df['trigger_text']
            contexts.extend(event_contexts.tolist())
            texts.extend(event_texts.tolist())

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
    features = tfidf_features

    # 情感极性，主动性，词性，情感词计数
    # reserved_dim = 40  # 统一维数
    # for i in range(len(contexts)):
    #     # 词性
    #     pos = nltk.pos_tag(nltk.word_tokenize(contexts[i]))
    #     length = len(pos)
    #
    #     # 情感极性，主动性
    #     polarity = []
    #     pos_count = 0  # 情感词计数
    #     neg_count = 0
    #     for j in range(length):
    #         lemmed = lemma(pos[j][0])
    #         polarity.append(sentiment(lemmed)[0])  # 极性
    #         polarity.append(sentiment(lemmed)[1])  # 主动性
    #         if sentiment(lemmed)[0] >= 0.5:
    #             pos_count += 1
    #         if sentiment(lemmed)[0] <= -0.5:
    #             neg_count += 1
    #     # print polarity
    #     # 统一维数
    #     while length < reserved_dim:
    #         pos.append(('', ''))
    #         polarity.append(0.0)
    #         polarity.append(0.0)
    #         length = len(pos)
    #     if length > reserved_dim:
    #         pos = pos[:reserved_dim]
    #         polarity = polarity[:reserved_dim * 2]
    #
    #     # 词性加入分类特征
    #     feature_cata = []
    #     for j in range(reserved_dim):
    #         feature_cata.append(pos[j][1])
    #     features_cata.append(feature_cata)
    #
    #     # 极性等用数值特征
    #     features[i].extend(polarity)
    #     features[i].append(pos_count)
    #     features[i].append(neg_count)
    #
    # # 词向量（之前试是降低了，可再试试）
    #
    #
    # target层次

    # 情感极性，主动性，词性，情感词计数
    features_cata = []  # 存放类别特征
    reserved_dim = 10  # 统一维数
    for i in range(len(texts)):
        # 词性
        pos = nltk.pos_tag(nltk.word_tokenize(texts[i]))
        length = len(pos)

        # 情感极性，主动性
        polarity = []
        pos_count = 0  # 情感词计数
        neg_count = 0
        for j in range(length):
            lemmed = lemma(pos[j][0])
            polarity.append(sentiment(lemmed)[0])  # 极性
            polarity.append(sentiment(lemmed)[1])  # 主动性
            if sentiment(lemmed)[0] >= 0.5:
                pos_count += 1
            if sentiment(lemmed)[0] <= -0.5:
                neg_count += 1
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

        # 词性加入分类特征
        feature_cata = []
        for j in range(reserved_dim):
            feature_cata.append(pos[j][1])
        features_cata.append(feature_cata)
        # 这里的独热方式好像不太合适，是不是会有很多重复的，应该是弄一个大字典，有则1，无则0；再看;但好像也需

        # 极性等用数值特征
        features[i].extend(polarity)
        features[i].append(pos_count)
        features[i].append(neg_count)

    # type加入
    for i in range(len(types)):
        # print features_cata[i]
        features_cata[i].extend(types[i])

    # 独热编码
    features_cata = pd.DataFrame(features_cata)
    features_cata = pd.get_dummies(features_cata)
    features_cata = features_cata.values
    # print features_cata.shape
    features_cata = features_cata.tolist()

    # 合并
    for i in range(len(features)):
        features[i].extend(features_cata[i])
    print 'Feature num and dim:', len(features), len(features[0])

    return features
