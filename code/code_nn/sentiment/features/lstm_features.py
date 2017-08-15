# coding=utf-8
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import numpy as np

MAX_NB_WORDS = 20000
EMBEDDING_DIM = 100
MAX_SEQUENCE_LENGTH = 1000


def gen_lstm_features_by_contexts(contexts, embeddings_index):
    tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
    tokenizer.fit_on_texts(contexts)
    sequences = tokenizer.texts_to_sequences(contexts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))

    nb_words = min(MAX_NB_WORDS, len(word_index))  # 20000
    embedding_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))

    for word, i in word_index.items():
        if i > MAX_NB_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector

    # print(embedding_matrix.shape)

    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    return data, embedding_matrix, nb_words


def gen_lstm_features(file_records, embeddings_index):
    # 特征
    # 按文件顺序合并
    contexts = []
    for file_info in file_records:
        if 'entity' in file_info:
            entity_df = file_info['entity']
            entity_contexts = entity_df['entity_mention_context']
            contexts.extend(entity_contexts.tolist())
        if 'relation' in file_info:
            relation_df = file_info['relation']
            rel_arg1_contexts = relation_df['rel_arg1_context']
            rel_arg2_contexts = relation_df['rel_arg2_context']
            relation_contexts = []
            for i in range(len(rel_arg1_contexts)):
                context = str(rel_arg1_contexts[i]) + ' ' + str(rel_arg2_contexts[i])
                relation_contexts.append(context)
            contexts.extend(relation_contexts)
        if 'event' in file_info:
            event_df = file_info['event']
            event_contexts = event_df['trigger_context']
            contexts.extend(event_contexts.tolist())

    # 用上下文提取特征
    features, embedding_matrix, nb_words = gen_lstm_features_by_contexts(contexts, embeddings_index)

    return features, embedding_matrix, nb_words