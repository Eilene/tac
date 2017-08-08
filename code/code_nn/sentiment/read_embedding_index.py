# coding=utf-8
import numpy as np


def read_embedding_index(filename):
    embeddings_index = {}
    dim = 100
    embedding_vectors_fp = open(filename)
    for line in embedding_vectors_fp:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    embedding_vectors_fp.close()
    # print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index, dim
