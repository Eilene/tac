# coding=utf-8

from matrix_features import *


def matrix_to_concat_vector(x_matrixs):
    x_vectors = []
    for x_mat in x_matrixs:
        x_vector = []
        for vec in x_mat:
            x_vector.extend(vec)
        x_vectors.append(x_vector)
    return x_vectors


def gen_concat_embedding_features(contexts, clip_length, embeddings_index, dim):
    x_matrixs = gen_matrix_features(contexts, clip_length, embeddings_index, dim)
    x_vectors = matrix_to_concat_vector(x_matrixs)
    print len(x_vectors[0])
    return x_vectors
