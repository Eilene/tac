# coding=utf-8

from matrix_features import *


def matrix_to_avg_vector(x_matrix):
    x_matrix = np.array(x_matrix)
    length = x_matrix.shape[-1]
    sample_num = x_matrix.shape[-2]
    # print length, sample_num
    x_vector = []
    for i in x_matrix:
        temp = np.zeros(length)
        for j in i:
            temp = temp + j
        temp /= float(sample_num)
        x_vector.append(list(temp))
    return x_vector


def gen_embeddings_vector_features(context, clip_length, embeddings_index, dim):
    x_matrix = gen_matrix_features(context, clip_length, embeddings_index, dim)
    x_vector = matrix_to_avg_vector(x_matrix)
    return x_vector
