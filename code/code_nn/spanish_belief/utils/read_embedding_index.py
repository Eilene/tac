# coding=utf-8
import numpy as np


def read_embedding_index(filename):
    embeddings_index = {}
    embedding_vectors_fp = open(filename)
    for line in embedding_vectors_fp:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        # print coefs.shape
        embeddings_index[word] = coefs
        dim = coefs.shape[0]
    embedding_vectors_fp.close()
    # print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index, dim

if __name__ == '__main__':
    from constants import glove_100d_path
    embeddings_index, dim = read_embedding_index('../'+glove_100d_path)
    spanish_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '"', "'"]
    for word in embeddings_index.keys():
        if word in spanish_punctuations:  # 竟然有标点！！上下文应该没去标点，只有窗口文本去了标点，应该可以表达感情
            print word

