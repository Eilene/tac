# coding=utf-8


import sys

reload(sys)
sys.setdefaultencoding("utf-8")

import pickle
from gensim.models import Word2Vec
from constants import word2vec_model_path


# 读取词模型
def read_embedding_index(word2vec_model_path):
    model = Word2Vec.load(word2vec_model_path)
    dim = model[u'我'].shape[0]
    return model, dim

if __name__ == '__main__':
    model, dim = read_embedding_index(word2vec_model_path)
    print dim
