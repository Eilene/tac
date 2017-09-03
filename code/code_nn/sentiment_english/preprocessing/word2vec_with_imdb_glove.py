# coding=utf-8

# 用glove词向量作为初始输入:???
# 用imdb作为训练语料
# 得到新的词向量输出为文件

from src.sentiment_english.utils.read_imdb_data import read_imdb_data
from src.sentiment_english.utils.constants import *

from gensim.models.word2vec import Word2Vec, LineSentence
import multiprocessing
import nltk

if __name__ == '__main__':
    # 分词，每行一个文章，写出去，再训练
    texts, labels = read_imdb_data(imdb_dir)
    tokenized_texts = []
    # fp = open('imdb_word2vec_input.txt', 'w', encoding='utf-8')
    for text in texts:
        words = nltk.word_tokenize(text)
        # for w in words:
        #     fp.write(w)
        #     fp.write(' ')
        # fp.write('\n')
        tokenized_texts.append(words)
    # fp.close()

    # word2vec训练
    # inp = 'imdb_word2vec_input.txt'
    model = Word2Vec(tokenized_texts, size=100, window=5, min_count=5, workers=multiprocessing.cpu_count())
    outp1 = data_prefix+'imdb.model'
    outp2 = data_prefix+'imdb.vector'
    model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)

    # 导入模型
    model = Word2Vec.load(data_prefix+"imdb.model")
    print model['me']

