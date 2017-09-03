# coding=utf-8

from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec
import multiprocessing
docslist = doc2vec.TaggedLineDocument('test.txt')
model = Doc2Vec(docslist, workers= multiprocessing.cpu_count(),min_count=1,size = 200)
# print model[0]
model.save("doc_model_200.txt")

model = Doc2Vec.load("doc_model_200.txt")
doc2vec_model = model.docvecs
print doc2vec_model[0]
