# coding:utf-8
from config_log import config
import os 
from os.path import join as fjoin
import logging
import time
import random
import numpy as np
import pandas as pd
import gensim
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import tensorflow
startup_time = time.strftime("%Y-%m-%d-%H:%M:%S")

logger = config(__file__)
logger.info("\n------------------------Start: {} -------------------------".format(startup_time))

# transform the dataset into txt file
def handle_imdb_dataset(sentiment):
	print "in"
	data_type = ['train', 'test']
	save_file = fjoin("/home/apple/best/external_data/aclImdb", sentiment + ".txt")
	save_file_handler = open(save_file, 'w+')
	reviews = []
	for dt in data_type:
		imdb_dir = fjoin("/home/apple/best/external_data/aclImdb", dt+"/"+sentiment)
		print imdb_dir
		for parent, dirnames, filenames in os.walk(imdb_dir):
			print len(filenames)
			for filename in filenames:
				print filename
				with open(fjoin(parent, filename), 'r') as infile:
					reviews.append("\n".join(infile.readlines())+"\n")
	save_file_handler.writelines(reviews)
	save_file_handler.close()
	#/home/apple/best/external_data/aclImdb/train_pos.txt
'''
handle_imdb_dataset('pos')
handle_imdb_dataset('neg')
handle_imdb_dataset('unsup')
'''

external_data = "/home/apple/best/external_data/aclImdb/"

TaggedDocument = gensim.models.doc2vec.TaggedDocument

def cleanText(corpus):
	punctuation = """.,?!:;(){}[]"""
	corpus = [z.lower().replace('\n','') for z in corpus]
	corpus = [z.replace('<br />', ' ') for z in corpus]
	for c in punctuation:
		corpus = [z.replace(c, ' %s '%c) for z in corpus]
	corpus = [z.split() for z in corpus]
	return corpus

# gensim doc2vec: docment/sentence need a unique label
def labelizeReviews(reviews, label_type):
	labelized = []
	for i, v in enumerate(reviews):
		label = '%s_%s'  % (label_type, i)
		labelized.append(TaggedDocument(v, [label]))
	return labelized

def get_dataset():
	logger.info('Get_dataset')
	train_all = pd.read_csv('/home/apple/best/data/train_all')
	pos = train_all[train_all.label == 'pos']
	neg = train_all[train_all.label == 'neg']
	none = train_all[train_all.label == 'none']
	logger.info('pos:{} neg:{} none:{}'.format(pos.shape, neg.shape, none.shape))
	y = np.concatenate((np.ones(pos.shape[0]), np.zeros(neg.shape[0])))
	x = np.concatenate((pos.sent, neg.sent))
	x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

	x_train = cleanText(x_train)
	x_test = cleanText(x_test)
	pos = cleanText(pos.sent)
	neg = cleanText(neg.sent)
	x_train = labelizeReviews(x_train, 'TRAIN')
	x_test = labelizeReviews(x_test, 'TEST')
	pos = labelizeReviews(pos, 'POS')
	neg = labelizeReviews(neg, 'NEG')
	return x_train, x_test, y_train, y_test, pos, neg


def get_external_dataset():
	logger.info("Get_external_dataset")
	with open(fjoin(external_data, 'pos.txt'), 'r') as infile:
		pos_reviews = infile.readlines()

	with open(fjoin(external_data, 'neg.txt'), 'r') as infile:
		neg_reviews = infile.readlines()

	with open(fjoin(external_data, 'unsup.txt'), 'r') as infile:
		unsup_reviews = infile.readlines()

	logger.info("pos:{} neg:{} unsup:{}".format(len(pos_reviews), len(neg_reviews), len(unsup_reviews)))
	y = np.concatenate((np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))))

	x_train, x_test, y_train, y_test = train_test_split(np.concatenate((pos_reviews, neg_reviews)), y, test_size=0.2)

	x_train = cleanText(x_train)
	x_test = cleanText(x_test)
	unsup_reviews = cleanText(unsup_reviews)
	
	x_train = labelizeReviews(x_train, 'EXTERNAL_TRAIN')
	x_test = labelizeReviews(x_test, 'EXTENAL_TEST')
	unsup_reviews = labelizeReviews(unsup_reviews, 'EXTENAL_UNSUP')
	return x_train, x_test, unsup_reviews, y_train, y_test

# read doc vec
def getVecs(model, corpus, size):
	vecs = [np.array(model.docvecs[z.tags[0]]).reshape((1,size)) for z in corpus]
	return np.concatenate(vecs)

# train dataset
def train(pos, neg, x_train, x_test, external_x_train, external_x_test,external_unsup_reviews, size=400, epoch_num=10):
	logger.info("Train sentence model(dm, dbow)")
	model_dm = gensim.models.Doc2Vec(min_count=1, \
		window=10, \
		size=size, \
		sample=1e-3, \
		negative=5, \
		workers=6)
	model_dbow = gensim.models.Doc2Vec(min_count=1, \
		window=10, \
		size=size, \
		sample=1e-3, \
		negative=5, \
		dm=0, \
		workers=6)
	# use all words build vocab
	vocab_document = x_train + x_test + external_x_train + external_x_test + external_unsup_reviews
	model_dm.build_vocab(vocab_document)
	model_dbow.build_vocab(vocab_document)

	# repeat train, everytime break the sequence to improve accuracy
	tmp_x_train = x_train + x_test + external_x_train + external_x_test
	print tmp_x_train[1:2]
	for epoch in range(epoch_num):
		logger.info("train epoch {}".format(epoch))
		random.shuffle(tmp_x_train)
		model_dm.train(tmp_x_train)
		model_dbow.train(tmp_x_train)

	# train test dataset
	'''
	tmp_x_test = x_test
	for epoch in range(epoch_num):
		logger.info("test epoch {}".format(epoch))
		random.shuffle(tmp_x_test)
		model_dm.train(tmp_x_test)
		model_dbow.train(tmp_x_test)
	'''
	model_dm.save(fjoin(model_dir, 'doc2vec_dm'))
	model_dbow.save(fjoin(model_dir, 'doc2vec_dbow'))
	return model_dm, model_dbow

def get_vectors(model_dm, model_dbow):
	logger.info("Sentence to vector")
	train_vecs_dm = getVecs(model_dm, x_train, size)
	train_vecs_dbow = getVecs(model_dbow, x_train, size)
	train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))

	test_vecs_dm = getVecs(model_dm, x_test, size)
	test_vecs_dbow = getVecs(model_dbow, x_test, size)
	test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

	return train_vecs, test_vecs

def Classifier(train_vecs, y_train, test_vecs, y_test):
	logger.info("Train model to predict")
	print train_vecs.shape
	print train_vecs
	
	from sklearn.linear_model import SGDClassifier
	clf = SGDClassifier(loss='log', penalty='l1')
	
	'''
	from sklearn.linear_model import LogisticRegression
	clf = LogisticRegression()
	
	from sklearn.svm import SVC
	clf = SVC(kernel='linear')
	'''
	clf.fit(train_vecs, y_train)
	score = clf.score(test_vecs, y_test)
	logger.info("Test Accuracy: %.3f"%score)
	report = score > 0.5
	#print classification_report(y_test, report, target_names=['neg', 'pos'])
	return clf

def ROC_curve(lr, y_test):
	logger.info("Plot roc curve")
	from sklearn.metrics import roc_curve, auc
	import matplotlib.pyplot as plt
	pred_probas = lr.predict_proba(test_vecs)[:,1]
	fpr, tpr, _ = roc_curve(y_test, pred_probas)
	roc_auc = auc(fpr, tpr)
	plt.plot(fpr, tpr, label='area=%.2f'%roc_auc)
	plt.plot([0,1], [0,1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])

	plt.show()


if __name__ == '__main__':
	size, epoch_num = 400, 10
	model_dir = '/home/apple/best/model'

	x_train, x_test, y_train, y_test, pos, neg = get_dataset()
	external_x_train, external_x_test, external_unsup_reviews, external_y_train, external_y_test = get_external_dataset()
	model_dm, model_dbow = train(pos, neg, x_train, x_test, external_x_train, external_x_test,external_unsup_reviews, size, epoch_num)
	model_dm = gensim.models.Doc2Vec.load(fjoin(model_dir, 'doc2vec_dm'))
	model_dbow = gensim.models.Doc2Vec.load(fjoin(model_dir, 'doc2vec_dbow'))
	train_vecs, test_vecs = get_vectors(model_dm, model_dbow)
	lr = Classifier(train_vecs, y_train, test_vecs, y_test)
	ROC_curve(lr, y_test)
