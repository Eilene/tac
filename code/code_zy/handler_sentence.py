# coding:utf-8
from config_log import config
import os 
import sys
from os.path import join as fjoin
import logging
import time
import random
import numpy as np
import pandas as pd
import gensim
import tensorflow
startup_time = time.strftime("%Y-%m-%d-%H:%M:%S")
logger = config(__file__)
logger.info("Start: {}".format(startup_time))

reload(sys)
sys.setdefaultencoding('utf8')

class Node:
	ntype = ''; offset = 0; deep = 0
	def __init__(self, t, o, d):
		self.ntype = t; self.offset = o; self.deep = d

def get_sentence(file, pos, start, end, stack, deep):
	sent = ''; offsets = []; save_start = start; save_end = end
	for i in xrange(pos, len(stack)):
		s = stack[i]
		if s.ntype == 'end' and s.deep == deep:
			break
		if s.ntype == 'start' and s.deep == deep+1:
			save_end = s.offset
			sent += file[save_start:save_end] + '\n'
			offsets.append(str(save_start) + ":" + str(save_end))
		if s.ntype == 'end' and s.deep == deep+1:
			save_start = s.offset+1
	save_end = end
	sent += file[save_start:save_end] + '\n'
	offsets.append(str(save_start) + ":" + str(save_end))
	offsets = ' '.join(offsets)
	return {'sent':sent, 'start':start, 'end':end, 'offsets':offsets}

def extract(file, stack, deep):
	sentences = []; start = 0; end = 0; flag = 0; pos = 0
	for i in range(len(stack)):
		s = stack[i]
		if s.deep == deep+1:
			flag = 1
		if s.ntype == 'start' and s.deep == deep:
			start = s.offset; pos = i
		if s.ntype == 'end' and s.deep == deep:
			end = s.offset
			sentences.append(get_sentence(file, pos, start, end, stack, deep))
	if flag == 1:
		sentences.extend(extract(file, stack, deep+1))
	return sentences

def decompose(file):
	sentence_pd = pd.DataFrame()
	stack = []; index = 0; cnt = 0
	while index < len(file):
		flag = 0
		if file[index:index+5] == '<post' or file[index:index+6] == '<quote':
			stack.append(Node('start', index, cnt))
			cnt = cnt + 1
		if file[index:index+7] == '</post>':
			cnt = cnt - 1
			stack.append(Node('end', index+7, cnt))
		if file[index:index+8] == '</quote>':
			cnt = cnt - 1
			stack.append(Node('end', index+8, cnt))
		index = index + 1
	#for s in stack:
	#	print s.ntype, s.offset, s.deep
	sentences = extract(file, stack, 0)
	for s in sentences:
		sentence_pd = sentence_pd.append(s, ignore_index=True)
	#logger.info("{} {}".format(len(sentences), len(stack)/2))
	logger.info(sentence_pd.columns)
	#logger.info(sentence_pd)
	return sentence_pd

def extract_sentence(fdir, sdir):
	for parents, dirnames, filenames in os.walk(fdir):
		for filename in filenames:
			#if not filename == '1d16a571f14fb1032bc19e9314a46deb.cmp.txt':
			#	continue
			logger.info(filename)
			save_file = fjoin(sdir, filename)
			with open(fjoin(parents, filename)) as infile:
				file = [f.decode("utf-8") for f in infile.readlines()]
			file = decompose(''.join(file))
			file.to_csv(save_file)

def find_sentence(annotation, sentences):
	sent = ''; label = None
	for index, row in sentences.iterrows():
		offsets = row.offsets.split(' ')
		for offset in offsets:
			pos = offset.split(":")
			if annotation.target_offset <= int(pos[1]) and \
				annotation.target_offset >= int(pos[0]):
				sent = row.sent
				break 
	if annotation.polarity == 'pos':
		label = 1
	elif annotation.polarity == 'neg':
		label = 0
	return sent, annotation.polarity

def generate_train_dataset(annotation_dir, sentence_dir):
	train_pd = pd.DataFrame()
	for parents, dirnames, filenames in os.walk(sentence_dir):
		for filename in filenames:
			annotation = fjoin(annotation_dir, filename[:-len('.cmp.txt')]+'.best.xml')
			print annotation
			annotation_pd = pd.read_csv(annotation)
			sentences_pd = pd.read_csv(fjoin(parents, filename))
			for index, row in annotation_pd.iterrows():
				sent, label = find_sentence(row, sentences_pd)
				train_pd = train_pd.append({'sent':sent, 'label':label}, ignore_index=True)
	logger.info(train_pd.columns)
	return train_pd

if __name__ == '__main__':
	root = "/home/apple/best/data"
	fdir = fjoin(root, "source")
	sentence_dir = fjoin(root, "source_sentence")
	if not os.path.isdir(sentence_dir):
		os.makedirs(sentence_dir)
	#extract_sentence(fdir, sentence_dir)

	#annotation_dir = fjoin(root, 'parse_annotation')
	#train_pd = generate_train_dataset(annotation_dir, sentence_dir)
	#train_pd.to_csv(fjoin(root, 'train_all'))
	train_pd = pd.read_csv(fjoin(root, 'train_all'))
	logger.info(train_pd.shape)
	logger.info(train_pd.head())