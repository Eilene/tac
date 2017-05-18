# coding:utf-8

import os
import sys
import logging
import pandas as pd
import numpy as np 
from xml.etree.ElementTree import ElementTree, fromstring, XML
import xml.etree.ElementTree as ET
reload(sys)  

sys.setdefaultencoding('utf8')
_filename = os.path.splitext(os.path.basename(__file__))[0]
root = '/home/apple/best/data'
logger = logging.getLogger('BEST.{}'.format(_filename))

def parse_annotation_sentiments(anno_file):
    etree = ElementTree()
    with open(anno_file, 'r') as f:
        etree.parse(f)
    logger.debug("Loaded annotation file: {}".format(anno_file))
    sentiments = pd.DataFrame()
    sentiments_elems = etree.findall('sentiment_annotations')
    for sentiments_elem in sentiments_elems:
        entities_elems = sentiments_elem.findall('entities')
        for entities_elem in entities_elems:
            entity_elems = entities_elem.findall('entity')
            for entity_elem in entity_elems:
                inter_sentiments_elems = entity_elem.findall('sentiments')
                for inter_sentiments_elem in inter_sentiments_elems:
                    inter_sentiment_elems = inter_sentiments_elem.findall('sentiment')
                    for inter_sentiment_elem in inter_sentiment_elems:
                        sentiment = {'target_id': entity_elem.attrib['ere_id'],
                                     'target_offset': int(entity_elem.attrib['offset']),
                                     'target_length': int(entity_elem.attrib['length']),
                                     'target_text': entity_elem.find('text').text,
                                     'polarity': inter_sentiment_elem.attrib['polarity'],
                                     'sarcasm': inter_sentiment_elem.attrib['sarcasm'],
                                     'source_id': None,
                        			 'source_offset': None,
                        			 'source_length': None,
                        			 'source_text': None,
                        			 'source_type': None }
                        source = inter_sentiment_elem.find('source')
                        if source != None:
	                        sentiment['source_id'] = source.attrib['ere_id'] 
	                        sentiment['source_offset'] = int(source.attrib['offset'])
	                        sentiment['source_length'] = int(source.attrib['length'])
	                        sentiment['source_text'] = source.text
	                        sentiment['source_type'] = 'ANNOTATION'
                        sentiments = sentiments.append(sentiment, ignore_index=True)
    sentiments.target_offset = sentiments.target_offset.astype(int)
    sentiments.target_length = sentiments.target_length.astype(int)
    return sentiments

def annotation(rootdir):
	annotation_rootdir = os.path.join(rootdir, "annotation")
	save_dir = os.path.join(rootdir, 'parse_annotation')
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	for parent, dirnames, filenames in os.walk(annotation_rootdir):
		for filename in filenames:
			print "\nfilename is :" + filename
			parse_annotation_sentiments(os.path.join(annotation_rootdir, filename)).to_csv(os.path.join(save_dir, filename))

def analyse(rootdir):
	annotation_dir = os.path.join(rootdir, 'parse_annotation')
	whole_cnt = 0; annotation_cnt = 0
	for parent, dirnames, filenames in os.walk(annotation_dir):
		for filename in filenames:
			annotation_pd = pd.read_csv(os.path.join(annotation_dir, filename))
			whole_cnt = whole_cnt + annotation_pd.shape[0]
			annotation_true = annotation_pd[annotation_pd.source_id.isnull() == False]
			annotation_cnt = annotation_cnt + annotation_true.shape[0]
	print "whole_cnt:", whole_cnt
	print "annotation_cnt:", annotation_cnt


annotation(root)
analyse(root)

'''
if entity.hasAttribute("ere_id"):
	entity_dict['ere_id'] = entity.getAttribute("ere_id")
	entity_dict['offset'] = entity.getAttribute("offset")
	entity_dict['length'] = entity.getAttribute("length")
	entity_dict['text'] = entity.getElementsByTagName("text")[0].childNodes[0].data
	entity_dict['sentiment'] = entity.getElementsByTagName("sentiments")[0].childNodes[0].data
	entity_mention_dict["mention_text"] = entity_mention.getElementsByTagName("mention_text")[0].childNodes[0].data
	print entity_dict
'''