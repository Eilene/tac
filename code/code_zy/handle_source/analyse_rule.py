# coding:utf-8

import os
import sys
import logging
import pandas as pd
import numpy as np 
from xml.etree.ElementTree import ElementTree, fromstring, XML
import xml.etree.ElementTree as ET
from find_source_from_ere import get_source
reload(sys)  

sys.setdefaultencoding('utf8')
_filename = os.path.splitext(os.path.basename(__file__))[0]
root = '/home/apple/best/data'
logger = logging.getLogger('BEST.{}'.format(_filename))

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


def analyse_rule(rootdir):
	annotation_dir = os.path.join(rootdir, 'parse_annotation')
	my_source_dir = os.path.join(root, "source_target")
	source_dir = os.path.join(root, "source")
	rule_cnt = 0
	for parent, dirnames, filenames in os.walk(my_source_dir):
		for filename in filenames:
			#print "filename: ", filename, filename[:-len(".rich_ere.xml")]
			my_source = pd.read_csv(os.path.join(parent, filename))
			annotation = pd.read_csv(os.path.join(annotation_dir, filename[:-len(".rich_ere.xml")]+".best.xml"))
			annotation_true = annotation[annotation.source_id.isnull() == False]
			my_source.rename(columns={'id':'target_id'}, inplace=True)
			annotation_true = pd.merge(annotation_true, my_source, on=['target_id'])
			annotation_false = annotation_true[(annotation_true.source_offset_x != annotation_true.source_offset_y)]
			for index, row in annotation_false.iterrows():
				text = get_source(source_dir, row.filesource)

				if index == 0:
					print "\n", "="*100
					print text
					print "\n", "="*100
				print "\n", "*"*50
				print row.filesource
				print "target_id: {} \ntarget_text: {} \ntarget_offset: {} \nsource_id: {} \nsource_text: {} \nsource_offset: {}".format(row.target_id,\
					 row.target_text, \
					 row.target_offset, \
					 row.source_id, \
					 row.source_text, \
					 row.source_offset_x)
				print "my_source_text: {} \nmy_source_offset: {}".format(row.source, row.source_offset_y)
				print "text:\n{}".format(text[row.target_offset-200:row.target_offset+200])
			rule_cnt = rule_cnt + annotation_false.shape[0]
	print "rule_cnt:", rule_cnt

analyse_rule(root)

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