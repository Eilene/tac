# coding:utf-8
import re
import sys
import os
import pandas as pd 
import xml.dom.minidom

reload(sys)
sys.setdefaultencoding('utf8')

root = '/home/apple/best/data/'

def parse_entity(parent, filename):
	entity_pd = pd.DataFrame({})
	DOMTree = xml.dom.minidom.parse(os.path.join(parent, filename))
	deft_ere = DOMTree.documentElement
	entities = deft_ere.getElementsByTagName("entities")
	for entity in entities:
		entity_mentions = entity.getElementsByTagName("entity_mention")
		for entity_mention in entity_mentions:
			entity_mention_dict = {}
			if entity_mention.hasAttribute("id"):
				entity_mention_dict["id"] = entity_mention.getAttribute("id")
				entity_mention_dict["offset"] = int(entity_mention.getAttribute("offset"))
				entity_mention_dict["noun_type"] = entity_mention.getAttribute("noun_type")
				entity_mention_dict["filesource"] = entity_mention.getAttribute("source")
				entity_mention_dict["length"] = int(entity_mention.getAttribute("length"))
				entity_mention_dict["mention_text"] = entity_mention.getElementsByTagName("mention_text")[0].childNodes[0].data
				entity_pd = entity_pd.append(entity_mention_dict, ignore_index=True)
			else:
				print "no mention id"
	print "entity_pd.shape: ", entity_pd.shape
	entity_pd.offset = entity_pd.offset.astype(int)
	entity_pd.length = entity_pd.length.astype(int)
	return entity_pd

def get_source(parent, source):
	text = ""
	with open(os.path.join(parent, source) + ".cmp.txt", 'r') as handler:
		while True:
			lines = handler.readline().decode("utf-8")
			text += lines
			if not lines:
				break
	return text

def match(text, offset, length, id):
	if text[offset-len('<quote orig_author="'):offset] == '<quote orig_author="':
		print id, '   entity in xml: <quote orig_author="', text[offset:offset+length]
		return None, None
	if text[offset-len('<post author="'):offset] == '<post author="':
		print id, '   entity in xml: <post author="', text[offset:offset+length]
		return None, None
	regex_author = re.compile(r' author="(.*?)"')
	regex_orig = re.compile(r' orig_author="(.*?)"')
	index = offset; stack = 0
	while index >= 5:
		# '<quote'
		if index >= 6 and text[index-6:index] == "<quote":
			if stack > 0:
				stack = stack - 1
			else:
				if text[index] == ">":
					print id, "   <quote>", text[offset:offset+length]
					return None, None
				print "<quote> : ", regex_orig.search(text[index:index+100]).group()[14:-1]
				return index+14, regex_orig.search(text[index:index+100]).group()[14:-1]
		# '</quote>'
		if index >= 8 and text[index-8:index] == "</quote>":
			stack = stack + 1
		# '<post'
		if text[index-5:index] == "<post":
			print "<post> : ", regex_author.search(text[index:index+100]).group()[9:-1]
			return index+9, regex_author.search(text[index:index+100]).group()[9:-1]
		index = index - 1

def backspace(rootdir, sourcefile, entities):
	parent = os.path.join(rootdir, "source")
	text = get_source(parent, sourcefile)
	source_pd = pd.DataFrame()
	
	for index, entity in entities.iterrows():
		if entity.filesource != sourcefile:
			print "not in origin file"
		if text[entity.offset:(entity.offset+entity.length)] != entity.mention_text:
			print entity.mention_text
		source_offset, source = match(text, entity.offset, entity.length, entity.id)
		source_pd = source_pd.append({'source':source, 'id':entity.id, 'source_offset':source_offset}, ignore_index=True)
	entities = pd.merge(entities, source_pd, on='id')
	print "entities.shape: {}, source_pd.shape: {}".format(entities.shape, source_pd.shape)
	return entities
	

def find_source(rootdir):
	ere_rootdir = os.path.join(root, "ere")
	save_dir = os.path.join(root, "source_target")
	if not os.path.isdir(save_dir):
		os.makedirs(save_dir)
	for parent, dirnames, filenames in os.walk(ere_rootdir):
		for filename in filenames:
			print "\nfilename is :" + filename
			entity_pd = backspace( rootdir, filename[:-len(".rich_ere.xml")], parse_entity(parent, filename) )
			entity_pd.to_csv(os.path.join(save_dir, filename))		
			
def extract_source_id(rootdir):
	annotation_dir = os.path.join(rootdir, "parse_annotation")
	my_source_dir = os.path.join(rootdir, "source_target")
	for parent, dirnames, filenames in os.walk(my_source_dir):
		for filename in filenames:
			print "filename: ", filename
			my_source = pd.read_csv(os.path.join(parent, filename))
			annotation = pd.read_csv(os.path.join(annotation_dir, filename[:-len(".rich_ere.xml")]+".best.xml"))
			annotation_true = annotation[annotation.source_id.isnull() == False]
			source_dict = {}
			for index, row in annotation_true.iterrows():
				source_dict[row.source_offset] = row.source_id
			my_source_id_pd = pd.DataFrame()
			for index, row in my_source.iterrows():
				if row.source_offset in source_dict:
					#row['source_id'] = source_dict[row.source_offset]
					my_source_id_pd = my_source_id_pd.append({'id':row.id, 'source_id':source_dict[row.source_offset]}, ignore_index=True)
				else:
					#row['source_id'] = None
					my_source_id_pd = my_source_id_pd.append({'id':row.id, 'source_id':None}, ignore_index=True)
			my_source = pd.merge(my_source, my_source_id_pd, how='left', on=['id'])
			my_source.to_csv(os.path.join(parent, filename))

find_source(root)
extract_source_id(root)

'''
<deft_ere kit_id="55490d330000000000000019" doc_id="0a421343005f3241376fa01e1cb3c6fb" source_type="multi_post">
	<entities>
		<entity id="ent-8" type="PER" specificity="specific">
			<entity_mention id="m-48" noun_type="NAM" source="0a421343005f3241376fa01e1cb3c6fb" offset="14" length="5">
				<mention_text>izzoh</mention_text>
			</entity_mention>
			<entity_mention id="m-54" noun_type="PRO" source="0a421343005f3241376fa01e1cb3c6fb" offset="70" length="1"><mention_text>I</mention_text></entity_mention>

<collection shelf="New Arrivals">
'''