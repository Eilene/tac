# -*- coding:utf-8 -*-

'''
Tools to transform source data and annotations.

'''

import os
from os.path import join as pjoin
import logging
import time
import argparse
from xml.etree.ElementTree import ElementTree, fromstring, XML
import xml.etree.ElementTree as ET

from file_tools import scan_file_list

startup_time = time.strftime("%Y-%m-%d-%H:%M:%S")
_filename = os.path.splitext(os.path.basename(__file__))[0]
# config logger
logger = logging.getLogger('BEST.{}'.format(_filename))
# 创建一个handler，用于写入日志文件
logger_fh = logging.FileHandler(os.path.join('logs', 'BEST-{}.log'.format(_filename)))
# 再创建一个handler，用于输出到控制台
logger_ch = logging.StreamHandler()
# 定义handler的输出格式formatter
logger_formatter = logging.Formatter('[%(levelname)s] %(asctime)s %(filename)s[line:%(lineno)d]: %(message)s')
logger_fh.setFormatter(logger_formatter)
logger_ch.setFormatter(logger_formatter)
logger.addHandler(logger_fh)
logger.addHandler(logger_ch)
logger.setLevel(logging.DEBUG)
# log_filter = logging.Filter('apple')
# logger.addFilter(log_filter)

logger.info("\n------------------------Start: {} -------------------------".format(startup_time))

source_type=['ANNOTATION', 'AUTHOR', 'NONE']

def writeln(f, out_string, tab_count, tab_as_space=False):
    tab_spaces = 4
    indent_str = " " * tab_spaces * tab_count if tab_as_space else "\t" * tab_count
    f.write(indent_str + out_string.encode('utf8') + "\n")

class SentimentXmlWriter(object):
    '''
    Write labeled sentiment to xml file.
    '''
    def __init__(self, sentiment=None):
        self._ere_id = sentiment['ere_id']
        self._offset = sentiment['offset']
        self._length = sentiment['length']
        self._text = sentiment['text']
        self._polarity = sentiment['polarity']
        self._sarcasm = sentiment['sarcasm']
        self._source = {
            'ere_id': sentiment['source']['ere_id'],
            'offset': sentiment['source']['offset'],
            'length': sentiment['source']['length'],
            'text': sentiment['source']['text'],
            'type': sentiment['source']['type'],
        }
        self._context_sents = sentiment['context_sents']

    def write_xml(self, f, tab_level=0):
        writeln(f, u'<sentiment>', tab_level)
        writeln(f, u'<ere_id>{}</ere_id>'.format(self._ere_id), tab_level+1)
        writeln(f, u'<offset>{}</offset>'.format(self._offset), tab_level+1)
        writeln(f, u'<length>{}</length>'.format(self._length), tab_level+1)
        writeln(f, u'<text>{}</text>'.format(self._text), tab_level+1)
        writeln(f, u'<polarity>{}</polarity>'.format(self._polarity), tab_level+1)
        writeln(f, u'<sarcasm>{}</sarcasm>'.format(self._sarcasm), tab_level+1)
        # write source sub node
        writeln(f, u'<source>', tab_level+1)
        writeln(f, u'<ere_id>{}</ere_id>'.format(self._source['ere_id']), tab_level+2)
        writeln(f, u'<offset>{}</offset>'.format(self._source['offset']), tab_level+2)
        writeln(f, u'<length>{}</length>'.format(self._source['length']), tab_level+2)
        writeln(f, u'<text>{}</text>'.format(self._source['text']), tab_level+2)
        writeln(f, u'<type>{}</type>'.format(self._source['type']), tab_level+2)
        writeln(f, u'</source>', tab_level+1)
        # END write source sub node
        # write context sentences sub node
        writeln(f, u'<context_sentences>', tab_level+1)
        for key,value in self._context_sents.items():
            writeln(f, u'<sentence index="{}" offset="{}" length="{}">{}</sentence>'.format(key,
                value['start_offset'],
                value['end_offset'] - value['start_offset'],
                value['sent_string'].replace("<", "&lt;").replace(">", "&gt;")), tab_level+2)
        writeln(f, u'</context_sentences>', tab_level+1)
        # END write source sub node
        writeln(f, u'</sentiment>', tab_level)

class SentimentCollectionXmlWriter(object):
    '''
    Write labeled sentiments list to xml file.
    '''
    def __init__(self, xml_file='out.xml', sentiments=[]):
        logger.debug("Init SentimenetCollectionXmlWriter with xml_file: {} and {} sentiments".format(
            xml_file, len(sentiments)))
        self._xml_file = xml_file
        self._sentiments = [SentimentXmlWriter(s) for s in sentiments]

    def write_xml(self, f, tab_level=0):
        writeln(f, u'<sentiments>', tab_level)
        for obj in self._sentiments:
            obj.write_xml(f, tab_level + 1)
        writeln(f, u'</sentiments>', tab_level)
def parsing_ere_entities(ere_file):
    etree = ElementTree()
    with open(ere_file, 'r') as f:
        etree.parse(f)
    logger.debug("Loaded ere file: {}".format(ere_file))
    entities = []
    entities_elems = etree.findall('entities')
    for entities_elem in entities_elems:
        entity_elems = entities_elem.findall('entity')
        for entity_elem in entity_elems:
            entity = {
                'id': entity_elem.attrib['id'],
                'type': entity_elem.attrib['type'],
                'specificity': entity_elem.attrib['specificity'] }
            entity_mention_elems = entity_elem.findall('entity_mention')
            mentions = []
            for entity_mention_elem in entity_mention_elems:
                mention = {
                        'id': entity_mention_elem.attrib['id'],
                        'offset': int(entity_mention_elem.attrib['offset']),
                        'length': int(entity_mention_elem.attrib['length']),
                        'noun_type': entity_mention_elem.attrib['noun_type'],
                        'text': entity_mention_elem.find('mention_text').text}
                mentions.append(mention)
            entity['mentions'] = mentions
            entities.append(entity)
    return entities

def parsing_annotation_sentiments(anno_file):
    etree = ElementTree()
    with open(anno_file, 'r') as f:
        etree.parse(f)
    logger.debug("Loaded annotation file: {}".format(anno_file))
    sentiments = []
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
                        sentiment = {'ere_id': entity_elem.attrib['ere_id'],
                                     'offset': int(entity_elem.attrib['offset']),
                                     'length': int(entity_elem.attrib['length']),
                                     'text': entity_elem.find('text').text,
                                     'polarity': inter_sentiment_elem.attrib['polarity'],
                                     'sarcasm': inter_sentiment_elem.attrib['sarcasm'] }
                        source = inter_sentiment_elem.find('source')
                        if source != None:
                            sentiment['source'] = {
                                     'ere_id': source.attrib['ere_id'],
                                     'offset': int(source.attrib['offset']),
                                     'length': int(source.attrib['length']),
                                     'text': source.text,
                                     'type': 'ANNOTATION'}
                        else:
                            sentiment['source'] = {
                                     'ere_id': None,
                                     'offset': None,
                                     'length': None,
                                     'text': None,
                                     'type': 'NONE'}
                        sentiments.append(sentiment)
    return sentiments

def parsing_posts(fstr):
    '''
    parsing all posts node from source data string.
    return {post_string, start_offset, end_offset}
    HINTS: post_string = fstr[start_offset:end_offset]
    '''
    posts_list = []
    test_len = len(fstr) - 5
    post_cnt = 0
    offset = 0
    start_offset = -1
    while offset < test_len:
        if "<post" == fstr[offset:offset+5]:
            if post_cnt == 0:
                start_offset = offset
            post_cnt += 1
            offset += 5
        elif "</post>" == fstr[offset:offset+7]:
            post_cnt -= 1
            if post_cnt == 0:
                end_offset = offset+7
                new_post = {'post_string': fstr[start_offset:end_offset],
                    'start_offset': start_offset,
                    'end_offset': end_offset}
                etree = ET.fromstring(new_post['post_string'].encode('utf-8'))
                new_post['id'] = etree.attrib['id']
                new_post['author'] = etree.attrib['author']
                posts_list.append(new_post)
            offset += 7
        else:
            offset += 1
    
    return posts_list

def parsing_sentences(posts):
    '''
    Parsing all sentences in post.
    return {post_string, start_offset, end_offset,
        sentences:[{sent_string, start_offset, end_offset}]}
    HINTS: post_string = fstr[start_offset:end_offset]
    HINTS: sent_string = fstr[start_offset:end_offset]
    '''
    for post in posts:
        sentences = []
        offset = 0
        fstr = post['post_string']
        end_offset = offset
        while fstr[end_offset] != '>':
            end_offset += 1
        end_offset += 1
        sentences.append({
            'sent_string': fstr[offset:end_offset],
            'start_offset': offset + post['start_offset'],
            'end_offset': end_offset + post['start_offset']})
        offset = end_offset
        test_len = len(fstr) - 8
        while offset < test_len:
            while offset < test_len and fstr[offset].isspace():
                offset += 1
            if offset >= test_len:
                break
            end_offset = offset
            if fstr[end_offset] == '<':
                while end_offset < test_len and fstr[end_offset] != '>':
                    end_offset += 1
                end_offset += 1
                sentences.append({
                    'sent_string': fstr[offset:end_offset],
                    'start_offset': offset + post['start_offset'],
                    'end_offset': end_offset + post['start_offset']})
                
                offset = end_offset
                continue
            q_cnt = 0
            while end_offset < test_len and (fstr[end_offset] not in '.!?' or q_cnt != 0):
                if fstr[end_offset] in '<([':
                    q_cnt += 1
                elif fstr[end_offset] in '>)]':
                    q_cnt -= 1
                end_offset += 1
            end_offset += 1
            sentences.append({
                'sent_string': fstr[offset:end_offset],
                'start_offset': offset + post['start_offset'],
                'end_offset': end_offset + post['start_offset']})
            offset = end_offset
        post['sentences'] = sentences
    return posts

def match_author_entity(entities, posts):
    '''
    match author entity from ere data and source post record
    '''
    for post in posts:
        author_key = 'author="'
        offset = post['post_string'].find(author_key) + len(author_key)
        length = len(post['author'])
        author_str = post['author']
        offset += post['start_offset'] # abstract offset in fstr
        is_find = False
        post['author_ere'] = {}
        post['author_ere']['ere_id'] = None
        post['author_ere']['offset'] = None
        post['author_ere']['length'] = None
        post['author_ere']['text'] = None
        post['author_ere']['type'] = 'AUTHOR'
        for entity in entities:
            for mention in entity['mentions']:
                if author_str.find(mention['text']) > -1\
                        and offset <= mention['offset']\
                        and offset + length >= mention['offset'] + mention['length']:
                    is_find = True
                    post['author_ere']['ere_id'] = mention['id']
                    post['author_ere']['offset'] = mention['offset']
                    post['author_ere']['length'] = mention['length'] 
                    post['author_ere']['text'] = mention['text'] 
                    post['author_ere']['type'] = 'AUTHOR'
                    break
            if is_find:
                break

    return posts

def get_sentiment_context(sentiment, posts,
        prev_words_num=3, next_words_num=3,
        prev_sents_num=1, next_sents_num=1):
    context = {}
    context['author'] = {
             'ere_id': None,
             'offset': None,
             'length': None,
             'text': None,
             'type': 'NONE'}
    context['context_sents'] = {}
    for post in posts:
        if post['start_offset'] <= sentiment['offset']\
                and sentiment['offset'] <= post['end_offset']:
            context['author'] = post['author_ere']
            context_sents = {}
            sentences = post['sentences']
            sent_id = None
            for i in range(len(sentences)):
                sent = sentences[i]
                if sent['start_offset'] <= sentiment['offset']\
                        and sentiment['offset'] + sentiment['length'] <= sent['end_offset']:
                    sent_id = i
                    break
            if None == sent_id:
                logger.warning("NOT match sentiment and sentences!\nsentiment: {}\npost: {}".format(
                    sentiment, post))
                return context
            context_sents[0] = sentences[sent_id]
            for i in range(-1, -prev_sents_num-1, -1):
                if sent_id + i < 0:
                    break
                context_sents[i] = sentences[sent_id+i]
            for i in range(1, next_sents_num + 1, 1):
                if sent_id + i >= len(sentences):
                    break
                context_sents[i] = sentences[sent_id+i]
            context['context_sents'] = context_sents
            break
    return context

def extract_labeled_sentiment_with_context(args):
    '''
    Extract all sentiment and labels with context.
        context: entity,
                 3 words before entity,
                 3 words after entity
                 sentence with entity,
                 sentence before entity,
                 sentence after entity
    '''
    _data_dir = args.data_dir
    # load source file list
    source_file_list = scan_file_list(pjoin(_data_dir, 'source'))

    def post_filter(source_file_list):
        filter_list = []
        for src in source_file_list:
            with open(pjoin(_data_dir, 'source', src), 'r') as f:
                lines = [x.decode('utf-8') for x in f.readlines()]
                source_data = ''.join(lines)
            if os.path.splitext(src)[-1] == ".txt":
                extend_data = "<root>" + source_data + "</root>"
                etree = ET.fromstring(extend_data.encode('utf-8'))
            else:
                etree = ET.fromstring(source_data.encode('utf-8'))
            if etree.attrib.get('type', None) == 'story':
                continue
            post_elems = etree.findall('post')
            if post_elems == None or len(post_elems) == 0:
                continue
            filter_list.append(src)
        logger.info("Post filter {} -> {} files".format(len(source_file_list),
            len(filter_list)))
        return filter_list
    # filter all post source files
    source_file_list = post_filter(source_file_list)

    logger.info("Loaded {} files from {}".format(len(source_file_list),
            pjoin(_data_dir, 'source')))
    # load annotation file list
    anno_file_list = scan_file_list(pjoin(_data_dir, 'annotation'))
    logger.info("Loaded {} files from {}".format(len(anno_file_list),
            pjoin(_data_dir, 'annotation')))
    ere_file_list = scan_file_list(pjoin(_data_dir, 'ere'))
    logger.info("Loaded {} files from {}".format(len(ere_file_list),
            pjoin(_data_dir, 'ere')))

    # match annotation files with source file
    matches = {}
    for s in source_file_list:
        annos = []
        eres = []
        sname,ext = os.path.splitext(s)
        if ext == '.txt':
            sname = os.path.splitext(sname)[0]
        for a in anno_file_list:
            if a.find(sname) == 0:
                annos.append(a)
        for a in annos:
            anno_file_list.remove(a)
        for e in ere_file_list:
            if e.find(sname) == 0:
                eres.append(e)
        for e in eres:
            ere_file_list.remove(e)
        matches[s] = (annos, eres)

    # ready for output folder
    _out_dir = pjoin(_data_dir, 'output')
    if not os.path.exists(_out_dir):
        os.makedirs(_out_dir)
    # load all sentiments from annotations
    for src,(annos, eres) in matches.items():
        sentiments = []
        for a in annos:
            sentiment = parsing_annotation_sentiments(pjoin(_data_dir, 'annotation', a))
            sentiments.extend(sentiment)
        logger.debug("Loaded {} sentiments of src: {}".format(len(sentiments), src))
    
        # parsing entities and mentions from ere file
        entities = []
        for ere in eres:
            entity = parsing_ere_entities(pjoin(_data_dir, 'ere', ere))
            entities.extend(entity)
        logger.debug("Loaded {} entities".format(len(entities)))

        # load whole source file to string
        with open(pjoin(_data_dir, 'source', src), 'r') as f:
            lines = [x.decode('utf-8') for x in f.readlines()]
            fstr = ''.join(lines)

        # parsing source file to posts list with offset[start, end)
        # format as {post_string, start, end}
        offset_posts = parsing_posts(fstr)
        logger.debug("Loaded {} posts".format(len(offset_posts)))
        # parsing sentences with offset in posts
        # format as {sent_string, start, end}
        offset_posts = parsing_sentences(offset_posts)
        # parsing author entity with offset in posts from ere data
        # format as {sent_string, start, end}
        offset_posts = match_author_entity(entities, offset_posts)
        '''
        for post in offset_posts:
            logger.debug("[{}, {})\n{}".format(
                post['start_offset'], post['end_offset'],
                post['post_string'].encode('utf-8')))
            for sent in post['sentences']:
                logger.debug("[{}, {})\n{}".format(
                    sent['start_offset'], sent['end_offset'],
                    sent['sent_string'].encode('utf-8')))
            logger.debug("\nauthor: {} author_ere: {}".format(
                post['author'],
                str(post['author_ere'])))
        '''
        
        for sentiment in sentiments:
            # calcute context for every sentiment
            # context={
            #   post_info: {
            #       pid,
            #       author: {ere_id, offset, length, text},
            #       date_time
            #   },
            #   context_words: {relative_index: word},
            #   context_sents: {relative_index: sent},
            # }
            context = get_sentiment_context(sentiment, offset_posts)

            # use post author ere_id to fill sentiments with source.ere_id=None
            if sentiment['source']['type'] == 'NONE':
                if context['author']['type'] != 'NONE':
                    sentiment['source']['ere_id'] = context['author']['ere_id']
                    sentiment['source']['offset'] = context['author']['offset']
                    sentiment['source']['length'] = context['author']['length']
                    sentiment['source']['text'] = context['author']['text']
                    sentiment['source']['type'] = context['author']['type']
            # add context sentences to sentiment
            sentiment['context_sents'] = context['context_sents']

        # output format info to xml file.
        # open output file
        # filename: <source file name>.xml
        writer = SentimentCollectionXmlWriter(pjoin(_out_dir, "{}.xml".format(src)), sentiments)
        with open(pjoin(_out_dir, "{}.xml".format(src)), 'w') as f:
            tab_level = 0
            writeln(f, u'<?xml version="1.0" encoding="UTF-8"?>', tab_level)
            writer.write_xml(f, tab_level=tab_level)
            f.close()

def parse_args():
    description = ('Tools to transform source data and annotations.')
    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--data_dir',
                        default="../../data/LDC2016E27_DEFT_English_Belief_and_Sentiment_Annotation_V2/data",
                        help='dataset dir')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    extract_labeled_sentiment_with_context(args)

if __name__ == '__main__':
    main()
