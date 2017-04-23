# -*- coding: UTF-8 -*-
from xml.dom.minidom import parse
import xml.dom.minidom
from getSource import get_Source


def ereParse(str,f,target_dic):
    # 获取可能的source
    quote_author, post_author, source_begin = get_Source(f)

    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(str)
    collection = DOMTree.documentElement
    entities = collection.getElementsByTagName("entity")

    for entity in entities:
        entity_id = entity.getAttribute("id")
        entity_type = entity.getAttribute("type")
        entity_specificity = entity.getAttribute("specificity")
        target_dic["entity"][entity_id,entity_type,entity_specificity] = []
        entity_mentions = entity.getElementsByTagName("entity_mention")
        for one_mention in entity_mentions:
            mention_text = one_mention.getElementsByTagName("mention_text")[0].childNodes[0].data
            mention_id = one_mention.getAttribute("id")
            mention_type = one_mention.getAttribute("noun_type")
            mention_offset = one_mention.getAttribute("offset")
            mention_length = one_mention.getAttribute("length")

            if (int(mention_offset) in source_begin and mention_text == source_begin[int(mention_offset)][0]):
                source_begin[int(mention_offset)] += [mention_id, mention_length]

            target_dic["entity"][entity_id, entity_type,entity_specificity]\
                .append([mention_text,mention_id,mention_type,mention_offset,mention_length])

    return target_dic,quote_author, post_author, source_begin