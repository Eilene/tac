# -*- coding:utf-8 -*-
#解析标注文件，得到{target_id:source_id,source_offset,source_length,source_text,polarity,sarcasm}
#(因为target的信息可以找到，但是source可能给错，所以要记录下来标注文件中的，现在是假定了一个target_mention_id只在标注的sentiment中出现一次)

from xml.dom.minidom import parse
import xml.dom.minidom

def getAnnotation(str,target_dic):
    # 使用minidom解析器打开 XML 文档
    DOMTree = xml.dom.minidom.parse(str)
    collection = DOMTree.documentElement

    annotation = {}
    annotation["entity"] = {}
    sentiment_annotations = collection.getElementsByTagName("sentiment_annotations")
    entities = sentiment_annotations[0].getElementsByTagName("entity")

    for entity in entities:
        target_id = entity.getAttribute("ere_id")
        sentiments = entity.getElementsByTagName("sentiments")
        for sentiment in sentiments:
            polarity = sentiment.childNodes[1].getAttribute("polarity")
            sarcasm = sentiment.childNodes[1].getAttribute("sarcasm")

            sources = sentiment.getElementsByTagName("source")

            for source in sources:
                source_id = source.getAttribute("ere_id")
                source_offset = source.getAttribute("offset")
                source_length = source.getAttribute("length")
                source_text = source.childNodes[0].data
                annotation["entity"][target_id] = [source_text,source_id,source_length,source_offset,polarity,sarcasm]

    for i in target_dic["entity"]:
        for j in target_dic["entity"][i]:
            if (j[1] in annotation["entity"]):
                j += annotation["entity"][j[1]]
            else:
                j += ["None"] * 6

    return target_dic



