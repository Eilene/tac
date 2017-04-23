# -*- coding:UTF-8 -*-
import logging
import os
import pandas as pd
from Constant import *
from getSource import find_Source
from xmlParse import ereParse
from textParse import getSentence
from best_xmlParse import getAnnotation

if __name__ == "__main__":
    # 通过下面的方式配置输出方式与日志级别
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                        datefmt='%a, %d %b %Y %H:%M:%S',
                        filename='sentiment_belief.log',
                        filemode='w')

    logging.info('*********run begin***********')
    logging.info("*********handle every file**********")
    target_dic = {}
    raw_info = []
    for f in os.listdir(RootDir):
        if(f.split(".")[-1] == 'txt'):#先只处理dw数据

            target_dic["entity"] = {}#不同文件有公用id ,则每次重新初始化
            str = f.split(".")[0]
            f = Source_path + f
            logging.info("*********handle file " + f + " **********")
            #取出所有的ere
            logging.info("*********get ere**********")
            str_temp = Ere_path + str +".rich_ere.xml"
            target_dic, quote_author, post_author, source_begin = ereParse(str_temp,f,target_dic)

            #为ere找到对应的计算出来的source
            logging.info("*********compute source**********")
            target_dic = find_Source(target_dic, quote_author, post_author, source_begin)

            #找到每个ere_mention对应的句子
            logging.info("*********get sentence and annotation**********")
            target_dic = getSentence(f,target_dic)

            #找到每个ere_mention对应的真正的标注结果，包括source 和情感
            str_temp =  Annotation_path + str + ".best.xml"
            annation = getAnnotation(str_temp,target_dic)

            for i in  target_dic["entity"]:
                for j in target_dic["entity"][i]:
                    raw_info.append(list(i)+j)

    df = pd.DataFrame(raw_info)
    df.columns = ["entity_id","entity_type","entity_specificity","mention_text",
                  "mention_id","mention_type","mention_offset","mention_length",
                  "source_text", "source_id", "source_length", "source_offset",
                  "pre_sentence","now_sentence","next_sentence",
                  "label_source_text","label_source_id","label_source_length",
                  "label_source_offset","label_polarity", "label_sarcasm"]
    df.to_csv(Write_path,index = False,encoding='utf-8')




