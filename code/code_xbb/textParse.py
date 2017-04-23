# -*- coding:utf-8 -*-
from Constant import *

def getSentence(str,target_dic):
    entity_dic = target_dic["entity"]
    file = open(str,'r')
    source_text = file.read().decode('utf-8')
    file.close()
    for key in entity_dic:
        for val in entity_dic[key]:
            pre_sentence = ''
            now_sentence = val[0]
            next_sentence = ''
            entity_begin = int(val[3])
            entity_end = int(val[3])+int(val[4])
            entity_begin -= 1

            #提取这个偏移位置单词属于的句子,从当前位置往两边找，找到认为是分隔符停止
            i = entity_begin
            while(i >= 0 and source_text[i] not in Seg_sentence):
                now_sentence = source_text[i] + now_sentence
                i -= 1

            # 找到前面一个句子
            if(i < 0):#开头
                pre_sentence = "begin of article"
            else:
                pre_sentence = source_text[i]
                pre_sentence_end = i - 1
                #防止前一个是[\n],连着两[.\n]
                if(pre_sentence_end >= 0 and source_text[pre_sentence_end]  in Seg_sentence):
                    pre_sentence = source_text[pre_sentence_end]
                    pre_sentence_end -= 1
                while (pre_sentence_end >= 0 and source_text[pre_sentence_end] not in Seg_sentence):
                    pre_sentence = source_text[pre_sentence_end] + pre_sentence
                    pre_sentence_end -= 1

            i = entity_end
            while(i<len(source_text) and source_text[i] not in Seg_sentence):
                now_sentence += source_text[i]
                i += 1

            if(i < len(source_text)):
                now_sentence += source_text[i] #补齐后面的标点
                next_sentence_begin = i + 1
                # 找到后一个句子
                if(next_sentence_begin >= len(source_text)):
                    next_sentence = "end of article"
                else:
                    if(source_text[next_sentence_begin] in Seg_sentence):
                        next_sentence_begin += 1
                    if (next_sentence_begin >= len(source_text)):
                        next_sentence = "end of article"
                    else:
                        while (next_sentence_begin < len(source_text) and source_text[next_sentence_begin] not in Seg_sentence):
                            next_sentence += source_text[next_sentence_begin]
                            next_sentence_begin += 1
                        if(next_sentence_begin < len(source_text)):#补齐后面的标点
                            next_sentence += source_text[next_sentence_begin]

            pre_sentence = pre_sentence.strip()
            now_sentence = now_sentence.strip()
            next_sentence = next_sentence.strip()

            val += [pre_sentence,now_sentence,next_sentence]

            #feature = raw_feature.sentence_feature(sentence)

            #提取window_size大小的两侧单词(先不考虑)

    return target_dic