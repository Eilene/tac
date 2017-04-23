# -*- coding: UTF-8 -*-
# 处理文章的source，传入文件名,获得post_author和quote,quote纪录text和开始结束位置，
# post_author纪录author的开始位置,source_begin{}纪录所有source的偏移位置和val，为了可以找到这些source的entity_id

def get_Source(str):
    file = open(str,'r')
    source_text = file.read().decode('utf-8')
    file.close()

    source_begin = {}
    post_author = []
    length = len(source_text)
    index = 0
    while(True):

        index = source_text.find("<post author=",index,length)
        if index == -1:
            break
        index = index + 14
        text = ""
        offset = index
        while(source_text[index] != '"'):
            text = text + source_text[index]
            index += 1
        post_author.append([text,offset])
        source_begin[offset] = [text]

    index = 0
    index2 = 0
    quote = []
    quote_author = []
    offset_begin = []
    offset_end = []
    while(True):
        index = source_text.find("<quote",index,length)
        if(source_text[index:index+19] == "<quote orig_author="):
            index = index + 20
            text = ""
            offset1 = index
            while (source_text[index] != '"'):
                text = text + source_text[index]
                index += 1
        else:#没有original author
            index = index + 6
            text = "None"
            offset1 = index

        index2 = source_text.find("</quote>",index2,length)
        if (index2 == -1):
            break
        offset2 = index2
        index2 += 1

        quote_author.append(text)
        offset_begin.append(offset1)
        offset_end.append(offset2)
        source_begin[offset1] = [text]


    num = len(offset_end)
    index = num - 1
    mid_array = []
    j = num - 1
    for i in reversed(offset_begin):
        if(len(mid_array) > 0):
            quote.append([quote_author[j],i,mid_array[-1]])
            j = j - 1
            mid_array = mid_array[:-1]
        else:
            mid_array = []
            while(offset_end[index] >i ):
                mid_array.append(offset_end[index])
                index -= 1
                if(0 - index > num):
                    break
            quote.append([quote_author[j], i, mid_array[-1]])
            j = j-1
            mid_array = mid_array[:-1]
    return quote,post_author,source_begin

    # 要找每一个mention_text对应的source,找本篇文章中，
    # 根据偏移地址，先看是不是属于quot的某一区间，否则给对应的post，

def find_Source(target_dic, quote_author, post_author, source_begin):
    for i in target_dic["entity"]:
        for j in target_dic["entity"][i]:
            offset = int(j[3])
            flag = False  # 用来标记是否属于quote
            for q in quote_author:
                if (offset > q[2]):
                    break
                if (int(offset) >= q[1] and offset <= q[2]):
                    flag = True
                    break
            if (flag == True):
                j += source_begin[q[1]]
                j += [q[1]]
            else:  # source 为post用户
                index = -1
                while (True):
                    if (offset >= post_author[index][1]):
                        j += source_begin[post_author[index][1]]
                        j += [post_author[index][1]]
                        break
                    index -= 1
    return target_dic
