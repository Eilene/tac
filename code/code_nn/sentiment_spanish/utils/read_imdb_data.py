# coding=utf-8

import os
from constants import imdb_dir


# 读外部数据，imdb
def read_imdb_data(imdb_dir):
    texts = []
    labels = []

    for dtype in ['train', 'test']:
        for polarity in ['pos', 'neg']:
            imdb_file_dir = imdb_dir + dtype + '/' + polarity
            print imdb_file_dir
            for parent, dirnames, filenames in os.walk(imdb_file_dir):
                for filename in filenames:
                    filepath = parent + '/' + filename
                    # print filepath
                    fp = open(filepath, 'r')
                    text = fp.read().decode("utf-8")
                    texts.append(text)
                    if polarity == 'pos':
                        labels.append(1)  # 这里暂且用0,1
                    else:
                        labels.append(0)

    return texts, labels


if __name__ == '__main__':
    texts, labels = read_imdb_data('../'+imdb_dir)
    # print texts
    print labels
    print len(labels)
