# coding=utf-8

# 可试不同数据源构造：
# best数据
# glove词向量
# 其他外部数据

from src.english_sentiment.utils.all_utils_package import *

from pattern.en import lemma
from pattern.en import sentiment

output_filename = st_output_prefix + 'word2sentivec_train_file.txt'


# 写入文件
def write_file(text, filename):
    fp = open(filename, 'w')
    fp.write(text)
    fp.close()


# glove词向量
def with_glove():
    embeddings_index, dim = read_embedding_index(glove_100d_path)
    format_text = ''
    for word in embeddings_index:
        lemmed = lemma(word)
        print lemmed
        polarity = sentiment(lemmed)[0]  # 不行啊，报错
        if polarity >= 0.5:
            label = 2
        elif polarity <= -0.5:
            label = 1
        else:
            label = 0
        format_text += (word + '_' + str(label) + ' ')
        # print word, lemmed, label
    write_file(format_text, output_filename)


# best官方数据
def with_best_data():
    format_text = ''
    word_labels = {}
    # 直接读文章，去掉标签，分词，词性还原，标记，输出
    for parent, dirnames, filenames in os.walk(train_source_dir):
        for filename in filenames:
            # 读文件
            source_fp = open(train_source_dir+filename)
            all_source_text = source_fp.read().decode("utf-8")  # 注意编码
            source_fp.close()
            # 去标签
            re_h = re.compile('</?\w+[^>]*>')  # HTML标签
            source_text = re_h.sub(" ", all_source_text)
            # 分词，词性还原，情感标记
            words = nltk.word_tokenize(source_text)
            print words
            for word in words:
                lemmed = lemma(word)
                polarity = sentiment(lemmed)[0]  # 不行啊，报错
                if polarity >= 0.5:
                    label = 2
                elif polarity <= -0.5:
                    label = 1
                else:
                    label = 0
                word_labels[lemmed] = label
                # print word, lemmed, label
    for word in word_labels:
        format_text += (word + '_' + str(word_labels[word]) + ' ')
    # 不对，搞成一句一行？？
    # 写入文件
    write_file(format_text, output_filename)

if __name__ == '__main__':
    with_best_data()

    # 不对啊，出不来
