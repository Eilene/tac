# coding=utf-8

import nltk
from src.spanish_belief.utils.all_utils_package import *

from gensim.models import doc2vec
from gensim.models.doc2vec import Doc2Vec
import multiprocessing


# 得到所有上下文
def get_contexts(file_records):
    contexts = []
    for file_info in file_records:
        if 'entity' in file_info:
            # 文本
            entity_df = file_info['entity']
            # print len(entity_df)  # 0也没关系，就正好是没加
            entity_contexts = entity_df['entity_mention_context5']
            contexts.extend(entity_contexts.tolist())
        if 'relation' in file_info:
            # 文本
            relation_df = file_info['relation']
            rel_arg1_contexts = relation_df['rel_arg1_context5']
            rel_arg2_contexts = relation_df['rel_arg2_context5']
            relation_contexts = []
            for i in range(len(rel_arg1_contexts)):
                # if rel_arg1_contexts[i] == np.nan:  # 寻么填充和这个都不管用。。
                #     rel_arg1_contexts[i] = ''
                # if rel_arg2_contexts[i] == np.nan:
                #     rel_arg2_contexts[i] = ''
                context = str(rel_arg1_contexts[i]) + ' ' + str(rel_arg2_contexts[i])
                relation_contexts.append(context)
            contexts.extend(relation_contexts)
        if 'event' in file_info:
            # 文本
            event_df = file_info['event']
            event_contexts = event_df['trigger_context5']
            contexts.extend(event_contexts.tolist())
    return contexts


# 转成doc2vec输入所需格式
def get_doc2vec_dataform(texts):
    # 分词，变字符串，空格分开
    token_texts = []
    for text in texts:
        words = nltk.word_tokenize(text.decode('utf-8'))
        token_text = ""
        for word in words:
            token_text += word + ' '
        token_text = token_text[:-1]
        token_text += '\n'
        token_texts.append(token_text)
    return token_texts


# 写入文件
def write_doc2vec_input(texts, filename):
    fp = open(filename, 'w')
    for text in texts:
        fp.write(text)
    fp.close()


if __name__ == '__main__':
    print 'Read data...'
    df_file_records, nw_file_records = \
        read_file_info_records(train_ere_dir, train_entity_info_dir, train_relation_info_dir, train_event_info_dir,
                               train_em_args_dir)
    test_df_file_records = \
        read_file_info_records(test_df_ere_dir, test_df_entity_info_dir, test_df_relation_info_dir,
                               test_df_event_info_dir, test_df_em_args_dir, False)
    test_nw_file_records = \
        read_file_info_records(test_nw_ere_dir, test_nw_entity_info_dir, test_nw_relation_info_dir,
                               test_nw_event_info_dir, test_nw_em_args_dir, False)
    file_records = df_file_records + nw_file_records + test_df_file_records + test_nw_file_records
    contexts = get_contexts(file_records)

    print 'Write doctext...'
    texts = get_doc2vec_dataform(contexts)
    write_doc2vec_input(texts, doctext_path)

    print 'Doc2vec...'
    docslist = doc2vec.TaggedLineDocument(doctext_path)
    model = Doc2Vec(docslist, workers=multiprocessing.cpu_count(), min_count=1, size=200)
    model.save(docmodel_path)
    model = Doc2Vec.load(docmodel_path)
    doc2vec_model = model.docvecs
    print doc2vec_model[0]

