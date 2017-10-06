# coding=utf-8
import os
import shutil
from read_write_best import *
from constants import *


def merge_sentiment_and_belief_predict(st_dir, be_dir, output_dir):
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    no = 0
    for parent, dirnames, filenames in os.walk(st_dir):
        for filename in filenames:
            print filename
            st_filepath = st_dir + filename  # 先默认两边文件名是都一样多的
            be_filepath = be_dir + filename
            st_file_record_dict = read_st_file(st_filepath)
            be_file_record_dict = read_be_file(be_filepath)
            write_best_file(st_file_record_dict, be_file_record_dict, output_dir+filename, no)
            no += 1

if __name__ == '__main__':
    if os.path.exists(best_output_prefix) is False:
        os.makedirs(best_output_prefix)
    merge_sentiment_and_belief_predict(st_dev_predict_dir, be_dev_predict_dir, best_dev_predict_dir)
