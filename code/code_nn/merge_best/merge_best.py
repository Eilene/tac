# coding=utf-8
import os
import shutil
from read_write_best import *
from constants import *


def merge_best(st_dir, be_dir, output_dir):
    print output_dir
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)
    no = 0
    for parent, dirnames, filenames in os.walk(st_dir):
        for filename in filenames:
            # print filename
            st_filepath = st_dir + filename  # 先默认st都有
            be_filepath = be_dir + filename
            st_file_record_dict = read_st_file(st_filepath)
            if os.path.exists(be_filepath):
                be_file_record_dict = read_be_file(be_filepath)
            else:
                be_file_record_dict = None
            write_best_file(st_file_record_dict, be_file_record_dict, output_dir+filename, no)
            no += 1

if __name__ == '__main__':
    # merge_best(dev_eng_df_st_predict_dir, dev_eng_df_be_predict_dir, dev_eng_df_best_predict_dir)
    # merge_best(dev_cmn_df_st_predict_dir, dev_cmn_df_be_predict_dir, dev_cmn_df_best_predict_dir)
    # merge_best(dev_spa_df_st_predict_dir, dev_spa_df_be_predict_dir, dev_spa_df_best_predict_dir)
    #
    # merge_best(dev_eng_nw_st_predict_dir, dev_eng_nw_be_predict_dir, dev_eng_nw_best_predict_dir)
    # merge_best(dev_cmn_nw_st_predict_dir, dev_cmn_nw_be_predict_dir, dev_cmn_nw_best_predict_dir)
    # merge_best(dev_spa_nw_st_predict_dir, dev_spa_nw_be_predict_dir, dev_spa_nw_best_predict_dir)
    #
    # merge_best(test_eng_df_st_predict_dir, test_eng_df_be_predict_dir, test_eng_df_best_predict_dir)
    # merge_best(test_cmn_df_st_predict_dir, test_cmn_df_be_predict_dir, test_cmn_df_best_predict_dir)
    # merge_best(test_spa_df_st_predict_dir, test_spa_df_be_predict_dir, test_spa_df_best_predict_dir)
    #
    # merge_best(test_eng_nw_st_predict_dir, test_eng_nw_be_predict_dir, test_eng_nw_best_predict_dir)
    # merge_best(test_cmn_nw_st_predict_dir, test_cmn_nw_be_predict_dir, test_cmn_nw_best_predict_dir)
    # merge_best(test_spa_nw_st_predict_dir, test_spa_nw_be_predict_dir, test_spa_nw_best_predict_dir)

    merge_best(test_r_eng_df_st_predict_dir, test_r_eng_df_be_predict_dir, test_r_eng_df_best_predict_dir)
    merge_best(test_r_cmn_df_st_predict_dir, test_r_cmn_df_be_predict_dir, test_r_cmn_df_best_predict_dir)
    merge_best(test_r_spa_df_st_predict_dir, test_r_spa_df_be_predict_dir, test_r_spa_df_best_predict_dir)

    merge_best(test_r_eng_nw_st_predict_dir, test_r_eng_nw_be_predict_dir, test_r_eng_nw_best_predict_dir)
    merge_best(test_r_cmn_nw_st_predict_dir, test_r_cmn_nw_be_predict_dir, test_r_cmn_nw_best_predict_dir)
    merge_best(test_r_spa_nw_st_predict_dir, test_r_spa_nw_be_predict_dir, test_r_spa_nw_best_predict_dir)
