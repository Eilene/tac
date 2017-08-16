# coding=utf-8

import os
import pandas as pd


def read_file_info_records(ere_dir, entity_info_dir, relation_info_dir, event_info_dir, em_args_dir):
    df_records = []
    nw_records = []

    ere_suffix = ".rich.ere.xml"
    ere_suffix_length = len(ere_suffix)
    for parent, dirnames, ere_filenames in os.walk(ere_dir):
        for ere_filename in ere_filenames:  # 输出文件信息
            part_name = ere_filename[:-ere_suffix_length]
            entity_filepath = entity_info_dir + part_name + '.csv'
            relation_filepath = relation_info_dir + part_name + '.csv'
            event_filepath = event_info_dir + part_name + '.csv'
            em_args_filepath = em_args_dir + part_name + '.csv'
            record = {}
            if os.path.exists(entity_filepath) is True:
                record['filename'] = part_name
                entity_info_df = pd.read_csv(entity_filepath)
                # entity_info_df.fillna(0)  # 没用？？
                record['entity'] = entity_info_df
            if os.path.exists(relation_filepath) is True:
                record['filename'] = part_name
                relation_info_df = pd.read_csv(relation_filepath)
                # relation_info_df.fillna(0)
                record['relation'] = relation_info_df
            if os.path.exists(event_filepath) is True:
                record['filename'] = part_name
                event_info_df = pd.read_csv(event_filepath)
                # event_info_df.fillna(0)
                record['event'] = event_info_df
                if os.path.exists(em_args_filepath) is True:
                    em_args_df = pd.read_csv(em_args_filepath)
                    record['em_args'] = em_args_df
            if record != {}:
                if record['filename'][:35] != 'CMN_DF':  # 论坛数据
                    df_records.append(record)
                else:  # 新闻数据
                    nw_records.append(record)

    return df_records, nw_records
