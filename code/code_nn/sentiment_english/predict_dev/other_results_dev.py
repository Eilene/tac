# coding=utf-8

from src.sentiment_english.utils.constants import train_ere_dir, data_prefix, train_source_dir
from src.sentiment_english.utils.read_file_info_records import read_file_info_records
from src.sentiment_english.utils.find_source import find_sources
from src.sentiment_english.utils.file_records_other_modification import to_dict
from src.sentiment_english.utils.write_best import write_best_files
from src.sentiment_english.utils.attach_predict_labels import set_neg


entity_info_dir = data_prefix + 'otherrs/middle2/middle_file_2017-09-01 15-12-29/entity_info/'
relation_info_dir = data_prefix + 'otherrs/middle2/middle_file_2017-09-01 15-12-29/relation_info/'
event_info_dir = data_prefix + 'otherrs/middle2/middle_file_2017-09-01 15-12-29/event_info/'
predict_dir = data_prefix + 'otherrs/predict/'


if __name__ == '__main__':
    df_file_records, nw_file_records = \
        read_file_info_records(train_ere_dir, entity_info_dir, relation_info_dir, event_info_dir, '')

    file_records = df_file_records + nw_file_records

    # set_neg(file_records)

    to_dict(file_records)

    find_sources(file_records, train_source_dir, train_ere_dir)

    write_best_files(file_records, predict_dir)
