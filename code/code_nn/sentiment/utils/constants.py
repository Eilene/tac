# coding=utf-8

data_prefix = '../../data/'

train_data_dir = data_prefix + '2016E27_V2/data/'
train_source_dir = train_data_dir + "source/"
train_ere_dir = train_data_dir + "ere/"
train_annotation_dir = train_data_dir + "annotation/"

test_data_dir = data_prefix + '2016E27_V2_test/data/'
test_source_dir = test_data_dir + "source/"
test_ere_dir = test_data_dir + "ere/"

glove_100d_path = data_prefix + 'glove.6B/glove.6B.100d.txt'

st_output_prefix = data_prefix + 'sentiment/'

train_mid_files_dir = st_output_prefix + 'train_middle_files/'
train_entity_info_dir = train_mid_files_dir + 'entity_info/'
train_relation_info_dir = train_mid_files_dir + 'relation_info/'
train_event_info_dir = train_mid_files_dir + 'event_info/'
train_em_args_dir = train_mid_files_dir + 'em_args/'

test_mid_files_dir = st_output_prefix + 'test_middle_files/'
test_entity_info_dir = test_mid_files_dir + 'entity_info/'
test_relation_info_dir = test_mid_files_dir + 'relation_info/'
test_event_info_dir = test_mid_files_dir + 'event_info/'
test_em_args_dir = test_mid_files_dir + 'em_args/'

dev_predict_dir = st_output_prefix + 'dev_predict/'
dev_y_predict_dir = st_output_prefix + 'dev_y_predict/'
test_predict_dir = st_output_prefix + 'test_predict/'
test_y_predict_dir = st_output_prefix + 'test_y_predict/'