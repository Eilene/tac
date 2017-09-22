# coding=utf-8

data_prefix = '../../../data/'

# train_data_dir = data_prefix + 'eng/'
train_data_dir = data_prefix + '2016E61/data/'
train_source_dir = train_data_dir + "source/"
train_ere_dir = train_data_dir + "ere/"
train_annotation_dir = train_data_dir + "annotation/"

test_data_dir = data_prefix + '2016E61_test/data/'
test_source_dir = test_data_dir + "source/"
test_ere_dir = test_data_dir + "ere/"

word2vec_model_path = data_prefix + "blog_word2vec_model_100.txt"
weibo_path = data_prefix + 'weibo.label.xml'

st_output_prefix = data_prefix + 'sentiment_chinese/'

train_mid_files_dir = st_output_prefix + 'middle_files_train/'
train_entity_info_dir = train_mid_files_dir + 'entity_info/'
train_relation_info_dir = train_mid_files_dir + 'relation_info/'
train_event_info_dir = train_mid_files_dir + 'event_info/'
train_em_args_dir = train_mid_files_dir + 'em_args/'

test_mid_files_dir = st_output_prefix + 'middle_files_test/'
test_entity_info_dir = test_mid_files_dir + 'entity_info/'
test_relation_info_dir = test_mid_files_dir + 'relation_info/'
test_event_info_dir = test_mid_files_dir + 'event_info/'
test_em_args_dir = test_mid_files_dir + 'em_args/'

dev_predict_dir = st_output_prefix + 'predict_dev/'
dev_y_predict_dir = st_output_prefix + 'y_predict_dev/'
test_predict_dir = st_output_prefix + 'predict_test/'
test_y_predict_dir = st_output_prefix + 'y_predict_test/'
