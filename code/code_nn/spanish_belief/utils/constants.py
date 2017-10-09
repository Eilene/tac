# coding=utf-8

# eng, be
# 文件路径

# 源数据路径
# 为官方数据，文件命名在提取中用到，所以文件名应保持官方原命名；
# 保留原格式，即df与nw训练数据、pre ere未分开，测试数据分开了
train_source_dir = '../../../data/tac_best_data/train_data/spa/source/'
train_ere_dir = '../../../data/tac_best_data/train_data/spa/ere/'
train_annotation_dir = '../../../data/tac_best_data/train_data/spa/annotation/'
test_df_source_dir = '../../../data/tac_best_data/test_data/spa/df/source/'
test_df_ere_dir = '../../../data/tac_best_data/test_data/spa/df/ere/'
test_nw_source_dir = '../../../data/tac_best_data/test_data/spa/nw/source/'
test_nw_ere_dir = '../../../data/tac_best_data/test_data/spa/nw/ere/'
predicted_ere_dir = '../../../data/tac_best_data/predicted_ere/spa/'

# 中间文件路径
train_entity_info_dir = '../../../data/output/spanish_belief/middle_files_train/entity_info/'
train_relation_info_dir = '../../../data/output/spanish_belief/middle_files_train/relation_info/'
train_event_info_dir = '../../../data/output/spanish_belief/middle_files_train/event_info/'
train_em_args_dir = '../../../data/output/spanish_belief/middle_files_train/em_args/'
test_df_entity_info_dir = '../../../data/output/spanish_belief/middle_files_test/df/entity_info/'
test_df_relation_info_dir = '../../../data/output/spanish_belief/middle_files_test/df/relation_info/'
test_df_event_info_dir = '../../../data/output/spanish_belief/middle_files_test/df/event_info/'
test_df_em_args_dir = '../../../data/output/spanish_belief/middle_files_test/df/em_args/'
test_nw_entity_info_dir = '../../../data/output/spanish_belief/middle_files_test/nw/entity_info/'
test_nw_relation_info_dir = '../../../data/output/spanish_belief/middle_files_test/nw/relation_info/'
test_nw_event_info_dir = '../../../data/output/spanish_belief/middle_files_test/nw/event_info/'
test_nw_em_args_dir = '../../../data/output/spanish_belief/middle_files_test/nw/em_args/'
pred_ere_test_df_entity_info_dir = '../../../data/output/spanish_belief/middle_files_pred_ere_test/df/entity_info/'
pred_ere_test_df_relation_info_dir = '../../../data/output/spanish_belief/middle_files_pred_ere_test/df/relation_info/'
pred_ere_test_df_event_info_dir = '../../../data/output/spanish_belief/middle_files_pred_ere_test/df/event_info/'
pred_ere_test_df_em_args_dir = '../../../data/output/spanish_belief/middle_files_pred_ere_test/df/em_args/'
pred_ere_test_nw_entity_info_dir = '../../../data/output/spanish_belief/middle_files_pred_ere_test/nw/entity_info/'
pred_ere_test_nw_relation_info_dir = '../../../data/output/spanish_belief/middle_files_pred_ere_test/nw/relation_info/'
pred_ere_test_nw_event_info_dir = '../../../data/output/spanish_belief/middle_files_pred_ere_test/nw/event_info/'
pred_ere_test_nw_em_args_dir = '../../../data/output/spanish_belief/middle_files_pred_ere_test/nw/em_args/'

# 结果路径
dev_df_predict_dir = '../../../data/output/spanish_belief/predict_dev/df/'
dev_nw_predict_dir = '../../../data/output/spanish_belief/predict_dev/nw/'
test_df_predict_dir = '../../../data/output/spanish_belief/predict_test/df/'
test_nw_predict_dir = '../../../data/output/spanish_belief/predict_test/nw/'
dev_df_y_predict = '../../../data/output/spanish_belief/y_predict_dev/df/'
dev_nw_y_predict = '../../../data/output/spanish_belief/y_predict_dev/nw/'
test_df_y_predict = '../../../data/output/spanish_belief/y_predict_test/df/'
test_nw_y_predict = '../../../data/output/spanish_belief/y_predict_test/nw/'

# 外部数据路径
imdb_dir = '../../../data/external_data/aclImdb/'
glove_6b_100d_path = '../../../data/external_data/glove.6B/glove.6B.100d.txt'
glove_840b_300d_path = '../../../data/external_data/glove.840B.300d/glove.840B.300d.txt'
# 英文情感词典……

# 生成其他数据路径
doctext_path = '../../../data/output/spanish_belief/doctext.txt'
docmodel_path = '../../../data/output/spanish_belief/doc2vec_model_200.txt'