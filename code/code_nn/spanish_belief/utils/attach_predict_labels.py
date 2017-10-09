# coding=utf-8


def attach_predict_labels(file_records_dict, y_predict):
    count = 0
    for i in range(len(file_records_dict)):
        for name in ['entity', 'relation', 'event']:
            if name in file_records_dict[i]:
                # 加上label
                for j in range(len(file_records_dict[i][name])):
                    if y_predict[count] == 0:
                        file_records_dict[i][name][j]['predict_type'] = 'na'
                    elif y_predict[count] == 1:
                        file_records_dict[i][name][j]['predict_type'] = 'rob'
                    elif y_predict[count] == 2:
                        file_records_dict[i][name][j]['predict_type'] = 'ncb'
                    else:
                        file_records_dict[i][name][j]['predict_type'] = 'cb'
                    count += 1


def set_cb(file_records):
    for i in range(len(file_records)):
        for name in ['entity', 'relation', 'event']:
            if name in file_records[i]:
                file_records[i][name]['predict_type'] = 'cb'


def use_annotation_labels(file_records):
    for i in range(len(file_records)):
        for name in ['entity', 'relation', 'event']:
            if name in file_records[i]:
                file_records[i][name]['predict_type'] = file_records[i][name]['predict_type']
