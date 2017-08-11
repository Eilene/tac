# coding=utf-8


def attach_predict_labels(file_records_dict, y_predict):
    count = 0
    for i in range(len(file_records_dict)):
        for name in ['entity', 'relation', 'event']:
            if name in file_records_dict[i]:
                # 加上label
                for j in range(len(file_records_dict[i][name])):
                    if y_predict[count] == 0:
                        file_records_dict[i][name][j]['predict_polarity'] = 'none'
                    elif y_predict[count] == 1:
                        file_records_dict[i][name][j]['predict_polarity'] = 'neg'
                    else:
                        file_records_dict[i][name][j]['predict_polarity'] = 'pos'
                    count += 1
    return file_records_dict


def set_neg(file_records):
    for i in range(len(file_records)):
        for name in ['entity', 'relation', 'event']:
            if name in file_records[i]:
                file_records[i][name]['predict_polarity'] = 'neg'
    return file_records


def use_annotation_labels(file_records):
    for i in range(len(file_records)):
        for name in ['entity', 'relation', 'event']:
            if name in file_records[i]:
                file_records[i][name]['predict_polarity'] = file_records[i][name]['label_polarity']
    return file_records
