# coding=utf-8


def get_labels(str_labels):
    datanum = len(str_labels)
    labels = [0] * datanum
    for i in range(datanum):
        if str_labels[i] == 'cb':
            labels[i] = 3
        elif str_labels[i] == 'ncb':
            labels[i] = 2
        elif str_labels[i] == 'rob':
            labels[i] = 1
        else:
            labels[i] = 0
    return labels


def get_merged_labels(file_records):
    labels = []
    for i in range(len(file_records)):
        for name in ['relation', 'event']:
            if name in file_records[i]:
                str_labels = file_records[i][name]['label_type']
                f_labels = get_labels(str_labels)
                labels.extend(f_labels)
    return labels
