# coding=utf-8


def without_none(file_records):
    for i in range(len(file_records)):
        for name in ['entity', 'relation', 'event']:
            if name in file_records[i]:
                file_records[i][name] = \
                    file_records[i][name][file_records[i][name].label_polarity != 'none'].reset_index()


def to_dict(file_records):
    for i in range(len(file_records)):
        for name in ['entity', 'relation', 'event', 'em_args']:
            if name in file_records[i]:
                file_records[i][name] = file_records[i][name].to_dict(orient='records')
