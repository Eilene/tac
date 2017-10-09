# coding=utf-8


def get_sample_num(file_records):
    total_num = 0
    for i in range(len(file_records)):
        for name in ['entity', 'relation', 'event']:
            if name in file_records[i]:
                num = len(file_records[i][name])
                total_num += num
    return total_num
