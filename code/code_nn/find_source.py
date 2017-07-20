# coding=utf-8


def find_source(test_files):
    # 对每个文件
    # 读对应source文件
    # 生成该文件source信息列表
    # 根据offset找上面一个最近的source

    # 先赋none
    types = ['entity', 'relation', 'event']
    for t in types:
        for i in range(len(test_files)):
            if test_files[i][t] is not None:
                for j in range(len(test_files[i][t])):
                    test_files[i][t][j]['source'] = None

    return test_files