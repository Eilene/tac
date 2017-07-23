# coding=utf-8

# 先随便写写
def find_source(offset, sources):
    source_num = len(sources)
    if source_num == 0:
        return
    for i in range(source_num-1):
        if sources[i]['offset'] <= offset < sources[i+1]['offset']:
            return sources[i]
    if offset >= sources[source_num-1]['offset']:
        return sources[source_num-1]
    return
