# coding=utf-8


def attach_predict_labels(test_files, labels):
    index = 0
    types = ['entity', 'relation', 'event']
    for t in types:
        for i in range(len(test_files)):
            if test_files[i][t] is not None:
                test_files[i][t] = test_files[i][t].to_dict(orient='records')
                for j in range(len(test_files[i][t])):
                    if labels[index] == [1]:
                        test_files[i][t][j]['predict_polarity'] = 'pos'
                    else:
                        test_files[i][t][j]['predict_polarity'] = 'neg'
                    # print test_files[i][t][j]['predict_polarity']
                    index += 1
            # print test_files[i][t]
    return test_files
