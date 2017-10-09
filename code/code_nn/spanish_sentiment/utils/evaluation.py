# coding=utf-8

from sklearn import metrics


# 3类的测试评价
# 0: none, 1: neg, 2: pos
def evaluation_3classes(y_test, y_predict):
    # 有无的评价
    print 'not none vs none: '
    y_test_none = [0 if y_test[i] == 0 else 1 for i in range(len(y_test))]
    y_predict_none = [0 if y_predict[i] == 0 else 1 for i in range(len(y_predict))]
    f1 = evaluation(y_test_none, y_predict_none)

    # 正类的评价
    print 'pos vs not pos: '
    y_test_none = [1 if y_test[i] == 2 else 0 for i in range(len(y_test))]
    y_predict_none = [1 if y_predict[i] == 2 else 0 for i in range(len(y_predict))]
    evaluation(y_test_none, y_predict_none)

    # 负类的评价
    print 'neg vs not neg: '
    y_test_none = [1 if y_test[i] == 1 else 0 for i in range(len(y_test))]
    y_predict_none = [1 if y_predict[i] == 1 else 0 for i in range(len(y_predict))]
    evaluation(y_test_none, y_predict_none)

    return f1  # 先暂用这个指标选择模型


def evaluation(y_test, y_predict):
    accuracy = metrics.accuracy_score(y_test, y_predict)
    precision = metrics.precision_score(y_test, y_predict)
    recall = metrics.recall_score(y_test, y_predict)
    f1 = metrics.f1_score(y_test, y_predict)
    print "Accuracy: ", accuracy
    print "Precision: ", precision
    print "Recall: ", recall
    print "F1: ", f1
    return f1


def evaluation_source(file_records_dict):
    count = 0
    right_count = 0
    for fr in file_records_dict:
        for name in ['entity', 'relation', 'event']:
            if name in fr:
                for r in fr[name]:
                    if r['label_polarity'] != 'none':
                        count += 1
                        if (('predict_source_id' not in r) and (r['source_length'] == 0)) or \
                                (('predict_source_id' in r) and (r['predict_source_id'] == r['source_id'])):
                            right_count += 1
                        # else:
                        #     print fr['filename'], r[name+'_mention_id']
                        #     print 'actual source:', r['source_id'], r['source_offset'], r['source_length'], r['source_text']
                        #     if 'predict_source_id' in r:
                        #         print 'predict source:', r['predict_source_id'], r['predict_source_offset'], \
                        #             r['predict_source_length'], r['predict_source_text']
                            # 错误情况及其他情况：
                            # 1.不对啊，有的明显是作者观点官方却标的无source；有的是belief有源，对应sentiment没有
                            # 2.saying，可只找say前接着的即可；主要都是这个错
                                # he genuinely was looking for a companion这种怎么办？
                                # The St. Louis Rams fired coach Steve Spagnuolo
                            # 3.quote的在原post中：不多
                            # 无作者quote，会用句中I作为source
    accuracy = float(right_count) / float(count)
    print 'Total, right and wrong number of source finding: ', count, right_count, count - right_count
    print 'Accuracy of find source:', accuracy
