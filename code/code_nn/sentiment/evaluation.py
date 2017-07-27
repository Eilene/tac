# coding=utf-8

from sklearn import metrics


# 3类的测试评价
# 0: none, 1: neg, 2: pos
def evaluation_3classes(y_test, y_predict):
    # 有无的评价
    print 'not none vs none: '
    y_test_none = [[0] if y_test[i] == [0] else [1] for i in range(len(y_test))]
    y_predict_none = [[0] if y_predict[i] == [0] else [1] for i in range(len(y_predict))]
    evaluation(y_test_none, y_predict_none)

    # 正类的评价
    print 'pos vs not pos: '
    y_test_none = [[1] if y_test[i] == [2] else [0] for i in range(len(y_test))]
    y_predict_none = [[1] if y_predict[i] == [2] else [0] for i in range(len(y_predict))]
    evaluation(y_test_none, y_predict_none)

    # 负类的评价
    print 'neg vs not neg: '
    y_test_none = [[1] if y_test[i] == [1] else [0] for i in range(len(y_test))]
    y_predict_none = [[1] if y_predict[i] == [1] else [0] for i in range(len(y_predict))]
    evaluation(y_test_none, y_predict_none)


def evaluation(y_test, y_predict):
    accuracy = metrics.accuracy_score(y_test, y_predict)
    precision = metrics.precision_score(y_test, y_predict)
    recall = metrics.recall_score(y_test, y_predict)
    f1 = metrics.f1_score(y_test, y_predict)
    print "Accuracy: ", accuracy
    print "Precision: ", precision
    print "Recall: ", recall
    print "F1: ", f1