# coding=utf-8

from sklearn import metrics


def evaluation(y_test, y_predict):
    accuracy = metrics.accuracy_score(y_test, y_predict)
    precision = metrics.precision_score(y_test, y_predict)
    recall = metrics.recall_score(y_test, y_predict)
    f1 = metrics.f1_score(y_test, y_predict)
    print "Accuracy: ", accuracy
    print "Precision: ", precision
    print "Recall: ", recall
    print "F1: ", f1
