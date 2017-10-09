# coding=utf-8


def predict_by_proba_3classes_threshold(probas, threshold1=0.1, threshold2=0.0):
    y_pred = []
    for p in probas:
        if p[1] - p[0] > threshold2:
            y_pred.append(2)
        elif p[0] - p[1] > threshold1:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred


def predict_by_proba(probas):
    y_pred = []
    for p in probas:
            y_pred.append(p.tolist().index(max(p)))
    return y_pred
