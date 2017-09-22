# coding=utf-8


def predict_by_proba(probas):
    y_pred = []
    for p in probas:
            y_pred.append(p.tolist().index(max(p)))
    return y_pred
