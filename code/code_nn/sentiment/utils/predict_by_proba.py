# coding=utf-8


def predict_by_proba(proba, threshold):
    y_pred = []
    for p in proba:
        if p[1] - p[0] > threshold:
            y_pred.append(2)
        elif p[0] - p[1] > threshold:
            y_pred.append(1)
        else:
            y_pred.append(0)
    return y_pred
