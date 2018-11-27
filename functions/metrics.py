from keras import backend as K

def f1(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)

    tp = K.sum(y_true_f * y_pred_f)
    fp = K.sum((1 - y_true_f) * y_pred_f)
    fn = K.sum(y_true_f * (1 - y_pred_f))

    prec = tp / (tp + fp)
    recall = tp / (tp + fn)

    f1_score = 2 * (prec * recall) / (prec + recall)
    return f1_score
