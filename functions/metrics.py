from keras import backend as K
from sklearn.metrics import f1_score

def f1(y_true, y_pred, cutoff = 0.5):
    return f1_score((y_true > 0).flatten(), (y_pred > cutoff).flatten())
