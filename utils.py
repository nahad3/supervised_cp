from sklearn.metrics import f1_score
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def get_f1_score(y,win_y,fpr,tpr,thresholds,change_metric):
    y_temp = y
    pos_indices = np.where(y == 1)[0]
    fnr = 1 - fpr
    best_thresh_no_enc = thresholds[np.argmax((tpr + fnr) / 2)]
    score_no_enc = change_metric >= best_thresh_no_enc
    change_metric_2 = score_no_enc
    peaks, _ = find_peaks(change_metric , height=best_thresh_no_enc,width = 1)
    y_pred = np.zeros(len(y))
    y_pred[peaks] = 1

    for pos_idx in pos_indices:
        y_temp[pos_idx - win_y: pos_idx, :] = 1
        y_temp[pos_idx: pos_idx + win_y, :] = 1

    #for pos_idx in pos_indices:
    #    if sum(change_metric_2[pos_idx - win_y/2 : pos_idx + win_y/2]) >= 1:
    #        change_metric_2[pos_idx - win_y/2 : pos_idx + win_y/2] = 0
    #        change_metric_2[pos_idx] = 1
    #fnr = 1 - fpr
    #best_thresh_no_enc = thresholds[np.argmax((tpr + fnr) / 2)]
    #score_no_enc = change_metric >= best_thresh_no_enc

    plt.plot(score_no_enc)
    plt.plot(0.1*y,linestyle='--')
    plt.show()
    return  f1_score(y_temp, score_no_enc, average='binary')
    #return f1_score(y_temp.reshape(-1,), y_pred )


def ChangePointF1Score(gtLabel,  window,fpr,tpr,thresholds,change_metric):
    # given ground truth sequence of labels, and real labels, computes precision, recall and f1 score assuming leniancy of (window)
    fnr = 1 - fpr
    best_thresh_no_enc = thresholds[np.argmax((tpr + fnr) / 2)]
    label = change_metric >= best_thresh_no_enc
    l = len(gtLabel)
    window = int(window)
    # Compute precision
    tp = 0
    totalLabel = np.sum(label)
    for i in np.argwhere(label == 1):
        if (np.max(gtLabel[np.maximum(1, i[0] - window):np.minimum(l, i[0] + window)]) == 1):  # true change point close to label
            tp += 1

    fn = 0
    totalGt = np.sum(gtLabel)
    for i in np.argwhere(gtLabel == 1):
        if (np.max(label[np.maximum(0, i[0] - window):np.minimum(l - 1, i[0] + window)]) == 1):
            fn += 1

    precision = tp / totalLabel
    recall = fn / totalGt
    if (precision + recall == 0):
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)
    if (totalLabel == 0):
        precision = 0
    if (totalGt == 0):
        recall = 0
    return (f1, precision, recall)