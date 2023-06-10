# from sklearn.base import accuracy_score
# import torch

import numpy as np
from scipy import stats
from sklearn import metrics


def d_prime(auc):
    
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    
    return d_prime

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    classes_num = 2
    stats = []

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    # acc = metrics.accuracy_score(target, output)
    conf_matrix = metrics.confusion_matrix(target, output)
    report = metrics.classification_report(target, output, output_dict=True)

    # Class-wise statistics
    # for k in range(classes_num):

    # Average precision
    avg_precision = metrics.average_precision_score(
        target, output, average=None)

    # AUC
    auc = metrics.roc_auc_score(target, output, average=None)

    # Precisions, recalls
    (precisions, recalls, thresholds) = metrics.precision_recall_curve(
        target, output)

    # FPR, TPR
    (fpr, tpr, thresholds) = metrics.roc_curve(target, output)

    save_every_steps = 10     # Sample statistics to reduce size
    dict = {'precisions': precisions[0::save_every_steps],
            'recalls': recalls[0::save_every_steps],
            'AP': avg_precision,
            'fpr': fpr[0::save_every_steps],
            'fnr': 1. - tpr[0::save_every_steps],
            'auc': auc,
            # note acc is not class-wise, this is just to keep consistent with other metrics
            'acc': ['acc']
            }
    stats.append(dict)

    return stats

def calculate_stats_(output, target, ):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    conf_matrix = metrics.confusion_matrix(target, output)
    report = metrics.classification_report(target, output, output_dict=True)

    # Average precision
    precisions = report['weighted avg']['precision']
    recalls = report['weighted avg']['recall']
    f1 = report['weighted avg']['f1-score']
    
    # Average precision
    avg_precision = metrics.average_precision_score(
        target, output, average=None)
    
    # AUC
    auc = metrics.roc_auc_score(target, output, average=None)

    dict = {'precisions': precisions,
            'recalls': recalls,
            'f1': f1,
            'auc': auc,
            'AP': avg_precision,
            'acc': report['accuracy'],
            '0_t': int(conf_matrix[0][0]),
            '0_f': int(conf_matrix[0][1]),
            '1_t': int(conf_matrix[1][0]),
            '1_f': int(conf_matrix[1][1])
            }
    
    return dict