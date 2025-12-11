from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics import roc_auc_score, average_precision_score
from imbens.metrics import geometric_mean_score, sensitivity_score, specificity_score


def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def f1_binary(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average="binary")
    return f1


def f1_micro(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average="micro")
    return f1


def f1_macro(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, average="macro")
    return f1


def gmean(y_true, y_pred):
    return geometric_mean_score(y_true, y_pred)


def sensitivity(y_true, y_pred):
    return sensitivity_score(y_true, y_pred)


def specificity(y_true, y_pred):
    return specificity_score(y_true, y_pred)


def roc_auc(y_true, y_pred_probas):
    return roc_auc_score(y_true, y_pred_probas[:, 1])


def generalized_performance(y_true, y_pred, y_pred_probas):
    """
    Calculate weighted performance
    :param y_true: true labels
    :param y_pred: predicted labels
    :return: weighted performance
    """
    return (f1_score(y_true, y_pred, average="macro") * 0.25 +
            roc_auc_score(y_true, y_pred_probas[:, 1]) * 0.25 +
            geometric_mean_score(y_true, y_pred) * 0.25 +
            average_precision_score(y_true, y_pred_probas[:, 1]) * 0.25)
