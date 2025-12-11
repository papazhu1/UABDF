import numpy as np


class Layer(object):
    def __init__(self, layer_id):
        self.layer_id = layer_id
        self.estimators = []
        self.confusion_matrices = []
        self.confusion_matrix_mean = []

    def add_est(self, estimator):
        if estimator is not None:
            self.estimators.append(estimator)

    # 返回每个estimator的预测概率
    def predict_proba(self, x):
        proba = None
        for est in self.estimators:
            proba = est.predict_proba(x) if proba is None else np.hstack((proba, est.predict_proba(x)))
        return proba

    # 返回每个estimator的预测概率的平均值
    def _predict_proba(self, x_test):
        proba = None
        for est in self.estimators:
            proba = est.predict_proba(x_test) if proba is None else proba + est.predict_proba(x_test)
        proba /= len(self.estimators)
        return proba
