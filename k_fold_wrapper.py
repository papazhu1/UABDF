from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from imbens.ensemble import *
# from balancedEnsembleClassifier import *
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score
from imbens.metrics import geometric_mean_score, sensitivity_score, specificity_score
from logger import get_logger
from multiprocessing import Pool
from funcs import train_K_fold_paralleling, predict_proba_parallel

'''
此处导入新的包
'''

LOGGER_2 = get_logger("KFoldWrapper")


class KFoldWrapper(object):
    def __init__(self, layer_id, index, config, random_state):
        self.config = config
        self.name = "layer_{}, estimator_{}, {}".format(layer_id, index, self.config["type"])
        if random_state is not None:
            self.random_state = (random_state + hash(self.name)) % 1000000007
        else:
            self.random_state = None
        self.n_fold = self.config["n_fold"]

        self.estimators = [None for i in range(self.config["n_fold"])]
        self.config.pop("n_fold")
        self.estimator_class = globals()[self.config["type"]]
        self.config.pop("type")

    def _init_estimator(self):

        estimator_args = self.config
        est_args = estimator_args.copy()

        return self.estimator_class(**est_args)

    def fit(self, x, y, buckets=None, bucket_variances=None, index_1_sorted=None, uncertainty_1_sorted=None, loss_A_B_stats=None):

        skf = StratifiedKFold(n_splits=self.n_fold, shuffle=True, random_state=self.random_state)
        cv = [(t, v) for (t, v) in skf.split(x, y)]

        n_label = len(np.unique(y))

        args = []

        y_probas = np.zeros((x.shape[0], n_label))
        y_evidences = np.zeros((x.shape[0], n_label))

        for k in range(self.n_fold):
            est = self._init_estimator()
            train_id, val_id = cv[k]

            args.append((train_id, val_id, x, y, est, buckets, bucket_variances, index_1_sorted, uncertainty_1_sorted, loss_A_B_stats))

        pool = Pool(self.n_fold)
        parallel_return = pool.map(train_K_fold_paralleling, args)
        pool.close()

        for k, ret in enumerate(parallel_return):
            est, y_proba, y_pred, val_id, y_evidence = ret

            # LOGGER_2.info(
            #     "{}, n_fold_{},Accuracy={:.4f}, f1_score={:.4f}, auc={:.4f}, gmean={:.4f}, sen={:.4f}, spe={:.4f}, aupr={:.4f}".format(
            #         self.name, k, accuracy_score(y[val_id], y_pred),
            #         f1_score(y[val_id], y_pred, average="macro"), roc_auc_score(y[val_id], y_proba[:, 1]),
            #         geometric_mean_score(y[val_id], y_pred),
            #         sensitivity_score(y[val_id], y_pred), specificity_score(y[val_id], y_pred),
            #         average_precision_score(y[val_id], y_proba[:, 1])))
            y_probas[val_id] += y_proba
            y_evidences[val_id] += y_evidence


            self.estimators[k] = est

        # LOGGER_2.info(
        #     "{}, {},Accuracy={:.4f}, f1_score={:.4f}, auc={:.4f}, gmean={:.4f}, sen={:.4f}, spe={:.4f}, aupr={:.4f}".format(
        #         self.name, "wrapper", accuracy_score(y, np.argmax(y_probas, axis=1)),
        #         f1_score(y, np.argmax(y_probas, axis=1), average="macro"), roc_auc_score(y, y_probas[:, 1]),
        #         geometric_mean_score(y, np.argmax(y_probas, axis=1)),
        #         sensitivity_score(y, np.argmax(y_probas, axis=1)), specificity_score(y, np.argmax(y_probas, axis=1)),
        #         average_precision_score(y, y_probas[:, 1])))
        # LOGGER_2.info("----------")

        alpha = np.sum(y_evidences) / len(y_evidence)
        return y_probas, y_evidences, alpha

    def predict_proba(self, x_test):

        with Pool(self.n_fold) as pool:

            results = pool.map(predict_proba_parallel, [(est, x_test) for est in self.estimators])

        proba = None

        for result in results:
            if proba is None:
                proba = result
            else:
                proba += result

        proba /= self.n_fold

        return proba
