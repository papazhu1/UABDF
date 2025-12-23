import json
from pickletools import uint8

import numpy as np
from sklearn import ensemble

from logger import get_logger
from k_fold_wrapper import KFoldWrapper
from layer import Layer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, average_precision_score, precision_score, \
    recall_score, \
    confusion_matrix, classification_report
from imbens.metrics import geometric_mean_score, sensitivity_score, specificity_score, classification_report_imbalanced
from sklearn.metrics import confusion_matrix
import pickle
from funcs import error_to_hardness, calculate_bin_capacities
from func_utils import *

LOGGER = get_logger("UncertaintyAwareBalancedDeepForest")


class UncertaintyAwareBalancedDeepForest(object):

    def __init__(self, config):
        self.random_state = config["random_state"]
        self.max_layers = config["max_layers"]
        self.early_stop_rounds = config["early_stop_rounds"]
        self.if_stacking = config["if_stacking"]
        self.if_save_model = config["if_save_model"]
        self.use_u_KL_method = config["use_u_KL_method"]

        self.train_evaluation = config["train_evaluation"]
        self.estimator_configs = config["estimator_configs"]
        self.enhancement_vector_method = config["enhancement_vector_method"]
        self.n_estimators = config["n_estimators"]
        self.layers = []
        self.alpha_list = []
        self.per_layer_res = []
        self.per_layer_res_weighted_layers = []
        self.mean_KL_per_layer = []
        self.lamb = config["lamb"]

    def calculate_KL_u_use_evidence(self, y_train_evidence_per_layer_per_forest, n_classes_):
        all_instance_KL = np.zeros((len(y_train_evidence_per_layer_per_forest[0]), 1))
        all_instance_u = np.zeros((len(y_train_evidence_per_layer_per_forest[0]), 1))

        for sample_idx in range(len(y_train_evidence_per_layer_per_forest[0])):
            evidence_array = np.zeros(
                (len(self.estimator_configs) * len(y_train_evidence_per_layer_per_forest), n_classes_))

            forest_idx = 0
            for layer_loc in range(len(y_train_evidence_per_layer_per_forest)):
                for forest_loc in range(len(self.estimator_configs)):
                    if layer_loc >= len(y_train_evidence_per_layer_per_forest):
                        raise ValueError(
                            "layer_loc {} >= len(y_train_probas_per_layer_per_forest) {}".format(
                                layer_loc, len(y_train_evidence_per_layer_per_forest)))
                    if sample_idx >= len(y_train_evidence_per_layer_per_forest[layer_loc]):
                        raise ValueError(
                            "sample_idx {} >= len(y_train_probas_per_layer_per_forest[layer_loc]) {}".format(
                                sample_idx, len(y_train_evidence_per_layer_per_forest[layer_loc])))
                    if forest_loc * n_classes_ + 1 >= len(
                            y_train_evidence_per_layer_per_forest[layer_loc][sample_idx]):
                        raise ValueError(
                            "forest_loc * n_classes_ + 1 {} >= len(y_train_probas_per_layer_per_forest[layer_loc][sample_idx]) {}".format(
                                forest_loc * n_classes_ + 1,
                                len(y_train_evidence_per_layer_per_forest[layer_loc][sample_idx])))

                    evidence_array[forest_idx, 0] = y_train_evidence_per_layer_per_forest[layer_loc][sample_idx][
                        forest_loc * n_classes_]
                    evidence_array[forest_idx, 1] = y_train_evidence_per_layer_per_forest[layer_loc][sample_idx][
                        forest_loc * n_classes_ + 1]
                    forest_idx += 1

            alpha_combined, b_combined, u_combined = DS_Combine_ensemble_for_instances(evidence_array[0].reshape(1, 2),
                                                                                       evidence_array[1].reshape(1, 2))
            for k in range(2, len(self.estimator_configs)):
                alpha_combined, b_combined, u_combined = DS_Combine_ensemble_for_instances(alpha_combined.reshape(1, 2),
                                                                                           evidence_array[k].reshape(1,
                                                                                                                     2))

            KL = calculate_KL(alpha_combined, n_classes_)

            all_instance_KL[sample_idx] = KL
            all_instance_u[sample_idx] = u_combined

        return all_instance_KL, all_instance_u

    def calculate_KL_u(self, y_probas_per_layer_per_forest, n_classes_):
        all_instance_KL = np.zeros((len(y_probas_per_layer_per_forest[0]), 1))
        all_instance_u = np.zeros((len(y_probas_per_layer_per_forest[0]), 1))

        for sample_idx in range(len(y_probas_per_layer_per_forest[0])):
            pred_list = []
            for layer_loc in range(len(y_probas_per_layer_per_forest)):
                for forest_loc in range(len(self.estimator_configs)):
                    if layer_loc >= len(y_probas_per_layer_per_forest):
                        raise ValueError(
                            "layer_loc {} >= len(y_train_probas_per_layer_per_forest) {}".format(
                                layer_loc, len(y_probas_per_layer_per_forest)))
                    if sample_idx >= len(y_probas_per_layer_per_forest[layer_loc]):
                        raise ValueError(
                            "sample_idx {} >= len(y_train_probas_per_layer_per_forest[layer_loc]) {}".format(
                                sample_idx, len(y_probas_per_layer_per_forest[layer_loc])))
                    if forest_loc * n_classes_ + 1 >= len(
                            y_probas_per_layer_per_forest[layer_loc][sample_idx]):
                        raise ValueError(
                            "forest_loc * n_label + 1 {} >= len(y_train_probas_per_layer_per_forest[layer_loc][sample_idx]) {}".format(
                                forest_loc * n_classes_ + 1,
                                len(y_probas_per_layer_per_forest[layer_loc][sample_idx])))

                    pred_list.append(y_probas_per_layer_per_forest[layer_loc][sample_idx][
                                         forest_loc * n_classes_ + 1])

            pred_array = np.array(pred_list)
            expanded_pred_array = np.vstack((1 - pred_array, pred_array)).T
            expanded_pred_array *= (self.n_estimators + 1)
            print("expanded_pred_array", expanded_pred_array)

            # 现在要对每个样本计算loss
            alpha_combined, b_combined, u_combined = DS_Combine_ensemble_for_instances(
                expanded_pred_array[0].reshape(1, 2),
                expanded_pred_array[1].reshape(1, 2))
            for k in range(2, len(self.estimator_configs)):
                alpha_combined, b_combined, u_combined = DS_Combine_ensemble_for_instances(alpha_combined.reshape(1, 2),
                                                                                           expanded_pred_array[
                                                                                               k].reshape(1,
                                                                                                          2))
            KL = calculate_KL(alpha_combined, n_classes_)

            all_instance_KL[sample_idx] = KL
            all_instance_u[sample_idx] = u_combined

        return all_instance_KL, all_instance_u

    def fit(self, x_train, y_train):

        x_train, n_feature, n_classes_ = self.preprocess(x_train, y_train)
        self.n_classes_ = n_classes_
        evaluate = self.train_evaluation
        best_layer_id = 0
        depth = 0
        best_layer_evaluation = 0.0

        enhanced_vector_per_layer = []

        buckets = None

        y_train_probas_per_layer = []
        y_train_evidence_per_layer = []

        y_train_probas_per_layer_per_forest = []
        y_train_evidence_per_layer_per_forest = []

        y_train_pred_per_layer = []

        y_train_probas_sum_all_est = np.zeros((x_train.shape[0], n_classes_))

        while depth < self.max_layers:

            cur_layer_x_train = None
            cur_layer_y_train = None

            y_train_probas = np.zeros((x_train.shape[0], n_classes_ * len(self.estimator_configs)))
            y_train_evidences = np.zeros((x_train.shape[0], n_classes_ * len(self.estimator_configs)))

            enhancement_vector_cm = np.zeros((x_train.shape[0], n_classes_ * len(self.estimator_configs)))
            enhancement_vector_cm_mean = np.zeros((x_train.shape[0], n_classes_))

            current_layer = Layer(depth)
            LOGGER.info(
                "-----------------------------------------layer-{}--------------------------------------------".format(
                    current_layer.layer_id))
            # LOGGER.info("The shape of x_train is {}".format(x_train.shape))

            y_train_probas_avg = np.zeros((x_train.shape[0], n_classes_))

            if depth == 0:
                cur_layer_x_train = x_train
                cur_layer_y_train = y_train
            else:
                if self.if_stacking == False:
                    cur_layer_x_train = np.hstack((x_train, enhanced_vector_per_layer[depth - 1]))
                else:
                    enhanced_vector = enhanced_vector_per_layer[0]
                    for i in range(1, depth):
                        enhanced_vector = np.hstack((enhanced_vector, enhanced_vector_per_layer[i]))

                    cur_layer_x_train = np.hstack((x_train, enhanced_vector))
                cur_layer_y_train = y_train

            if cur_layer_x_train is None or cur_layer_y_train is None:
                raise ValueError("cur_layer_x_train or cur_layer_y_train is None")

            for index in range(len(self.estimator_configs)):
                config = self.estimator_configs[index].copy()

                y_train_probas_summed = np.zeros((x_train.shape[0], n_classes_))
                index_0 = np.where(cur_layer_y_train == 0)[0]
                index_1 = np.where(cur_layer_y_train == 1)[0]
                if depth == 0:
                    pass
                else:
                    for per_depth in range(depth):
                        y_train_probas_summed += y_train_probas_per_layer[per_depth]
                y_train_probas_summed = (y_train_probas_summed * len(
                    self.estimator_configs) + y_train_probas_avg) / (
                                                len(self.estimator_configs) * depth + index)

                error_0 = y_train_probas_summed[index_0, 0] * (-1) + 1
                error_1 = y_train_probas_summed[index_1, 1] * (-1) + 1

                hardness_0 = error_to_hardness(error_0)
                hardness_1 = error_to_hardness(error_1)

                hardness_1_sorted_idx = np.argsort(hardness_1)
                index_1_sorted = index_1[hardness_1_sorted_idx]
                hardness_1_sorted = hardness_1[hardness_1_sorted_idx]

                num_bins = 5
                alpha = 0.2

                capacities = calculate_bin_capacities(num_bins, depth, alpha)

                q = [0 for _ in range(num_bins)]

                for i in range(num_bins):
                    q[i] = 100 * np.sum(capacities[:i + 1])
                    q[i] = round(q[i])

                percentiles = np.percentile(hardness_0, q)
                buckets = [[] for _ in range(num_bins)]
                buckets_error_idx = [[] for _ in range(num_bins)]

                for err_idx, (i, err) in enumerate(zip(index_0, hardness_0)):
                    for j in range(num_bins):
                        if err <= percentiles[j]:
                            buckets[j].append(i)
                            buckets_error_idx[j].append(err_idx)
                            break

                mean_hardness_per_bins = []

                if depth != 0 and index == 0:
                    for i, bucket in enumerate(buckets):
                        hardnesses = hardness_0[buckets_error_idx[i]]
                        mean_hardness = np.mean(hardnesses)
                        mean_hardness_per_bins.append(mean_hardness)

                if len(buckets[0]) == 0:

                    for i, (i_0, err) in enumerate(zip(index_0, hardness_0)):
                        buckets[i % num_bins].append(i_0)

                k_fold_est = KFoldWrapper(current_layer.layer_id, index, config, random_state=self.random_state)

                def error_to_uncertainty_for_class_1(index_1_sorted, error_1_sorted):

                    uncertainty_for_class_1 = []

                    for serial_idx, sample_idx in enumerate(index_1_sorted):
                        pred_list_class_1 = []
                        for layer_loc in range(len(y_train_probas_per_layer_per_forest)):
                            for forest_loc in range(len(self.estimator_configs)):
                                if layer_loc >= len(y_train_probas_per_layer_per_forest):
                                    raise ValueError(
                                        "layer_loc {} >= len(y_train_probas_per_layer_per_forest) {}".format(layer_loc,
                                                                                                             len(y_train_probas_per_layer_per_forest)))
                                if sample_idx >= len(y_train_probas_per_layer_per_forest[layer_loc]):
                                    raise ValueError(
                                        "sample_idx {} >= len(y_train_probas_per_layer_per_forest[layer_loc]) {}".format(
                                            sample_idx, len(y_train_probas_per_layer_per_forest[layer_loc])))
                                if forest_loc * n_classes_ + 1 >= len(
                                        y_train_probas_per_layer_per_forest[layer_loc][sample_idx]):
                                    raise ValueError(
                                        "forest_loc * n_classes_ + 1 {} >= len(y_train_probas_per_layer_per_forest[layer_loc][sample_idx]) {}".format(
                                            forest_loc * n_classes_ + 1,
                                            len(y_train_probas_per_layer_per_forest[layer_loc][sample_idx])))

                                pred_list_class_1.append(y_train_probas_per_layer_per_forest[layer_loc][sample_idx][
                                                             forest_loc * n_classes_ + 1])

                        pred_array = np.array(pred_list_class_1)

                        expanded_pred_array = np.vstack((1 - pred_array, pred_array)).T
                        expanded_pred_array *= (self.n_estimators + 1)

                        # 现在要对每个样本计算loss
                        loss_combined, b_combined, u_combined = DS_Combine_ensemble_for_instances(
                            expanded_pred_array[0].reshape(1, 2),
                            expanded_pred_array[1].reshape(1, 2))

                        for i in range(2, len(self.estimator_configs)):
                            loss_combined, b_combined, u_combined = DS_Combine_ensemble_for_instances(
                                loss_combined.reshape(1, 2),
                                expanded_pred_array[i].reshape(1, 2))

                        loss, A, B = ce_loss(y_train[sample_idx], loss_combined, n_classes_, lamb=self.lamb)

                        errors = 1 - pred_array
                        hardnesses = error_to_hardness(errors)
                        cur_sample_uncertainty = np.var(hardnesses)

                        uncertainty_for_class_1.append(cur_sample_uncertainty)

                    resort_serial_idx_for_class_1 = np.argsort(uncertainty_for_class_1)
                    resort_idx_for_class_1 = [index_1_sorted[idx] for idx in resort_serial_idx_for_class_1]
                    uncertainty_for_class_1 = [uncertainty_for_class_1[idx] for idx in resort_serial_idx_for_class_1]

                    returns = []
                    returns.append(uncertainty_for_class_1)
                    returns.append(resort_idx_for_class_1)

                    return returns

                def error_to_uncertainty(buckets):

                    bucket_variances = []
                    loss_A_B_stats = {
                        "index": [],
                        "loss": [],
                        "A": [],
                        "B": []
                    }
                    all_instance_B = np.zeros((x_train.shape[0], 1))
                    all_instance_u = np.zeros((x_train.shape[0], 1))

                    for i, bucket in enumerate(buckets):
                        cur_bucket_sample_uncertainty = []

                        for j, sample_idx in enumerate(bucket):
                            if i == 0 and j == 0:
                                pass

                            pred_list = []
                            evidence_array = np.zeros(
                                (len(self.estimator_configs) * len(y_train_probas_per_layer_per_forest), n_classes_))
                            # print("evidence_array.shape", evidence_array.shape)
                            # print(len(self.estimator_configs) * len(y_train_probas_per_layer_per_forest))
                            # print(n_classes_)

                            for layer_loc in range(len(y_train_probas_per_layer_per_forest)):
                                for forest_loc in range(len(self.estimator_configs)):
                                    if layer_loc >= len(y_train_probas_per_layer_per_forest):
                                        raise ValueError(
                                            "layer_loc {} >= len(y_train_probas_per_layer_per_forest) {}".format(
                                                layer_loc, len(y_train_probas_per_layer_per_forest)))
                                    if sample_idx >= len(y_train_probas_per_layer_per_forest[layer_loc]):
                                        raise ValueError(
                                            "sample_idx {} >= len(y_train_probas_per_layer_per_forest[layer_loc]) {}".format(
                                                sample_idx, len(y_train_probas_per_layer_per_forest[layer_loc])))
                                    if forest_loc * n_classes_ + 1 >= len(
                                            y_train_probas_per_layer_per_forest[layer_loc][sample_idx]):
                                        raise ValueError(
                                            "forest_loc * n_classes_ + 1 {} >= len(y_train_probas_per_layer_per_forest[layer_loc][sample_idx]) {}".format(
                                                forest_loc * n_classes_ + 1,
                                                len(y_train_probas_per_layer_per_forest[layer_loc][sample_idx])))

                                    pred_list.append(y_train_probas_per_layer_per_forest[layer_loc][sample_idx][
                                                         forest_loc * n_classes_ + 1])
                            forest_idx = 0
                            for layer_loc in range(len(y_train_evidence_per_layer_per_forest)):
                                for forest_loc in range(len(self.estimator_configs)):
                                    if layer_loc >= len(y_train_evidence_per_layer_per_forest):
                                        raise ValueError(
                                            "layer_loc {} >= len(y_train_probas_per_layer_per_forest) {}".format(
                                                layer_loc, len(y_train_evidence_per_layer_per_forest)))
                                    if sample_idx >= len(y_train_evidence_per_layer_per_forest[layer_loc]):
                                        raise ValueError(
                                            "sample_idx {} >= len(y_train_probas_per_layer_per_forest[layer_loc]) {}".format(
                                                sample_idx, len(y_train_evidence_per_layer_per_forest[layer_loc])))
                                    if forest_loc * n_classes_ + 1 >= len(
                                            y_train_evidence_per_layer_per_forest[layer_loc][sample_idx]):
                                        raise ValueError(
                                            "forest_loc * n_classes_ + 1 {} >= len(y_train_probas_per_layer_per_forest[layer_loc][sample_idx]) {}".format(
                                                forest_loc * n_classes_ + 1,
                                                len(y_train_evidence_per_layer_per_forest[layer_loc][sample_idx])))

                                    evidence_array[forest_idx, 0] = \
                                        y_train_evidence_per_layer_per_forest[layer_loc][sample_idx][
                                            forest_loc * n_classes_] / 50
                                    evidence_array[forest_idx, 1] = \
                                        y_train_evidence_per_layer_per_forest[layer_loc][sample_idx][
                                            forest_loc * n_classes_ + 1] / 50
                                    forest_idx += 1

                            pred_array = np.array(pred_list)

                            expanded_pred_array = np.vstack((1 - pred_array, pred_array)).T
                            expanded_pred_array *= (self.n_estimators + 1)

                            Opinion = DS_Combine2(evidence_array[0].reshape(1, 2),
                                                  evidence_array[1].reshape(1, 2))

                            W = 1
                            beta_distributions = []
                            for k in range(len(self.estimator_configs)):
                                alpha_k, beta_k = evidence_array[k][0] + W, evidence_array[k][1] + W
                                beta_distributions.append([alpha_k, beta_k])

                            for k in range(2, len(self.estimator_configs)):
                                Opinion = DS_Combine_evidence_with_opinion(evidence_array[k].reshape(1, 2),
                                                                           Opinion)

                            b_combined, d_combined, u_combined = Opinion

                            alpha_combined = (b_combined * 2 * W) / u_combined + W
                            beta_combined = (d_combined * 2 * W) / u_combined + W

                            # loss, A, B = ce_loss(y_train[sample_idx], alpha_combined, n_classes_)
                            loss, A, B = ce_loss2(y_train[sample_idx], alpha_combined, beta_combined,
                                                  beta_distributions)
                            loss_A_B_stats["index"].append(sample_idx)
                            loss_A_B_stats["loss"].append(loss)
                            loss_A_B_stats["A"].append(A)
                            loss_A_B_stats["B"].append(B)

                            all_instance_B[sample_idx] = B
                            all_instance_u[sample_idx] = u_combined

                            # print("type(expanded_pred_array):", type(expanded_pred_array))
                            # print("shape(expanded_pred_array):", expanded_pred_array.shape)
                            # print("expanded_pred_array:", expanded_pred_array)
                            errors = 1 - pred_array
                            # print("type(errors):", type(errors))
                            # print("shape(errors):", errors.shape)
                            # print("errors:", errors)
                            hardnesses = error_to_hardness(errors)
                            # print("type(hardnesses):", type(hardnesses))
                            # print("shape(hardnesses):", hardnesses.shape)
                            # print("hardnesses:", hardnesses)

                            # cur_sample_uncertainty = np.var(hardnesses)
                            cur_sample_uncertainty = loss
                            cur_bucket_sample_uncertainty.append(cur_sample_uncertainty)

                        if len(cur_bucket_sample_uncertainty) != len(bucket):
                            raise ValueError(
                                f"Mismatch before sorting: len(cur_bucket_sample_uncertainty) = {len(cur_bucket_sample_uncertainty)}, len(bucket) = {len(bucket)}")

                        sorted_sample_idx = np.argsort(cur_bucket_sample_uncertainty)
                        # print(
                        #     f"Before sorting, bucket[{i}] length: {len(bucket)}, cur_bucket_sample_uncertainty length: {len(cur_bucket_sample_uncertainty)}")
                        buckets[i] = [bucket[idx] for idx in sorted_sample_idx]
                        cur_bucket_sample_uncertainty = np.array(cur_bucket_sample_uncertainty)
                        cur_bucket_sample_uncertainty = cur_bucket_sample_uncertainty[sorted_sample_idx]
                        bucket_variances.append(cur_bucket_sample_uncertainty)
                        # print(
                        #     f"After sorting, buckets[{i}] length: {len(buckets[i])}, cur_bucket_sample_uncertainty length: {len(cur_bucket_sample_uncertainty)}")

                        if len(cur_bucket_sample_uncertainty) != len(buckets[i]):
                            raise ValueError("len(cur_bucket_sample_uncertainty) != len(buckets[{}])".format(i))

                    return bucket_variances, loss_A_B_stats, all_instance_B, all_instance_u

                bucket_variances = None
                loss_A_B_stats = None
                all_instance_KL = None
                all_instance_u = None

                if depth == 0:
                    buckets = None
                else:
                    bucket_variances, loss_A_B_stats, all_instance_KL, all_instance_u = error_to_uncertainty(buckets)

                    if depth == 1:
                        uncertainty_for_class_1, resort_idx_for_class_1 = error_to_uncertainty_for_class_1(
                            index_1_sorted, hardness_1_sorted)
                        self.uncertainty_for_class_1_list = []
                        self.uncertainty_for_class_1_list.append(uncertainty_for_class_1)
                    else:

                        uncertainty_for_class_1, resort_idx_for_class_1 = error_to_uncertainty_for_class_1(
                            index_1_sorted, hardness_1_sorted)
                        self.uncertainty_for_class_1_list.append(uncertainty_for_class_1)

                if buckets is not None:
                    for i in range(num_bins):
                        # print(f"Bucket {i} has {len(buckets[i])} samples.")
                        # print(f"Bucket {i} has {len(bucket_variances[i])} uncertainties.")
                        buckets[i] = np.array(buckets[i])
                        bucket_variances[i] = np.array(bucket_variances[i])
                        if len(bucket_variances[i]) != len(buckets[i]):
                            raise ValueError("len(bucket_variances[{i}]) != len(buckets[{i}])".format(i=i))

                if depth == 0:
                    uncertainty_for_class_1 = None
                    resort_idx_for_class_1 = None
                    loss_A_B_stats = None
                else:
                    pass

                y_proba, y_evidence, alpha = k_fold_est.fit(cur_layer_x_train, cur_layer_y_train, buckets,
                                                            bucket_variances,
                                                            resort_idx_for_class_1, uncertainty_for_class_1,
                                                            loss_A_B_stats)
                self.alpha_list.append(alpha)
                y_pred_cur_k_fold_forest = self.category[np.argmax(y_proba, axis=1)]
                cur_k_fold_forest_cm = confusion_matrix(y_train, y_pred_cur_k_fold_forest)
                cur_k_fold_forest_cm = cur_k_fold_forest_cm.astype('float') / cur_k_fold_forest_cm.sum(axis=1)[:,
                                                                              np.newaxis]

                difference_cm_line0 = y_proba - cur_k_fold_forest_cm[0]
                difference_cm_line1 = y_proba - cur_k_fold_forest_cm[1]

                norm_difference_cm_line0 = np.linalg.norm(difference_cm_line0, axis=1)
                norm_difference_cm_line1 = np.linalg.norm(difference_cm_line1, axis=1)

                enhancement_vector_cm[:, index * n_classes_:index * n_classes_ + n_classes_] = np.vstack(
                    (norm_difference_cm_line0, norm_difference_cm_line1)).T

                current_layer.add_est(k_fold_est)

                current_layer.confusion_matrices.append(cur_k_fold_forest_cm)

                y_train_probas[:, index * n_classes_:index * n_classes_ + n_classes_] += y_proba
                y_train_evidences[:, index * n_classes_:index * n_classes_ + n_classes_] += y_evidence

                y_train_probas_avg += y_proba

            y_train_probas_avg /= len(self.estimator_configs)
            y_train_pred = self.category[np.argmax(y_train_probas_avg, axis=1)]

            if depth == 0:
                y_train_probas_per_layer_per_forest.append(y_train_probas.tolist())
                y_train_evidence_per_layer_per_forest.append(y_train_evidences.tolist())

                y_train_probas_per_layer.append(y_train_probas_avg)
                y_train_pred_per_layer.append(y_train_pred)




            else:
                y_train_probas_per_layer_per_forest.append(y_train_probas.tolist())
                y_train_evidence_per_layer_per_forest.append(y_train_evidences.tolist())

                y_train_probas_per_layer.append(y_train_probas_avg)
                y_train_pred_per_layer.append(y_train_pred)

            if evaluate.__name__ == "roc_auc":
                current_evaluation = evaluate(y_train, y_train_probas_summed)
            elif evaluate.__name__ == "generalized_performance":
                current_evaluation = evaluate(y_train, y_train_pred, y_train_probas_summed)
            else:
                current_evaluation = evaluate(y_train, y_train_pred)

            if depth == 0:
                self.per_layer_generalized_performance_list = []
                self.per_layer_generalized_performance_list.append(current_evaluation)
            else:
                self.per_layer_generalized_performance_list.append(current_evaluation)

            cur_layer_confusion_matrices = confusion_matrix(y_train, y_train_pred)
            cur_layer_confusion_matrices = cur_layer_confusion_matrices.astype(
                'float') / cur_layer_confusion_matrices.sum(axis=1)[:, np.newaxis]
            difference_cm_line0 = y_train_probas_avg - cur_layer_confusion_matrices[0]
            difference_cm_line1 = y_train_probas_avg - cur_layer_confusion_matrices[1]
            norm_difference_cm_line0 = np.linalg.norm(difference_cm_line0, axis=1)
            norm_difference_cm_line1 = np.linalg.norm(difference_cm_line1, axis=1)
            enhancement_vector_cm_mean = np.vstack((norm_difference_cm_line0, norm_difference_cm_line1)).T

            current_layer.confusion_matrix_mean = cur_layer_confusion_matrices

            if self.if_stacking:

                if self.enhancement_vector_method == "mean_confusion_matrix":
                    enhanced_vector_per_layer.append(enhancement_vector_cm_mean)

                elif self.enhancement_vector_method == "confusion_matrix":

                    enhanced_vector_per_layer.append(enhancement_vector_cm)


                elif self.enhancement_vector_method == "class_proba_vector":

                    enhanced_vector_per_layer.append(y_train_probas)
                elif self.enhancement_vector_method == "trusted_enhancement_vector":

                    all_instance_KL, all_instance_u = self.calculate_KL_u_use_evidence(
                        y_train_probas_per_layer_per_forest, n_classes_)
                    if all_instance_KL is not None and all_instance_u is not None:
                        if self.use_u_KL_method == "u":
                            enhanced_vector_cur_layer = np.hstack((all_instance_u, y_train_probas))
                        elif self.use_u_KL_method == "KL":
                            enhanced_vector_cur_layer = np.hstack((all_instance_KL, y_train_probas))
                        elif self.use_u_KL_method == "all":
                            enhanced_vector_cur_layer = np.hstack((all_instance_KL, all_instance_u, y_train_probas))
                        else:
                            raise ValueError("use_u_KL_method must be u or KL or all")

                        # enhanced_vector_cur_layer = np.hstack((all_instance_KL, all_instance_u, y_train_probas))
                        # enhanced_vector_cur_layer = np.hstack((all_instance_KL, y_train_probas))
                        enhanced_vector_per_layer.append(enhanced_vector_cur_layer)
                        print(f"Final enhanced_vector_cur_layer type: {type(enhanced_vector_cur_layer)}")
                        if isinstance(enhanced_vector_cur_layer, np.ndarray):
                            print(f"enhanced_vector_cur_layer shape: {enhanced_vector_cur_layer.shape}")
                        else:
                            print(f"enhanced_vector_cur_layer content length: {len(enhanced_vector_cur_layer)}")
                    else:
                        raise ValueError("all_instance_KL is None or all_instance_u is None")
                else:
                    raise ValueError(
                        "enhancement_vector_method must be mean_confusion_matrix, confusion_matrix or class_proba")
            else:

                if self.enhancement_vector_method == "mean_confusion_matrix":

                    enhanced_vector_per_layer.append(enhancement_vector_cm_mean)


                elif self.enhancement_vector_method == "confusion_matrix":

                    enhanced_vector_per_layer.append(enhancement_vector_cm)

                elif self.enhancement_vector_method == "class_proba_vector":

                    enhanced_vector_per_layer.append(y_train_probas)
                elif self.enhancement_vector_method == "trusted_enhancement_vector":

                    all_instance_KL, all_instance_u = self.calculate_KL_u_use_evidence(
                        y_train_probas_per_layer_per_forest, n_classes_)
                    if all_instance_KL is not None and all_instance_u is not None:

                        if self.use_u_KL_method == "u":
                            enhanced_vector_cur_layer = np.hstack((all_instance_u, y_train_probas))
                        elif self.use_u_KL_method == "KL":
                            enhanced_vector_cur_layer = np.hstack((all_instance_KL, y_train_probas))
                        elif self.use_u_KL_method == "all":
                            enhanced_vector_cur_layer = np.hstack((all_instance_KL, all_instance_u, y_train_probas))
                        else:
                            raise ValueError("use_u_KL_method must be u or KL or all")
                        enhanced_vector_per_layer.append(enhanced_vector_cur_layer)

                        if isinstance(enhanced_vector_cur_layer, np.ndarray):
                            pass
                        else:
                            print(f"enhanced_vector_cur_layer content length: {len(enhanced_vector_cur_layer)}")

                    else:
                        raise ValueError("all_instance_KL is None or all_instance_u is None")
                else:
                    raise ValueError(
                        "enhancement_vector_method must be mean_confusion_matrix, confusion_matrix or class_proba")

            if current_evaluation > best_layer_evaluation:
                best_layer_id = current_layer.layer_id
                best_layer_evaluation = current_evaluation
            LOGGER.info(
                "The evaluation[{}] of layer_{} is {:.4f}".format(evaluate.__name__, depth, current_evaluation))

            self.layers.append(current_layer)
            print("num_layers:", len(self.layers))

            if current_layer.layer_id - best_layer_id >= self.early_stop_rounds:
                self.layers = self.layers[0:best_layer_id + 1]

                LOGGER.info(
                    "best_layer: {}, current_layer:{}, save layers: {}".format(best_layer_id, current_layer.layer_id,
                                                                               len(self.layers)))
                break
                pass
            depth += 1

        if self.if_save_model:
            pickle.dump(self, open("gc.pkl", "wb"))
        LOGGER.info("training finish...")

    def predict(self, x):
        prob = self.predict_proba(x)
        label = self.category[np.argmax(prob, axis=1)]
        return label

    def predict_weighted_layers(self, x):
        prob = self.predict_proba_weighted_layers(x)
        label = self.category[np.argmax(prob, axis=1)]
        return label

    def predict_proba_weighted_layers(self, x, method="sum"):
        n_layers = len(self.layers)

        if method == "sum":
            weights = self.per_layer_generalized_performance_list / np.sum(self.per_layer_generalized_performance_list)
        elif method == "ln":
            weights = np.array(
                [np.log(performance / (1 - performance)) for performance in
                 self.per_layer_generalized_performance_list])
            weights = weights / np.sum(weights)

        else:
            raise ValueError("method must be sum or ln")

        x_test = x.copy()
        x_test = x_test.reshape((x.shape[0], -1))
        n_feature = x_test.shape[1]

        x_test_proba = None

        x_test_probas = []
        summed_probas = np.zeros((x_test.shape[0], len(self.category)))

        enhanced_vectors = []

        for layer_index in range(n_layers):

            x_test_cur_layer = None

            if layer_index == 0:
                x_test_cur_layer = x_test
            else:
                if self.if_stacking == False:
                    x_test_cur_layer = np.hstack((x_test, enhanced_vectors[layer_index - 1]))
                else:
                    enhanced_vector = enhanced_vectors[0]
                    for i in range(1, layer_index):
                        enhanced_vector = np.hstack((enhanced_vector, enhanced_vectors[i]))
                    x_test_cur_layer = np.hstack((x_test, enhanced_vector))

            if layer_index == n_layers - 1:
                print("last_layer_index", layer_index)
                x_test_proba = self.layers[layer_index]._predict_proba(x_test_cur_layer)
                summed_probas += weights[layer_index] * x_test_proba
            else:

                x_test_proba = self.layers[layer_index].predict_proba(x_test_cur_layer)
                summed_probas += weights[layer_index] * self.layers[layer_index]._predict_proba(x_test_cur_layer)

                if self.enhancement_vector_method == "class_proba_vector":
                    enhanced_vectors.append(x_test_proba)
                    x_test_probas.append(x_test_proba)
                    # print("x_test_proba", x_test_proba)
                    # print("x_test_probas:", x_test_probas)
                    #
                    # print("x_test_proba.shape:", x_test_proba.shape)
                    # print("x_test_probas.shape:", x_test_probas.shape)
                elif self.enhancement_vector_method == "trusted_enhancement_vector":
                    x_test_probas.append(x_test_proba)
                    B, u = self.calculate_KL_u_use_evidence(x_test_probas, len(self.category))
                    # enhanced_vector_cur_layer = np.hstack((B, x_test_proba))
                    # enhanced_vector_cur_layer = np.hstack((u, x_test_proba))

                    if self.use_u_KL_method == "u":
                        enhanced_vector_cur_layer = np.hstack((u, x_test_proba))
                    elif self.use_u_KL_method == "KL":
                        enhanced_vector_cur_layer = np.hstack((B, x_test_proba))
                    elif self.use_u_KL_method == "all":
                        enhanced_vector_cur_layer = np.hstack((B, u, x_test_proba))
                    else:
                        raise ValueError("use_u_KL_method must be u or KL or all")

                    enhanced_vectors.append(enhanced_vector_cur_layer)

                else:
                    raise ValueError(
                        "enhancement_vector_method must be class_proba_vector")

        return summed_probas

    def predict_proba(self, x):

        n_layers = len(self.layers)
        x_test = x.copy()
        x_test = x_test.reshape((x.shape[0], -1))
        n_feature = x_test.shape[1]

        x_test_proba = None

        x_test_probas = []
        summed_probas = np.zeros((x_test.shape[0], len(self.category)))

        enhanced_vectors = []

        for layer_index in range(n_layers):

            x_test_cur_layer = None

            if layer_index == 0:
                x_test_cur_layer = x_test
            else:
                if self.if_stacking == False:
                    x_test_cur_layer = np.hstack((x_test, enhanced_vectors[layer_index - 1]))
                else:
                    enhanced_vector = enhanced_vectors[0]
                    for i in range(1, layer_index):
                        enhanced_vector = np.hstack((enhanced_vector, enhanced_vectors[i]))
                    x_test_cur_layer = np.hstack((x_test, enhanced_vector))

            if layer_index == n_layers - 1:
                print("last_layer_index", layer_index)
                x_test_proba = self.layers[layer_index]._predict_proba(x_test_cur_layer)
                summed_probas += x_test_proba


            else:
                x_test_proba = self.layers[layer_index].predict_proba(x_test_cur_layer)
                summed_probas += self.layers[layer_index]._predict_proba(x_test_cur_layer)

                if self.enhancement_vector_method == "mean_confusion_matrix":

                    x_test_proba = self.layers[layer_index]._predict_proba(x_test_cur_layer)
                    cur_layer_confusion_matrices = self.layers[layer_index].confusion_matrix_mean
                    difference_cm_line0 = x_test_proba - cur_layer_confusion_matrices[0]
                    difference_cm_line1 = x_test_proba - cur_layer_confusion_matrices[1]
                    norm_difference_cm_line0 = np.linalg.norm(difference_cm_line0, axis=1)
                    norm_difference_cm_line1 = np.linalg.norm(difference_cm_line1, axis=1)
                    enhancement_vector_cm_mean = np.vstack((norm_difference_cm_line0, norm_difference_cm_line1)).T

                    enhanced_vectors.append(enhancement_vector_cm_mean)
                    x_test_probas.append(x_test_proba)

                elif self.enhancement_vector_method == "confusion_matrix":
                    cur_layer_confusion_matrices = self.layers[layer_index].confusion_matrices
                    enhancement_vector_cm = np.zeros(
                        (x_test.shape[0], len(self.estimator_configs) * len(self.category)))
                    n_label = self.category.shape[0]

                    for i in range(len(self.layers[layer_index].estimators)):
                        cur_k_fold_forest_cm = cur_layer_confusion_matrices[i]

                        difference_cm_line0 = x_test_proba[:,
                                              i * n_label: i * n_label + n_label] - cur_k_fold_forest_cm[0, :]
                        difference_cm_line1 = x_test_proba[:,
                                              i * n_label: i * n_label + n_label] - cur_k_fold_forest_cm[1, :]

                        norm_difference_cm_line0 = np.linalg.norm(difference_cm_line0, axis=1)
                        norm_difference_cm_line1 = np.linalg.norm(difference_cm_line1, axis=1)
                        enhancement_vector_cm[:, i * n_label:i * n_label + n_label] = np.vstack(
                            (norm_difference_cm_line0, norm_difference_cm_line1)).T

                        enhanced_vectors.append(enhancement_vector_cm)
                        x_test_probas.append(x_test_proba)


                elif self.enhancement_vector_method == "class_proba_vector":

                    enhanced_vectors.append(x_test_proba)
                    x_test_probas.append(x_test_proba)

                elif self.enhancement_vector_method == "trusted_enhancement_vector":
                    x_test_probas.append(x_test_proba)
                    B, u = self.calculate_KL_u_use_evidence(x_test_probas, len(self.category))

                    if self.use_u_KL_method == "u":
                        enhanced_vector_cur_layer = np.hstack((u, x_test_proba))
                    elif self.use_u_KL_method == "KL":
                        enhanced_vector_cur_layer = np.hstack((B, x_test_proba))
                    elif self.use_u_KL_method == "all":
                        enhanced_vector_cur_layer = np.hstack((B, u, x_test_proba))
                    else:
                        raise ValueError("use_u_KL_method must be u or KL or all")

                    enhanced_vectors.append(enhanced_vector_cur_layer)
                else:
                    raise ValueError(
                        "enhancement_vector_method must be mean_confusion_matrix, confusion_matrix or class_proba_vector")
        return summed_probas / n_layers

    def preprocess(self, x_train, y_train):
        x_train = x_train.reshape((x_train.shape[0], -1))
        category = np.unique(y_train)
        self.category = category

        n_feature = x_train.shape[1]
        n_label = len(np.unique(y_train))
        LOGGER.info("Begin to train....")
        LOGGER.info("the shape of training samples: {}".format(x_train.shape))
        LOGGER.info("use {} as training evaluation".format(self.train_evaluation.__name__))
        LOGGER.info("stacking: {}, save model: {}".format(self.if_stacking, self.if_save_model))
        return x_train, n_feature, n_label

    def predict_proba_last_layer(self, x):
        x_test = x.copy()
        x_test = x_test.reshape((x.shape[0], -1))
        n_feature = x_test.shape[1]

        x_test_proba = None

        x_test_preds = []
        x_test_probas = []
        summed_probas = np.zeros((x_test.shape[0], len(self.category)))

        enhanced_vectors = []

        for layer_index in range(len(self.layers)):

            x_test_cur_layer = None

            if layer_index == 0:
                x_test_cur_layer = x_test
            else:
                if self.if_stacking == False:
                    x_test_cur_layer = np.hstack((x_test, enhanced_vectors[layer_index - 1]))
                else:
                    enhanced_vector = enhanced_vectors[0]
                    for i in range(1, layer_index - 1):
                        enhanced_vector = np.hstack((enhanced_vector, enhanced_vectors[i]))
                    x_test_cur_layer = np.hstack((x_test, enhanced_vector))

            if layer_index == len(self.layers) - 1:
                print("last_layer_index", layer_index)
                x_test_proba = self.layers[layer_index]._predict_proba(x_test_cur_layer)
                summed_probas += x_test_proba


            else:
                x_test_proba = self.layers[layer_index].predict_proba(x_test_cur_layer)

                if self.enhancement_vector_method == "mean_confusion_matrix":

                    x_test_proba = self.layers[layer_index]._predict_proba(x_test_cur_layer)
                    cur_layer_confusion_matrices = self.layers[layer_index].confusion_matrix_mean
                    difference_cm_line0 = x_test_proba - cur_layer_confusion_matrices[0]
                    difference_cm_line1 = x_test_proba - cur_layer_confusion_matrices[1]
                    norm_difference_cm_line0 = np.linalg.norm(difference_cm_line0, axis=1)
                    norm_difference_cm_line1 = np.linalg.norm(difference_cm_line1, axis=1)
                    enhancement_vector_cm_mean = np.vstack((norm_difference_cm_line0, norm_difference_cm_line1)).T

                    enhanced_vectors.append(enhancement_vector_cm_mean)
                    x_test_probas.append(x_test_proba)

                elif self.enhancement_vector_method == "confusion_matrix":
                    cur_layer_confusion_matrices = self.layers[layer_index].confusion_matrices
                    enhancement_vector_cm = np.zeros(
                        (x_test.shape[0], len(self.estimator_configs) * len(self.category)))
                    n_label = self.category.shape[0]

                    for i in range(len(self.layers[layer_index].estimators)):
                        cur_k_fold_forest_cm = cur_layer_confusion_matrices[i]
                        difference_cm_line0 = x_test_proba[:,
                                              i * n_label: i * n_label + n_label] - cur_k_fold_forest_cm[0, :]
                        difference_cm_line1 = x_test_proba[:,
                                              i * n_label: i * n_label + n_label] - cur_k_fold_forest_cm[1, :]
                        norm_difference_cm_line0 = np.linalg.norm(difference_cm_line0, axis=1)
                        norm_difference_cm_line1 = np.linalg.norm(difference_cm_line1, axis=1)
                        enhancement_vector_cm[:, i * n_label:i * n_label + n_label] = np.vstack(
                            (norm_difference_cm_line0, norm_difference_cm_line1)).T

                        enhanced_vectors.append(enhancement_vector_cm)
                        x_test_probas.append(x_test_proba)


                elif self.enhancement_vector_method == "class_proba_vector":

                    enhanced_vectors.append(x_test_proba)
                    x_test_probas.append(x_test_proba)
                else:
                    raise ValueError(
                        "enhancement_vector_method must be mean_confusion_matrix, confusion_matrix or class_proba_vector")
        return summed_probas
