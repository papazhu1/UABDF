import numpy as np
import csv
from scipy.stats import norm
import matplotlib.pyplot as plt

# from model.test_rf_leaf_getting import leaf_indices


def error_to_hardness(x):
    return np.tanh(6 * (x - 0.5)) / 2 + 1


def append_results_to_csv(results,
                          csv_file="gcForest_sampled_res.csv"):
    """
    Appends a dictionary of results to a CSV file, ensuring required fields are present.

    Parameters:
    results (dict): The results dictionary to append to the CSV file.
    csv_file (str): The path to the CSV file.
    """
    required_keys = [
        "layer", "accuracy", "f1_score", "auc", "gmean", "sensitivity", "specificity",
        "aupr", "generalized_performance", "true_neg", "false_pos", "false_neg", "true_pos", "recall_0",
        "recall_1", "precision_0", "precision_1", "imbalanced_rate", "neg_num", "pos_num"
    ]


    for key in required_keys:
        if key not in results:
            raise ValueError(f"Missing required key in results dictionary: {key}")


    file_exists = False
    try:
        with open(csv_file, 'r') as file:
            file_exists = True
    except FileNotFoundError:
        file_exists = False


    with open(csv_file, mode='a', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=required_keys)
        if (not file_exists) or (file.tell() == 0):
            writer.writeheader()
        writer.writerow(results)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def uncertainty_to_probability_by_power(uncertainties, power=5):

    scores = np.power(uncertainties, power)
    scores = scores / np.max(scores)
    probabilities = softmax(scores)
    return probabilities


def uncertainty_to_probability_by_sum(uncertainties):
    uncertainties = np.array(uncertainties)

    sum_uncertainty = np.sum(uncertainties)
    if sum_uncertainty == 0:
        probabilities = np.array([1 / len(uncertainties) for _ in uncertainties])
    else:
        probabilities = uncertainties / sum_uncertainty
    return probabilities


def calculate_bin_capacities(num_bins, t, alpha=0.1):
    indices = np.arange(num_bins)
    raw_capacities = np.exp(-alpha * indices * t)


    normalized_capacities = raw_capacities / raw_capacities.sum()
    return normalized_capacities



def train_K_fold_paralleling(args):
    use_uncertainty = True
    use_bucket = True
    use_resample = True

    train_index, val_index, X, y, est, buckets, bucket_variances, index_1_sorted, uncertainty_1_sorted, loss_A_B_stats = args
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]
    # if buckets is not None:
    #     for i in range(len(buckets)):
    #         print("len(buckets_", i, "): ", len(buckets[i]))
    #     for i in range(len(bucket_variances)):
    #         print("len(bucket_variances_" , i, "): ", len(bucket_variances[i]))

    sampled_1_index = None
    if index_1_sorted is None or uncertainty_1_sorted[0] == np.nan:
        pass
    else:
        in_train_index_1 = np.in1d(index_1_sorted, train_index)
        if isinstance(in_train_index_1, np.ndarray) and in_train_index_1.dtype == bool:
            in_train_index_1 = np.where(in_train_index_1)[0]

        index_1_sorted = np.array(index_1_sorted)
        uncertainty_1_sorted = np.array(uncertainty_1_sorted)
        index_1_sorted = index_1_sorted[in_train_index_1]
        uncertainty_1_sorted = uncertainty_1_sorted[in_train_index_1]

        probability = uncertainty_to_probability_by_sum(uncertainty_1_sorted)
        if np.isnan(probability).any():
            print("probability[0] == np.NAN")
        else:
            if use_uncertainty is True:
                sampled_1_index = np.random.choice(index_1_sorted, int(len(index_1_sorted)), replace=True,
                                                   p=probability)

            else:
                sampled_1_index = np.random.choice(index_1_sorted, int(len(index_1_sorted)), replace=True)
    if buckets is not None:
        index_0 = np.where(y_train == 0)[0]
        index_1 = np.where(y_train == 1)[0]


        for i in range(len(buckets)):


            in_train_index = np.in1d(buckets[i], train_index)
            if isinstance(in_train_index, np.ndarray) and in_train_index.dtype == bool:
                in_train_index = np.where(in_train_index)[0]
            else:
                raise ValueError("in_train_index is not a boolean array.")

            # print("len(buckets[i])", len(buckets[i]))
            # print("len(bucket_variances[i]", len(bucket_variances[i]))

            buckets[i] = buckets[i][in_train_index]
            if bucket_variances is not None:
                bucket_variances[i] = bucket_variances[i][in_train_index]

                # print("in_train_index", in_train_index)
                # print("len(buckets[i])", len(buckets[i]))
                # print("len(bucket_variances[i])", len(bucket_variances[i]))


        index_0_selected = []
        num_empty_buckets = 0

        for i in range(len(buckets)):
            if len(buckets[i]) == 0:
                num_empty_buckets += 1

        # for j in range(len(buckets)):
        #     print(f"Bucket {j} has {len(buckets[j])} samples.")
        # print("\n")
        # for j in range(len(buckets)):
        #     print(f"Bucket {j} sampled {int(len(index_1) // (len(buckets) - num_empty_buckets))} samples.")

        for i in range(len(buckets)):

            if len(buckets[i]) == 0:
                continue

            if bucket_variances is not None:

                probabilities = uncertainty_to_probability_by_sum(bucket_variances[i])
                # 用高斯拟合
                # 使用 norm 对大类的 loss_instances 进行拟合
                # mu, sigma = norm.fit(bucket_variances[i])
                #
                # # 计算每个大类样本的概率密度
                # weights = norm.pdf(bucket_variances[i], mu, sigma)

                # 将大类的采样权重归一化
                # weights /= np.sum(weights)
                # weights = weights.ravel()

                # 统计一下每个 bucket 的桶容量，统计一下每个桶中要被采样的数量


                # assert len(probabilities) == len(buckets[i])
                assert len(bucket_variances[i]) == len(buckets[i])
                assert len(bucket_variances[i]) == len(probabilities)
                assert len(probabilities) == len(buckets[i])

                if use_uncertainty is True:
                    actual_sample_size = min(int(len(index_1) // (len(buckets) - num_empty_buckets)), len(buckets[i]))
                    sampled_cur_bucket = np.random.choice(buckets[i],
                                                          actual_sample_size,
                                                          replace=False,
                                                          p=probabilities)
                    # sampled_cur_bucket = np.random.choice(buckets[i],
                    #                                   int(len(index_1) // (len(buckets) - num_empty_buckets)),
                    #                                   replace=False,
                    #                                   p=weights)
                else:
                    sampled_cur_bucket = np.random.choice(buckets[i],
                                                          int(len(index_1) // (len(buckets) - num_empty_buckets)),
                                                          replace=False)
                index_0_selected.extend(sampled_cur_bucket)
            else:
                sampled_cur_bucket = np.random.choice(buckets[i],
                                                      int(len(index_1) // (len(buckets) - num_empty_buckets)),
                                                      replace=False)
                index_0_selected.extend(sampled_cur_bucket)

        # 假设 index_0_selected 是挑选的 0 类样本的索引
        index_0_all = np.where(y == 0)[0]  # 所有 0 类样本的索引（假设 y_train 是标签）
        index_0_not_selected = np.setdiff1d(index_0_all, index_0_selected)  # 未被选中的 0 类样本的索引

        # 绘制图形
        # plt.figure(figsize=(6, 6))
        #
        #
        #
        # # 画出未挑选的 0 类样本，颜色为白色
        # plt.scatter(X[index_0_not_selected, 0], X[index_0_not_selected, 1], c='w', alpha=0.6, edgecolor='k',
        #             label="Not selected 0 samples")
        #
        # # 画出挑选的 0 类样本，颜色为蓝色
        # plt.scatter(X[index_0_selected, 0], X[index_0_selected, 1], c='b', alpha=0.6, edgecolor='k',
        #             label="Selected 0 samples")
        #
        # plt.title(f"Selected vs Not Selected 0 samples")
        # plt.legend()
        # plt.show()

        # print("len(index_0_selected): ", len(index_0_selected))
        # print("len(index_0_not_selected): ", len(index_0_not_selected))
        # print("len(index_0_all): ", len(index_0_all))



        if use_bucket is False:

            all_majority_samples = []
            for i in range(len(buckets)):
                all_majority_samples.extend(buckets[i])

            index_0_selected = np.random.choice(all_majority_samples, int(len(index_1)), replace=False)








        index_0_selected = np.array(index_0_selected)

        if sampled_1_index is not None:
            new_train_index = np.concatenate([index_0_selected, index_1, sampled_1_index])
        else:
            new_train_index = np.concatenate([index_0_selected, index_1])

        new_train_index = new_train_index.astype(int)
        X_train, y_train = X[new_train_index], y[new_train_index]

    if use_resample is False:
        X_train, y_train = X[train_index], y[train_index]

    # 画出loss，A，B的图像
    if loss_A_B_stats is not None:
        index, loss, A, B = loss_A_B_stats["index"], loss_A_B_stats["loss"], loss_A_B_stats["A"], loss_A_B_stats["B"]

        # print(index)
        # print(type(index[0]))
        # print("len(X)", len(X))
        # print("len(index)", len(index))
        # print("len(X_train", len(X_train))
        # plt.figure(figsize=(6, 6))
        # scatter = plt.scatter(X[index, 0], X[index, 1], c=loss, cmap='coolwarm', alpha=0.6, edgecolor='k')
        # plt.title(f"Classification loss")
        # plt.colorbar(scatter, orientation='vertical', label='Predicted Class')
        # plt.show()
        #
        # plt.figure(figsize=(6, 6))
        # scatter = plt.scatter(X[index, 0], X[index, 1], c=A, cmap='coolwarm', alpha=0.6, edgecolor='k')
        # plt.title(f"Classification A")
        # plt.colorbar(scatter, orientation='vertical', label='Predicted Class')
        # plt.show()
        #
        # plt.figure(figsize=(6, 6))
        # scatter = plt.scatter(X[index, 0], X[index, 1], c=B, cmap='coolwarm', alpha=0.6, edgecolor='k')
        # plt.title(f"Classification B")
        # plt.colorbar(scatter, orientation='vertical', label='Predicted Class')
        # plt.show()





    est.fit(X_train, y_train)


    y_proba = est.predict_proba(X_val)

    # print(y_proba.shape)
    # print(type(y_proba))

    y_evidence = np.zeros(y_proba.shape)

    for val_idx, instance in enumerate(X_val):
        instance = instance.reshape(1, -1)
        leaf_indices = est.apply(instance).flatten()
        total_samples_per_class = np.zeros(est.n_classes_)
        for tree_idx, tree in enumerate(est.estimators_):
            tree_structure = tree.tree_
            leaf_index = leaf_indices[tree_idx]
            samples_per_class = tree_structure.value[leaf_index, 0]
            total_samples_per_class += samples_per_class

        y_evidence[val_idx] = total_samples_per_class

    y_pred = est.predict(X_val)

    return [est, y_proba, y_pred, val_index, y_evidence]


def predict_proba_parallel(ars):
    estimator, x_test = ars
    return estimator.predict_proba(x_test)
