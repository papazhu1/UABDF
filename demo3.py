import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold

from UABDF import UncertaintyAwareBalancedDeepForest
from evaluation import gmean


model_dict = {}
model_dict["rf"] = "RandomForestClassifier"
model_dict["et"] = "ExtraTreesClassifier"
model_dict["spe"] = "SelfPacedEnsembleClassifier"
model_dict["bc"] = "BalancedCascadeClassifier"
model_dict["brf"] = "BalancedRandomForestClassifier"
model_dict["ee"] = "EasyEnsembleClassifier"
model_dict["rusb"] = "RUSBoostClassifier"
model_dict["be"] = "BalancedEnsembleClassifier"

use_u_KL_method_list = ["u", "KL", "all"]
use_vector_list = ["class_proba_vector", "trusted_enhancement_vector"]


def get_config(lamb=0.5):
    config = {}
    config["enhancement_vector_method"] = use_vector_list[1]
    config["use_u_KL_method"] = use_u_KL_method_list[2]
    config["random_state"] = np.random.randint(0, 10000)
    config["max_layers"] = 5
    config["early_stop_rounds"] = 1
    config["if_stacking"] = False
    config["if_save_model"] = False
    config["train_evaluation"] = gmean
    config["estimator_configs"] = []
    config["n_estimators"] = 50
    config["lamb"] = lamb

    for _ in range(1):
        for _ in range(4):
            config["estimator_configs"].append(
                {
                    "n_fold": 5,
                    "type": model_dict["et"],
                    "n_estimators": config["n_estimators"],
                    "n_jobs": -1,
                }
            )

    return config


def load_npz_dataset(npz_path):
    """
    加载本地保存的 .npz 数据集。

    要求 .npz 内部字段为:
        data
        label

    兼容你之前的调用方式:
        dataset = np.load("zenodo_1/libras_move.npz")
        X, y = dataset["data"], dataset["label"]
        y = np.where(y == -1, 0, y)
    """
    dataset = np.load(npz_path)

    print("npz 文件字段:", dataset.files)

    if "data" not in dataset.files:
        raise ValueError(f"{npz_path} 中缺少 data 字段")

    if "label" not in dataset.files:
        raise ValueError(f"{npz_path} 中缺少 label 字段")

    X = dataset["data"]
    y = dataset["label"]

    # 保证 y 是一维
    y = np.asarray(y).ravel()

    # 如果标签里有 -1，统一转成 0
    y = np.where(y == -1, 0, y)

    # 保证标签是 int 类型
    y = y.astype(int)

    # 保证 X 是 numpy 数组
    X = np.asarray(X)

    return X, y


def save_current_results(out_path, fold_records, dataset_name):
    fold_df = pd.DataFrame(fold_records)

    if len(fold_df) > 0:
        final_mean = pd.DataFrame(
            [
                {
                    "row_type": "final_mean",
                    "dataset": dataset_name,
                    "repeat": "",
                    "fold": "",
                    "auc": fold_df["auc"].mean(),
                    "f1_macro": fold_df["f1_macro"].mean(),
                }
            ]
        )

        result_df = pd.concat([fold_df, final_mean], ignore_index=True)

    else:
        result_df = pd.DataFrame(
            columns=[
                "row_type",
                "dataset",
                "repeat",
                "fold",
                "auc",
                "f1_macro",
            ]
        )

    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")


def run_dataset(dataset_name, npz_path, save_dir):
    X, y = load_npz_dataset(npz_path)

    print("dataset_name:", dataset_name)
    print("npz_path:", npz_path)

    pos_count = int(np.sum(y == 1))
    neg_count = int(np.sum(y == 0))

    print(f"{dataset_name} total samples: {len(y)}, pos: {pos_count}, neg: {neg_count}")
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("label unique:", np.unique(y, return_counts=True))

    # 检查是否是二分类
    unique_labels = np.unique(y)
    if len(unique_labels) != 2:
        raise ValueError(
            f"{dataset_name} 不是二分类数据集，当前标签为: {unique_labels}"
        )

    # 检查是否只有 0 和 1
    if not np.array_equal(unique_labels, np.array([0, 1])):
        raise ValueError(
            f"{dataset_name} 的标签不是 0/1，当前标签为: {unique_labels}"
        )

    rskf = RepeatedStratifiedKFold(
        n_splits=5,
        n_repeats=10,
        random_state=42
    )

    os.makedirs(save_dir, exist_ok=True)

    out_path = os.path.join(
        save_dir,
        f"{dataset_name}_50folds_with_mean.csv"
    )

    fold_records = []

    # 先建一个空文件/表头
    save_current_results(out_path, fold_records, dataset_name)

    for split_idx, (train_index, test_index) in enumerate(rskf.split(X, y), start=1):
        repeat_idx = (split_idx - 1) // 5 + 1
        fold_idx = (split_idx - 1) % 5 + 1

        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        model = UncertaintyAwareBalancedDeepForest(get_config())
        model.fit(X_train, y_train)

        pred_proba = model.predict_proba_weighted_layers(X_test)

        if np.isnan(pred_proba).any():
            raise ValueError(
                f"{dataset_name} repeat={repeat_idx}, fold={fold_idx} 的预测概率中出现 NaN"
            )

        y_pred = model.category[np.argmax(pred_proba, axis=1)]

        # 默认第 1 列是正类概率
        y_score = pred_proba[:, 1]

        fold_result = {
            "row_type": "fold",
            "dataset": dataset_name,
            "repeat": repeat_idx,
            "fold": fold_idx,
            "auc": roc_auc_score(y_test, y_score),
            "f1_macro": f1_score(y_test, y_pred, average="macro"),
        }

        fold_records.append(fold_result)

        # 每一折跑完后，立刻把“当前所有折 + 当前平均”写回同一个文件
        save_current_results(out_path, fold_records, dataset_name)

        print(
            f"{dataset_name} | repeat={repeat_idx} fold={fold_idx} | "
            f"auc={fold_result['auc']:.6f}, "
            f"f1_macro={fold_result['f1_macro']:.6f} | "
            f"saved to {out_path}"
        )

    print(f"{dataset_name} finished. Final results saved to: {out_path}")

    return out_path


if __name__ == "__main__":
    dataset_files = {
        "yeast1": "zenodo_1/yeast1.npz",
        "yeast2": "zenodo_1/yeast2.npz",
        "yeast3": "zenodo_1/yeast3.npz",
        "yeast4": "zenodo_1/yeast4.npz",

        # 如果你也保存了 yeast5，就取消下面这一行注释
        # "yeast5": "zenodo_1/yeast5.npz",

        # 如果还有其他 npz，也按这个格式继续加
        # "libras_move": "zenodo_1/libras_move.npz",
        # "abalone": "zenodo_1/abalone.npz",
    }

    save_dir = "cv_results_single_file"

    for dataset_name, npz_path in dataset_files.items():
        run_dataset(dataset_name, npz_path, save_dir)