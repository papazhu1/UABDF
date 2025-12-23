import os

from imbens.datasets import fetch_datasets
from imbens.metrics import geometric_mean_score, sensitivity_score, specificity_score
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score, \
    precision_score
from sklearn.model_selection import StratifiedKFold
import data_util
from UADF import UncertaintyAwareDeepForest
from data_util import *
from evaluation import f1_macro, gmean
import shap
import inspect

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

# 加载数据集
def load_data(dataset_name):
    dataset = fetch_datasets()[dataset_name]
    X, y = dataset['data'], dataset['target']
    y = np.where(y == -1, 0, y)  # 将 -1 类别转换为 0
    print(f"Original class distribution: {Counter(dataset['target'])}")
    print(f"Transformed class distribution: {Counter(y)}")
    return X, y

def get_config(lamb = 0.5):
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

    for i in range(1):
        config["estimator_configs"].append(
            {"n_fold": 5, "type": model_dict["et"], "n_estimators": config["n_estimators"], "n_jobs": -1})
        config["estimator_configs"].append(
            {"n_fold": 5, "type": model_dict["et"], "n_estimators": config["n_estimators"], "n_jobs": -1})
        config["estimator_configs"].append(
            {"n_fold": 5, "type": model_dict["et"], "n_estimators": config["n_estimators"], "n_jobs": -1})
        config["estimator_configs"].append(
            {"n_fold": 5, "type": model_dict["et"], "n_estimators": config["n_estimators"], "n_jobs": -1})
    return config




if __name__ == "__main__":

    # X = np.load("../dataset/dataset/drug_cell_feature.npy", allow_pickle=True)
    # y = np.load("../dataset/dataset/drug_cell_label.npy", allow_pickle=True)
    # X, y, dataset_name = get_ecoli2()

    # dataset = fetch_datasets()[dataset_name]
    # X, y = dataset['data'], dataset['target']
    # y = np.where(y == -1, 0, y)

    # X, y = np.load("pred_results/X.npy"), np.load("pred_results/y.npy")

    # dataset_names = ["yeast_ml8", "scene", "libras_move", "thyroid_sick", "coil_2000", "solar_flare_m0", "oil", "car_eval_4", "wine_quality", "webpage", "letter_img", "yeast_me2", "ozone_level", "mammography", "protein", "abalone_19"]

    # dataset_names = [
    #                 'isolet', 'libras_move', 'thyroid_sick', 'solar_flare_m0',
    #                 'oil', 'wine_quality', 'letter_img', 'yeast_me2', 'ozone_level',
    #                 'ecoli', 'abalone', 'yeast_ml8', 'scene', 'us_crime', 'car_eval_4', 'abalone_19',
    #                 'satimage', 'optical_digits', 'pen_digits', 'spectrometer'
    # ]

    # function_list = [get_ecoli2, get_yeast3, get_yeast4, get_yeast5, get_waveform1, get_waveform2, get_waveform3,
    #                  get_statlog_vehicle_silhouettes2, get_statlog_vehicle_silhouettes4]

    # function_list = ['get_statlog_vehicle_silhouettes1', 'get_statlog_vehicle_silhouettes2', 'get_statlog_vehicle_silhouettes3', 'get_statlog_vehicle_silhouettes4', 'get_waveform1', 'get_waveform2', 'get_waveform3', 'get_wine1', 'get_wine2', 'get_yeast1', 'get_yeast2', 'get_yeast3', 'get_yeast4', 'get_yeast5']
    function_list = ["get_ecoli1"]
    # function_list = [name for name, obj in inspect.getmembers(data_util) if
    #              inspect.isfunction(obj) and name.startswith('get')]

    # for dataset_name in dataset_names:
    #     X, y = load_data(dataset_name)
    for func in function_list:
        func = globals()[func]
        X, y, dataset_name = func()
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        DGBDF_weighted_layers_acc_list = []
        DGBDF_weighted_layers_auc_list = []
        DGBDF_weighted_layers_gmean_list = []
        DGBDF_weighted_layers_sen_list = []
        DGBDF_weighted_layers_spe_list = []
        DGBDF_weighted_layers_aupr_list = []
        DGBDF_weighted_layers_f1_macro_list = []
        DGBDF_weighted_layers_precision_list = []
        DGBDF_weighted_layers_recall_list = []

        per_layer_res = []
        per_layer_res_weighted_layers = []

        print(dataset_name)
        print("Counter(y)", Counter(y))

        model = UncertaintyAwareDeepForest(get_config())
        model_name = "UncertaintyAwareDeepForest"

        save_dir = os.path.join("compared_results_evidence", f"{dataset_name}_result")
        os.makedirs(save_dir, exist_ok=True)

        # 存储所有样本的预测结果
        all_proba = []
        all_pred = []
        all_true_label = []


        for fold_idx, (train_index, test_index) in enumerate(skf.split(X, y)):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            config = get_config()

            UADF = UncertaintyAwareDeepForest(config)
            UADF.fit(X_train, y_train)

            # shap_analysis_per_layer(DGBDF, X_test, y_test)

            per_layer_res.append(UADF.per_layer_res)
            per_layer_res_weighted_layers.append(UADF.per_layer_res_weighted_layers)
            DGBDF_pred_proba_weighted = UADF.predict_proba_weighted_layers(
                X_test)

            # 检查 DGBDF_pred_proba_weighted 是否有 NaN
            if np.isnan(DGBDF_pred_proba_weighted).any():
                print("DGBDF_pred_proba_weighted contains NaN values.")
                print("DGBDF_pred_proba_weighted\n", DGBDF_pred_proba_weighted)
                # 找出 NaN 元素的位置
                nan_indices = np.argwhere(np.isnan(DGBDF_pred_proba_weighted))
                print("Indices of NaN values:", nan_indices)


            DGBDF_pred_weighted = UADF.category[
                np.argmax(DGBDF_pred_proba_weighted, axis=1)]

            # 保存当前折的预测结果
            all_proba.extend(DGBDF_pred_proba_weighted)
            all_pred.extend(DGBDF_pred_weighted)
            all_true_label.extend(y_test)

            print("DGBDF_weighted_layers acc: ", accuracy_score(y_test, DGBDF_pred_weighted))
            print("DGBDF_weighted_layers auc: ",
                  roc_auc_score(y_test, DGBDF_pred_proba_weighted[:, 1]))
            print("DGBDF_weighted_layers gmean: ",
                  geometric_mean_score(y_test, DGBDF_pred_weighted))
            print("DGBDF_weighted_layers sen: ", sensitivity_score(y_test, DGBDF_pred_weighted))
            print("DGBDF_weighted_layers spe: ", specificity_score(y_test, DGBDF_pred_weighted))
            print("DGBDF_weighted_layers aupr: ",
                  average_precision_score(y_test, DGBDF_pred_proba_weighted[:, 1]))
            print("DGBDF_weighted_layers f1_macro: ",
                  f1_score(y_test, DGBDF_pred_weighted, average="macro"))
            print("DGBDF_weighted_layers precision: ",
                  precision_score(y_test, DGBDF_pred_weighted, average="macro"))
            print("DGBDF_weighted_layers recall: ",
                    sensitivity_score(y_test, DGBDF_pred_weighted, average="macro"))


            DGBDF_weighted_layers_acc_list.append(accuracy_score(y_test, DGBDF_pred_weighted))
            DGBDF_weighted_layers_auc_list.append(roc_auc_score(y_test, DGBDF_pred_proba_weighted[:, 1]))
            DGBDF_weighted_layers_gmean_list.append(geometric_mean_score(y_test, DGBDF_pred_weighted))
            DGBDF_weighted_layers_sen_list.append(sensitivity_score(y_test, DGBDF_pred_weighted))
            DGBDF_weighted_layers_spe_list.append(specificity_score(y_test, DGBDF_pred_weighted))
            DGBDF_weighted_layers_aupr_list.append(
                average_precision_score(y_test, DGBDF_pred_proba_weighted[:, 1]))
            DGBDF_weighted_layers_f1_macro_list.append(f1_score(y_test, DGBDF_pred_weighted, average="macro"))
            DGBDF_weighted_layers_precision_list.append(precision_score(y_test, DGBDF_pred_weighted, average="macro"))
            DGBDF_weighted_layers_recall_list.append(sensitivity_score(y_test, DGBDF_pred_weighted, average="macro"))

            # 保存当前折的预测结果到 .npy 文件
            # fold_save_dir = os.path.join(save_dir, f"fold_{fold_idx + 1}")
            # os.makedirs(fold_save_dir, exist_ok=True)  # 确保文件夹存在
            #
            # np.save(os.path.join(fold_save_dir, f"{dataset_name}_{model_name}_proba.npy"), np.array(DGBDF_pred_proba_weighted))
            # np.save(os.path.join(fold_save_dir, f"{dataset_name}_{model_name}_pred.npy"), np.array(DGBDF_pred_weighted))
            # np.save(os.path.join(fold_save_dir, f"{dataset_name}_{model_name}_true_label.npy"), np.array(y_test))
            #
            # print(f"Saved predictions for fold {fold_idx + 1} to {fold_save_dir}")

        # print(f"All folds predictions for {model_name} saved successfully.")

        print("DGBDF weighted_layers acc mean: ", np.mean(DGBDF_weighted_layers_acc_list))
        print("DGBDF weighted_layers auc mean: ", np.mean(DGBDF_weighted_layers_auc_list))
        print("DGBDF weighted_layers gmean mean: ", np.mean(DGBDF_weighted_layers_gmean_list))
        print("DGBDF weighted_layers sen mean: ", np.mean(DGBDF_weighted_layers_sen_list))
        print("DGBDF weighted_layers spe mean: ", np.mean(DGBDF_weighted_layers_spe_list))
        print("DGBDF weighted_layers aupr mean: ", np.mean(DGBDF_weighted_layers_aupr_list))
        print("DGBDF weighted_layers f1_macro mean: ",
              np.mean(DGBDF_weighted_layers_f1_macro_list))
        print("DGBDF weighted_layers precision mean: ",
                np.mean(DGBDF_weighted_layers_precision_list))
        print("DGBDF weighted_layers recall mean: ",
                np.mean(DGBDF_weighted_layers_recall_list))

        print("dataset_name: ", dataset_name)
