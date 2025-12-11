from imbens.ensemble.under_sampling import SelfPacedEnsembleClassifier
from imbens.metrics import geometric_mean_score
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from sklearn.model_selection import train_test_split
from sympy.physics.quantum.gate import CPHASE
from ucimlrepo import fetch_ucirepo
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import  matplotlib.pyplot as plt


# wisconsin diagnostic breast cancer 数据集
# 01类型
def get_WDBC():
    heart_disease = fetch_ucirepo(id=17)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = pd.get_dummies(X)
    X, y = np.array(X), np.array(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print(Counter(y))
    return X, y, "WDBC"

# Wine1 数据集
def get_wine1():
    dataset = fetch_ucirepo(id=109)  # 假设 Wine 数据集的 ID 为 9
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    print("Wine 类别分布:", Counter(y))
    # 将标签转换为二分类：2, 3 为少数类，其余为多数类
    y = y.apply(lambda x: 0 if x in [2, 3] else 1)  # 1 表示少数类, 0 表示多数类
    print(y)

    X = pd.get_dummies(X)
    X, y = np.array(X), np.array(y)
    print("Wine 类别分布:", Counter(y))
    return X, y, "wine1"

# Wine1 数据集
def get_wine2():
    dataset = fetch_ucirepo(id=109)  # 假设 Wine 数据集的 ID 为 9
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    print("Wine 类别分布:", Counter(y))
    # 将标签转换为二分类：2, 3 为少数类，其余为多数类
    y = y.apply(lambda x: 1 if x in [1, 2] else 0)  # 1 表示少数类, 0 表示多数类
    print(y)

    X = pd.get_dummies(X)
    X, y = np.array(X), np.array(y)
    print("Wine 类别分布:", Counter(y))
    return X, y, "wine2"



# Ecoli1 数据集
def get_ecoli1():
    dataset = fetch_ucirepo(id=39)  # 假设 Ecoli 数据集的 ID 为 10
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    print("Ecoli 类别分布:", Counter(y))

    # 将标签转换为二分类：im 为少数类，其余为多数类
    y = y.apply(lambda x: 1 if x == 'im' or x == 'pp' else 0)  # 1 表示少数类, 0 表示多数类

    X = pd.get_dummies(X)
    X, y = np.array(X), np.array(y)
    print("Ecoli1 类别分布:", Counter(y))
    return X, y, "ecoli1"

# Ecoli2 数据集
def get_ecoli2():
    dataset = fetch_ucirepo(id=39)  # 假设 Ecoli 数据集的 ID 为 10
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    # 将标签转换为二分类：im 为少数类，其余为多数类
    y = y.apply(lambda x: 1 if x == 'imU' or x == 'pp' or x == 'om' else 0)  # 1 表示少数类, 0 表示多数类

    X = pd.get_dummies(X)
    X, y = np.array(X), np.array(y)
    print("Ecoli2 类别分布:", Counter(y))
    print("ecoli.shape", X.shape)
    return X, y, "ecoli2"

def get_glass1():
    dataset = fetch_ucirepo(id=42)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 5 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("Glass1 类别分布:", Counter(y))
    return X, y, "Glass1"

def get_glass2():
    dataset = fetch_ucirepo(id=42)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("Glass2 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 7 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("Glass2 类别分布:", Counter(y))
    return X, y, "Glass2"

def get_glass3():
    dataset = fetch_ucirepo(id=42)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("Glass3 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 1 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("Glass3 类别分布:", Counter(y))
    return X, y, "Glass3"

def get_glass4():
    dataset = fetch_ucirepo(id=42)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("Glass4 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 3 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("Glass4 类别分布:", Counter(y))
    return X, y, "Glass4"


def get_glass5():
    dataset = fetch_ucirepo(id=42)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("Glass5 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 2 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("Glass5 类别分布:", Counter(y))
    return X, y, "Glass5"

# 01类型
def get_haberman():
    heart_disease = fetch_ucirepo(id=43)
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X = pd.get_dummies(X)
    X, y = np.array(X), np.array(y)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    print(Counter(y))
    return X, y, "haberman"

def get_car1():
    dataset = fetch_ucirepo(id=19)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("car1 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == "acc" else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("car1 处理后类别分布:", Counter(y))
    return X, y, "car1"

def get_car2():
    dataset = fetch_ucirepo(id=19)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("car2 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == "good" else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("car2 处理后类别分布:", Counter(y))
    return X, y, "car2"


def get_car3():
    dataset = fetch_ucirepo(id=19)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("car3 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == "vgood" else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("car3 处理后类别分布:", Counter(y))
    return X, y, "car3"

# 01类型,但是数据太少了，用不了
def get_hepatitis():
    dataset = fetch_ucirepo(id=46)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("hepatitis 处理前类别分布:", Counter(y))

    # X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    # y = np.array([1 if label == "vgood" else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("hepatitis 处理后类别分布:", Counter(y))
    return X, y, "hepatitis"

def get_poker_hand():
    dataset = fetch_ucirepo(id=158)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("poker_hand 处理前类别分布:", Counter(y))

    # X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    # y = np.array([1 if label == "vgood" else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("poker_hand 处理后类别分布:", Counter(y))
    return X, y, "poker_hand"

def get_liver_disorders1():
    dataset = fetch_ucirepo(id=60)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("liver_disorders1 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 0.5 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("liver_disorders1 处理后类别分布:", Counter(y))
    return X, y, "liver_disorders1"

def get_liver_disorders2():
    dataset = fetch_ucirepo(id=60)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("liver_disorders2 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 4.0 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("liver_disorders2 处理后类别分布:", Counter(y))
    return X, y, "liver_disorders2"

def get_liver_disorders3():
    dataset = fetch_ucirepo(id=60)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("liver_disorders3 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 6.0 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("liver_disorders3 处理后类别分布:", Counter(y))
    return X, y, "liver_disorders3"

def get_liver_disorders4():
    dataset = fetch_ucirepo(id=60)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("liver_disorders4 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 2.0 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("liver_disorders4 处理后类别分布:", Counter(y))
    return X, y, "liver_disorders4"


def get_yeast1():
    dataset = fetch_ucirepo(id=110)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("yeast1 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'CYT' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("yeast1 处理后类别分布:", Counter(y))
    return X, y, "yeast1"

def get_yeast2():
    dataset = fetch_ucirepo(id=110)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("yeast2 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'NUC' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("yeast2 处理后类别分布:", Counter(y))
    return X, y, "yeast2"

def get_yeast3():
    dataset = fetch_ucirepo(id=110)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("yeast3 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'MIT' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("yeast3 处理后类别分布:", Counter(y))
    return X, y, "yeast3"

def get_yeast4():
    dataset = fetch_ucirepo(id=110)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("yeast4 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'ME3' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("yeast4 处理后类别分布:", Counter(y))
    return X, y, "yeast4"

def get_yeast5():
    dataset = fetch_ucirepo(id=110)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("yeast5 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'ME2' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("yeast5 处理后类别分布:", Counter(y))
    return X, y, "yeast5"

def get_waveform1():
    dataset = fetch_ucirepo(id=107)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("waveform1 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 0 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("waveform1 处理后类别分布:", Counter(y))
    return X, y, "waveform1"

def get_waveform2():
    dataset = fetch_ucirepo(id=107)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("waveform2 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 1 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("waveform2 处理后类别分布:", Counter(y))
    return X, y, "waveform2"

def get_waveform3():
    dataset = fetch_ucirepo(id=107)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("waveform3 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 2 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("waveform3 处理后类别分布:", Counter(y))
    return X, y, "waveform3"

def get_page_blocks1():
    dataset = fetch_ucirepo(id=78)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("page_blocks1 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 2 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("page_blocks1 处理后类别分布:", Counter(y))
    return X, y, "page_blocks1"

def get_page_blocks2():
    dataset = fetch_ucirepo(id=78)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("page_blocks2 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 5 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("page_blocks2 处理后类别分布:", Counter(y))
    return X, y, "page_blocks2"


def get_page_blocks3():
    dataset = fetch_ucirepo(id=78)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("page_blocks3 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 4 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("page_blocks3 处理后类别分布:", Counter(y))
    return X, y, "page_blocks3"


def get_page_blocks4():
    dataset = fetch_ucirepo(id=78)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("page_blocks4 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 3 else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("page_blocks4 处理后类别分布:", Counter(y))
    return X, y, "page_blocks4"

def get_statlog_vehicle_silhouettes1():
    dataset = fetch_ucirepo(id=149)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("statlog_vehicle_silhouettes1 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'saab' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("statlog_vehicle_silhouettes1 处理后类别分布:", Counter(y))
    return X, y, "statlog_vehicle_silhouettes1"

def get_statlog_vehicle_silhouettes2():
    dataset = fetch_ucirepo(id=149)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("statlog_vehicle_silhouettes2 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'bus' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("statlog_vehicle_silhouettes2 处理后类别分布:", Counter(y))
    return X, y, "statlog_vehicle_silhouettes2"

def get_statlog_vehicle_silhouettes3():
    dataset = fetch_ucirepo(id=149)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("statlog_vehicle_silhouettes3 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'opel' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("statlog_vehicle_silhouettes3 处理后类别分布:", Counter(y))
    return X, y, "statlog_vehicle_silhouettes3"

def get_statlog_vehicle_silhouettes4():
    dataset = fetch_ucirepo(id=149)
    X = dataset.data.features
    y = dataset.data.targets

    # 删除缺失值
    data = pd.concat([X, y], axis=1).dropna()

    # 分离特征和标签
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    print("statlog_vehicle_silhouettes4 处理前类别分布:", Counter(y))

    X = pd.get_dummies(pd.DataFrame(X)).values  # 转为 DataFrame 后编码
    y = np.array([1 if label == 'van' else 0 for label in y])  # 使用列表推导式处理

    X, y = np.array(X), y

    print("statlog_vehicle_silhouettes4 处理后类别分布:", Counter(y))
    return X, y, "statlog_vehicle_silhouettes4"


if __name__ == "__main__":
    # X, y, name = get_yeast5()
    # print(len(X[0]))
    # # 计算正负样本数
    # positive_samples = sum(y)  # y 中值为 1 的样本
    # negative_samples = len(y) - positive_samples  # 负样本数
    #
    # # 计算 IR（不平衡比率）
    # IR = negative_samples / positive_samples if negative_samples != 0 else float('inf')  # 防止除以零
    #
    # # 输出 IR 值
    # print(f"{IR:.1f}")

    from imbens.datasets import fetch_datasets

    dataset_name = "isolet"
    dataset = fetch_datasets()[dataset_name]
    X, y = dataset['data'], dataset['target']
    y = np.where(y == -1, 0, y)
    print(len(X[0]))
    # 计算正负样本数
    positive_samples = sum(y)  # y 中值为 1 的样本
    negative_samples = len(y) - positive_samples  # 负样本数

    print(positive_samples)
    print(negative_samples)
    # 计算 IR（不平衡比率）
    IR = negative_samples / positive_samples if negative_samples != 0 else float('inf')  # 防止除以零

    # 输出 IR 值
    print(f"{IR:.1f}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

    f1_macro_list = []
    auc_list = []
    aupr_list = []
    gmean_list = []


    for percent in range(2, 102, 2):
        train_size = percent / 100  # 当前训练集比例
        X_train_subset, _, y_train_subset, _ = train_test_split(X_train, y_train, train_size=train_size,
                                                                stratify=y_train)

        # 使用逻辑回归模型
        # model = UncertaintyAwareDeepForest(get_config())
        model = SelfPacedEnsembleClassifier()
        model.fit(X_train_subset, y_train_subset)  # 训练模型

        # 预测
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        # 计算准确率
        f1_macro = f1_score(y_test, y_pred, average='macro')
        auc = roc_auc_score(y_test, y_pred_proba)
        aupr = average_precision_score(y_test, y_pred_proba)
        gmean = geometric_mean_score(y_test, y_pred)

        f1_macro_list.append(f1_macro)
        auc_list.append(auc)
        aupr_list.append(aupr)
        gmean_list.append(gmean)

        # 输出当前训练集比例和各项指标
        print(f"Training size: {percent}%")
        print(f"F1-macro: {f1_macro:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"AUPR: {aupr:.4f}")
        print(f"Gmean: {gmean:.4f}")
        print("-" * 40)

        # 绘制性能指标变化图
        plt.figure(figsize=(12, 8))

        # F1-macro 曲线
        plt.subplot(2, 2, 1)
        plt.plot(range(2, 102, 2), f1_macro_list, marker='o', label="F1-macro")
        plt.title('F1-macro vs Training Size')
        plt.xlabel('Training Size (%)')
        plt.ylabel('F1-macro')
        plt.grid(True)

        # AUC 曲线
        plt.subplot(2, 2, 2)
        plt.plot(range(2, 102, 2), auc_list, marker='o', label="AUC", color='orange')
        plt.title('AUC vs Training Size')
        plt.xlabel('Training Size (%)')
        plt.ylabel('AUC')
        plt.grid(True)

        # AUPR 曲线
        plt.subplot(2, 2, 3)
        plt.plot(range(2, 102, 2), aupr_list, marker='o', label="AUPR", color='green')
        plt.title('AUPR vs Training Size')
        plt.xlabel('Training Size (%)')
        plt.ylabel('AUPR')
        plt.grid(True)

        # Gmean 曲线
        plt.subplot(2, 2, 4)
        plt.plot(range(2, 102, 2), gmean_list, marker='o', label="Gmean", color='red')
        plt.title('Gmean vs Training Size')
        plt.xlabel('Training Size (%)')
        plt.ylabel('Gmean')
        plt.grid(True)

        # 显示图表
        plt.tight_layout()
        plt.show()

        # 保存性能数据到文件
        np.savetxt("performance_metrics_SPE.csv", np.array([f1_macro_list, auc_list, aupr_list, gmean_list]).T,
                   delimiter=',', header="F1-macro,AUC,AUPR,Gmean", comments='')










