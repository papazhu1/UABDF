import numpy as np
from scipy.special import gammaln, digamma
from scipy.stats import linregress
import matplotlib.pyplot as plt
from sklearn.metrics import consensus_score
from sympy.abc import alpha

def combine_opinions(opinion_1, opinion_2):
    b_i, d_i, u_i = opinion_1
    b_j, d_j, u_j = opinion_2
    C = b_i * d_j + b_j * d_i
    b_combined = (b_i * b_j + b_i * u_j + b_j * u_i) / (1 - C)
    d_combined = (d_i * d_j + d_i * u_j + d_j * u_i) / (1 - C)
    u_combined = (u_i * u_j) / (1 - C)
    b_combined = b_combined / (b_combined + d_combined + u_combined)
    d_combined = d_combined / (b_combined + d_combined + u_combined)
    u_combined = u_combined / (b_combined + d_combined + u_combined)
    return b_combined, d_combined, u_combined


# 传入的参数已经是对次数的统计值了

def DS_Combine_evidence_with_opinion(E, Opinion):
    W = 50 / 2
    beta1, alpha1 = (E + W)[0]
    S1 = alpha1 + beta1

    b1 = (alpha1 - W) / S1
    d1 = (beta1 - W) / S1
    u1 = 2 * W / S1

    o1 = np.array([b1, d1, u1])
    o2 = Opinion
    #
    # elementwise_product = o1 * o2
    # C = np.sum(elementwise_product)
    #
    # consensus_opinion = (1 / C) * elementwise_product + (o1 + o2)
    # consensus_opinion /= np.sum(consensus_opinion)

    consensus_opinion = combine_opinions(o1, o2)
    return consensus_opinion


def DS_Combine2(E1, E2):

    W = 50 / 2
    beta1, alpha1 = (E1 + W)[0]
    beta2, alpha2 = (E2 + W)[0]

    S1 = alpha1 + beta1
    S2 = alpha2 + beta2

    b1 = (alpha1-W) / S1
    d1 = (beta1-W) / S1
    u1 = 2 * W / S1

    b2 = (alpha2-W) / S2
    d2 = (beta2-W) / S2
    u2 = 2 * W / S2

    o1 = np.array([b1, d1, u1])
    o2 = np.array([b2, d2, u2])

    # elementwise_product = o1 * o2
    # C = np.sum(elementwise_product)
    #
    # consensus_opinion = (1 / C) * elementwise_product + (o1 + o2)
    # consensus_opinion /= np.sum(consensus_opinion)

    consensus_opinion = combine_opinions(o1, o2)
    return consensus_opinion

def DS_Combine_ensemble_for_instances(E1, E2):
    n_classes = E1.shape[1]
    alpha1 = E1 + 1
    alpha2 = E2 + 1

    S1 = np.sum(alpha1, axis=1, keepdims=True)
    S2 = np.sum(alpha2, axis=1, keepdims=True)

    # print("S1", S1)
    # print("S2", S2)
    b1 = E1 / S1
    b2 = E2 / S2

    # print("b1", b1)
    # print("b2", b2)

    u1 = n_classes / S1
    u2 = n_classes / S2

    # print("u1", u1)
    # print("u2", u2)

    bb = np.einsum('ij,ik->ijk', b1, b2)

    # 计算 b^0 * u^1
    uv1_expand = np.broadcast_to(u2, b1.shape)  # 使用 np.broadcast_to 匹配形状
    bu = b1 * uv1_expand

    # 计算 b^1 * u^0
    uv_expand = np.broadcast_to(u1, b2.shape)
    ub = b2 * uv_expand

    # 计算 K
    bb_sum = np.sum(bb, axis=(1, 2))  # 计算 bb 的总和
    bb_diag = np.einsum('ijj->i', bb)  # 提取对角线并在批量中求和
    K = bb_sum - bb_diag

    # 计算 b^a
    b_combined = (b1 * b2 + bu + ub) / np.expand_dims((1 - K), axis=1)

    # 计算 u^a
    u_combined = (u1 * u2) / np.expand_dims((1 - K), axis=1)

    # 计算新的 S
    S_combined = n_classes / u_combined

    # 计算新的 e_k
    e_combined = b_combined * np.broadcast_to(S_combined, b_combined.shape)
    alpha_combined = e_combined + 1

    return alpha_combined, b_combined, u_combined




# 需要传入的参数：
# alpha1 = evidence1 + 1
# alpha2 = evidence2 + 1
def DS_Combin_two(alpha1, alpha2, n_classes):
    # 计算两个DS证据的合并
    alpha = {0: alpha1, 1: alpha2}
    b, S, E, u = {}, {}, {}, {}

    for v in range(2):

        # S[v]将每个样本的所有类别的alpha[v]相加，每个样本的S值都为 4
        S[v] = np.sum(alpha[v], axis=1, keepdims=True)# 使用 np.sum 计算 S
        E[v] = alpha[v] - 1
        # print("alpha[v].shape", alpha[v].shape)
        # print("E[v].shape", E[v].shape)
        # print("S[v].shape", S[v].shape)
        # print("alpha[v]", alpha[v])
        # print("E[v]", E[v])
        # print("S[v]", S[v])
        # b[v] = E[v] / np.expand_dims(S[v], axis=1)  # 使用 np.expand_dims 进行广播
        b[v] = E[v] / S[v]
        # print("np.expand_dims(S[v], axis=1).shape", np.expand_dims(S[v], axis=1).shape)
        # print("b[v].shape", b[v].shape)
        u[v] = n_classes / S[v]

        print(u[v])

    # 计算 b^0 @ b^(0+1)

    # print("b[0].shape", b[0].shape)
    # print("b[1].shape", b[1].shape)
    bb = np.einsum('ij,ik->ijk', b[0], b[1])  # 使用 np.einsum 实现批量矩阵乘法
    # print("bb.shape", bb.shape)



    # 计算 b^0 * u^1
    uv1_expand = np.broadcast_to(u[1], b[0].shape)  # 使用 np.broadcast_to 匹配形状
    bu = b[0] * uv1_expand

    # 计算 b^1 * u^0
    uv_expand = np.broadcast_to(u[0], b[1].shape)
    ub = b[1] * uv_expand

    # 计算 K
    bb_sum = np.sum(bb, axis=(1, 2))  # 计算 bb 的总和
    bb_diag = np.einsum('ijj->i', bb)  # 提取对角线并在批量中求和
    K = bb_sum - bb_diag

    # 计算 b^a
    b_combined = (b[0] * b[1] + bu + ub) / np.expand_dims((1 - K), axis=1)

    # 计算 u^a
    u_combined = (u[0] * u[1]) / np.expand_dims((1 - K), axis=1)

    # 计算新的 S
    S_combined = n_classes / u_combined

    # 计算新的 e_k
    e_combined = b_combined * np.broadcast_to(S_combined, b_combined.shape)
    alpha_combined = e_combined + 1

    return alpha_combined, b_combined, u_combined


# 需要传入的参数：
# alpha = evidence + 1
# c = n_classes
def KL(alpha, c):
    beta = np.ones((1, c))
    S_alpha = np.sum(alpha, axis=1, keepdims=True)  # 使用 np.sum 计算 S_alpha
    S_beta = np.sum(beta, axis=1, keepdims=True)

    # 使用 scipy.special 中的 gammaln 计算 lnB 和 lnB_uni
    lnB = gammaln(S_alpha) - np.sum(gammaln(alpha), axis=1, keepdims=True)
    lnB_uni = np.sum(gammaln(beta), axis=1, keepdims=True) - gammaln(S_beta)

    # 计算 digamma 值
    dg0 = digamma(S_alpha)
    dg1 = digamma(alpha)

    kl = np.sum((alpha - beta) * (dg1 - dg0), axis=1, keepdims=True) + lnB + lnB_uni

    if kl < 0:
        raise ValueError('kl < 0')
    return kl

def calculate_KL(alpha, c):
    return KL(alpha, c)

def calculate_A(alpha, p, c):
    S = np.sum(alpha, axis=1, keepdims=True)
    label = np.eye(c)[p]  # 创建独热编码标签
    A = np.sum(label * (digamma(S) - digamma(alpha)), axis=1, keepdims=True)
    return A


def KL_divergence(alpha1, beta1, alpha2, beta2):
    """
    计算 Beta 分布的 KL 散度 D_KL(Beta(alpha1, beta1) || Beta(alpha2, beta2))
    """
    term1 = gammaln(alpha2 + beta2) - gammaln(alpha2) - gammaln(beta2)
    term2 = gammaln(alpha1 + beta1) - gammaln(alpha1) - gammaln(beta1)
    term3 = (alpha1 - alpha2) * (digamma(alpha1) - digamma(alpha1 + beta1))
    term4 = (beta1 - beta2) * (digamma(beta1) - digamma(alpha1 + beta1))
    KL = term1 - term2 + term3 + term4
    return KL


def JS_divergence(alpha_combined, beta_combined, alpha_i, beta_i):
    """
    计算 Beta 分布的 JS 散度
    D_JS = 0.5 * D_KL(P || M) + 0.5 * D_KL(Q || M)
    """
    # 平均分布 M 的参数
    alpha_m = 0.5 * (alpha_combined + alpha_i)
    beta_m = 0.5 * (beta_combined + beta_i)

    # 计算 KL 散度的两部分
    KL_P_M = KL_divergence(alpha_combined, beta_combined, alpha_m, beta_m)
    KL_Q_M = KL_divergence(alpha_i, beta_i, alpha_m, beta_m)

    # 计算 JS 散度
    JS = 0.5 * KL_P_M + 0.5 * KL_Q_M

    # 打印中间结果
    print(f"alpha_combined: {alpha_combined}, beta_combined: {beta_combined}")
    print(f"alpha_i: {alpha_i}, beta_i: {beta_i}")
    # print(f"alpha_m: {alpha_m}, beta_m: {beta_m}")
    # print(f"KL(P || M): {KL_P_M}, KL(Q || M): {KL_Q_M}")
    print(f"JS Divergence: {JS}")

    return JS


def wasserstein_distance(alpha1, beta1, alpha2, beta2):
    """
    计算 Beta 分布的 Wasserstein 距离
    """
    # 计算均值
    mu1 = alpha1 / (alpha1 + beta1)
    mu2 = alpha2 / (alpha2 + beta2)

    # 计算标准差
    sigma1 = np.sqrt((alpha1 * beta1) / ((alpha1 + beta1) ** 2 * (alpha1 + beta1 + 1)))
    sigma2 = np.sqrt((alpha2 * beta2) / ((alpha2 + beta2) ** 2 * (alpha2 + beta2 + 1)))

    # Wasserstein 距离
    W = np.sqrt((mu1 - mu2) ** 2 + (sigma1 - sigma2) ** 2)

    return W

def ce_loss2(y, alpha_combined, beta_combined, beta_distributions, lamb=0.5):
    """
    计算 Wasserstein 距离的损失函数
    """
    # 将真实目标值 y 转换为一个极端 Beta 分布
    epsilon = 1e-8  # 用于平滑以避免数值问题
    alpha_y = y + epsilon
    beta_y = 1 - y + epsilon

    # print(alpha_y, beta_y, alpha_combined, beta_combined)
    # 计算 Wasserstein 距离与目标值分布的误差
    W_A = wasserstein_distance(alpha_y, beta_y, alpha_combined, beta_combined)

    # 计算 Wasserstein 距离与分类器 Beta 分布的误差
    W_B = 0
    for k in range(len(beta_distributions)):
        alpha_k, beta_k = beta_distributions[k]
        W_B += wasserstein_distance(alpha_combined, beta_combined, alpha_k, beta_k)

    W_B /= len(beta_distributions)

    # print("W_A", W_A, "W_B", W_B)
    Weight = lamb * W_A + (1-lamb) * W_B
    return Weight, W_A, W_B


# 需要传入的参数：
# p 样本真实标签
# alpha = evidence + 1
# c = n_classes
# global_step 当前训练步数
# annealing_step 退火步数
def ce_loss(p, alpha, c, global_step=0, lamb=0.5, average=True):
    # 计算 S 和 E
    S = np.sum(alpha, axis=1, keepdims=True)
    E = alpha - 1

    p = int(p)
    # 独热编码
    label = np.eye(c)[p]  # 创建独热编码标签

    # 计算 A 项
    A = np.sum(label * (digamma(S) - digamma(alpha)), axis=1, keepdims=True)

    alp = E * (1 - label) + 1

    # 现在B项是随着alpha的分布越接近均匀分布，KL散度越大
    B = lamb * KL(alp, c)
    # 返回 (A + B) 的均值
    if average is True:
        res = np.mean(A + B)
        return res, A.reshape(-1, 1), B.reshape(-1, 1)
    else:
        res = lamb * A + (1-lamb) * B

        # 用一个y = kx + b的图像来画出A和B的变化
        A = A.flatten()
        B = B.flatten()

        return res, A.reshape(-1, 1), B.reshape(-1, 1)
