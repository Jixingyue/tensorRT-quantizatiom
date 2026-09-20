"""
part5 —— TensorRT 熵校准（Entropy Calibration / KL）算法
=======================================================
这是 TensorRT 做 INT8 量化时，求"动态范围阈值"最经典的方法。

核心思路（一句话）：
  把激活值统计成直方图后，不是直接取最大值，而是尝试每一个可能的阈值，
  把"阈值以内的直方图"量化成 128 个 bin（模拟 INT8），
  再用 KL 散度衡量"量化后的分布 q"与"原始分布 p"的差距，
  选择 KL 最小、也就是信息丢失最少的那一个阈值。

运行结果示例：
  最大的激活值 4562.1814033289875
  threshold 所在组: 1814                    ← 最优阈值落在第 1814 个区间
  threshold 所在组的区间范围: 4040.9467531556247   ← 阈值约 4040，而不是直接用最大值 4562
"""

import random
import numpy as np
import matplotlib.pyplot as plt


def generator_P(size):
    """生成一组模拟的激活值数据（服从高斯分布）。
    随机均值和方差，模拟神经网络某层 ReLU 之后的激活值（都是正数）。
    仅供测试算法使用，实际工程中替换成真实模型跑出来的激活值即可。
    """
    walk = []
    avg = random.uniform(3.000, 600.999)
    std = random.uniform(500.000, 1024.959)
    for _ in range(size):
        walk.append(random.gauss(avg, std))
    return walk


def smooth_distribution(p, eps=0.0001):
    """平滑概率分布，避免出现 0（0 会让后续 log 计算变成无穷大）。
    与 part4 的 smooth_data 思路一致，这是带校验的完整版。
    """
    is_zeros = (p == 0).astype(np.float32)
    is_nonzeros = (p != 0).astype(np.float32)
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros
    if not n_nonzeros:
        raise ValueError('The discrete probability distribution is malformed. All entries are 0.')
    eps1 = eps * float(n_zeros) / float(n_nonzeros)
    assert eps1 < 1.0, 'n_zeros=%d, n_nonzeros=%d, eps1=%f' % (n_zeros, n_nonzeros, eps1)
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    assert (hist <= 0).sum() == 0
    return hist


import copy
import scipy.stats as stats


def threshold_distribution(distribution, target_bin=128):
    """★★★ 熵校准核心函数：寻找"信息丢失最少"的直方图阈值 ★★★

    参数：
      distribution : 激活值的直方图（每个格子记录了落在该区间的数据个数）
      target_bin   : 量化后希望的格子数，默认 128（对应 INT8 能表达的级别数）

    流程（对每一个候选阈值 threshold，从 target_bin 一直试到最后一个格子）：
      1. 截取直方图 [0, threshold) 作为候选分布，把多余的"尾巴"并入最后一个格子；
         这就是"原始分布 p"；
      2. 把 p 按比例合并成 target_bin 个格子 → 模拟量化成 INT8 后的"量化分布 q"；
      3. 再把 q 展开回原来的格子数，方便和 p 对齐比较；
      4. 分别平滑 p、q 后，用 scipy.stats.entropy 计算 KL 散度；
      5. 记录所有候选阈值对应的 KL，取 KL 最小的那个阈值作为最终结果。

    返回：
      最优阈值对应的直方图格子下标（再用 bins[threshold] 即可得到真实的浮点阈值）。
    """
    distribution = distribution[1:]                  # 去掉第 0 格（通常是 0 值区，对校准无意义）
    length = distribution.size
    threshold_sum = sum(distribution[target_bin:])   # 第一个候选阈值右侧的所有"尾巴"
    kl_divergence = np.zeros(length - target_bin)    # 记录每个候选阈值的 KL

    for threshold in range(target_bin, length):
        sliced_nd_hist = copy.deepcopy(distribution[:threshold])   # 阈值以内的直方图

        # ---- 1. 构造参考分布 p：把阈值右侧的尾巴全部并入最后一格 ----
        p = sliced_nd_hist.copy()
        p[threshold - 1] += threshold_sum
        threshold_sum = threshold_sum - distribution[threshold]    # 尾巴随阈值右移而更新

        # is_nonzeros[k] 标记 p 中非 0 的位置（展开 q 时要用）
        is_nonzeros = (p != 0).astype(np.int64)

        # ---- 2. 构造量化分布 q：把 p 合并成 target_bin 格（模拟 INT8）----
        quantized_bins = np.zeros(target_bin, dtype=np.int64)
        num_merged_bins = sliced_nd_hist.size // target_bin        # 每格合并几个原始格

        for j in range(target_bin):
            start = j * num_merged_bins
            stop = start + num_merged_bins
            quantized_bins[j] = sliced_nd_hist[start:stop].sum()   # 组内求和
        quantized_bins[-1] += sliced_nd_hist[target_bin * num_merged_bins:].sum()  # 余数并入最后一格

        # ---- 3. 把 q 展开回 p 的格子数，使两者长度一致、可逐格对比 ----
        q = np.zeros(sliced_nd_hist.size, dtype=np.float64)
        for j in range(target_bin):
            start = j * num_merged_bins
            if j == target_bin - 1:
                stop = -1
            else:
                stop = start + num_merged_bins
            norm = is_nonzeros[start:stop].sum()
            if norm != 0:
                q[start:stop] = float(quantized_bins[j]) / float(norm)   # 均值展开

        # ---- 4. 平滑 + 计算 KL 散度 ----
        p = smooth_distribution(p)
        q = smooth_distribution(q)
        kl_divergence[threshold - target_bin] = stats.entropy(p, q)      # KL(p||q)

    # ---- 5. 取 KL 最小（信息丢失最少）的阈值 ----
    min_kl_divergence = np.argmin(kl_divergence)
    threshold_value = min_kl_divergence + target_bin
    return threshold_value


if __name__ == '__main__':
    # ========== 用模拟数据完整跑一遍熵校准 ==========

    size = 20480
    P = generator_P(size)
    P = np.array(P)
    P = P[P > 0]                     # ReLU 之后激活值非负，只保留正数
    print("最大的激活值", max(np.absolute(P)))

    # 统计直方图：分成 2048 个格子
    hist, bins = np.histogram(P, bins=2048)
    # 跑熵校准，找到最优阈值（返回的是格子下标）
    threshold = threshold_distribution(hist, target_bin=128)
    print("threshold 所在组:", threshold)
    print("threshold 所在组的区间范围:", bins[threshold])

    # 画图：直方图 + 红色虚线标出最优阈值位置
    plt.title("Relu activation value Histogram")
    plt.xlabel("Activation values")
    plt.ylabel("Normalized number of Counts")
    plt.hist(P, bins=2047)
    plt.vlines(bins[threshold], 0, 30, colors = "r", linestyles = "dashed")
    plt.show()
