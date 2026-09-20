"""
part4 —— KL 散度入门（熵校准算法的核心思想）
===========================================
KL 散度（Kullback-Leibler Divergence）衡量两个概率分布 p、q 的"差距"：
    KL(p||q) = Σ p[i] * log(p[i] / q[i])

它在量化中的意义：p 是原始浮点分布（直方图），q 是量化后的分布，
KL 越小说明量化前后分布越接近、信息丢失越少。
这也是 part5 中 TensorRT 熵校准算法挑选阈值的判据。

本文件演示两件事：
  1. 如何把多个直方图区间"合并"成更少的区间（量化分布 q 的雏形）；
  2. 如何计算两个分布之间的 KL 散度。
"""

import numpy as np
import matplotlib.pyplot as plt


def smooth_data(p, eps=0.0001):
    """平滑处理：避免概率分布中出现 0。
    因为 log(0) 无定义，0 概率会让 KL 计算直接崩溃。

    做法：给所有为 0 的位置加上一个很小的 eps，
          同时从非 0 位置扣掉相应总量，保证分布总和仍是 1。
    """
    is_zeros = (p == 0).astype(np.float32)     # 为 0 的位置标记为 1
    is_nonzeros = (p != 0).astype(np.float32)  # 非 0 的位置标记为 1
    n_zeros = is_zeros.sum()
    n_nonzeros = p.size - n_zeros

    eps1 = eps * n_zeros / n_nonzeros          # 从每个非 0 位置平均扣除的量
    hist = p.astype(np.float32)
    hist += eps * is_zeros + (-eps1) * is_nonzeros
    return hist


def cal_kl(p, q):
    """计算 KL 散度：KL(p||q) = Σ p[i] * log(p[i]/q[i])。
    数值越小 → 两个分布越接近。
    注意：调用前最好先对 p、q 做 smooth_data，避免 q 中出现 0 导致除零。
    """
    KL = 0.
    for i in range(len(p)):
        KL += p[i] * np.log(p[i] / (q[i]))
    return KL


def kl_test(x, kl_threshold=0.01, size=10):
    """演示函数：随机生成分布 y，直到它与 x 的 KL 散度小于阈值才停止。
    目的：直观感受"KL 越小、两个分布形状越像"。
    """
    y_out = []
    while True:
        y = [np.random.uniform(1, size + 1) for i in range(size)]  # 随机生成一个分布
        y /= np.sum(y)                                              # 归一化：总和为 1
        kl_result = cal_kl(x, y)
        if kl_result < kl_threshold:
            print(kl_result)
            y_out = y
            plt.plot(x)   # 红色：原始分布
            plt.plot(y)   # 蓝色：拟合出来的分布
            break
    return y_out


def KL_main():
    """随机生成一个分布 x，然后寻找一个 KL 散度足够小的 y 来近似它。"""
    np.random.seed(1)
    size = 10
    x = [np.random.uniform(1, size + 1) for i in range(size)]
    x = x / np.sum(x)
    y_out = kl_test(x, kl_threshold=0.01)
    plt.show()
    print(x, y_out)


if __name__ == '__main__':
    # ========== 重点演示：区间合并 + KL 计算（熵校准的核心思想） ==========

    p = [1, 0, 2, 3, 5, 3, 1, 7]   # 模拟一个 8 个区间的直方图（记录各区间计数）
    bin = 4                         # 要合并成 4 组（模拟量化后只有 4 个表示级别）

    # 1) 把 8 个区间分成 4 组，每组取"非 0 项的平均值"作为代表值 → 得到量化分布 q
    split_p = np.array_split(p, bin)
    q = []
    for arr in split_p:
        avg = np.sum(arr) / np.count_nonzero(arr)   # 组内非 0 项的平均值
        for item in arr:
            if item != 0:
                q.append(avg)
                continue
            q.append(0)
    print(q)

    # 2) 归一化，把计数转成真正的概率分布（各项之和为 1）
    p /= np.sum(p)
    q /= np.sum(q)
    print(p)
    print(q)

    # 3) 平滑处理，避免 0 概率导致 KL 计算崩溃
    p = smooth_data(p)
    q = smooth_data(q)
    print(p)
    print(q)

    # 4) 计算 KL：衡量"量化合并"前后的分布差距（输出约 0.1311）
    print(cal_kl(p, q))
