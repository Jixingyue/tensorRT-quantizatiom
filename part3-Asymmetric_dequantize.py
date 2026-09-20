"""
part3 —— 手写"对称量化"（INT8，范围 -127~127）+ 直方图动态范围
=============================================================
对称量化：只用一个 scale，没有零点 z，整数范围固定为 [-127, 127]。

本文件提供两种求 scale 的方法：
  1. Max 方法      ：scale = max(|x|) / 127        （简单粗暴，受离群点影响大）
  2. 直方图方法     ：统计直方图后去掉两头离群点，只保留覆盖 99% 数据的范围
                     （更贴近真实激活值的分布，是 TensorRT 校准思想的雏形）

⚠️ 注意：文件名虽是 Asymmetric，但代码实现的是对称量化（详见 README 说明）。
"""

import numpy as np


def saturate(x):
    """数值饱和：把 x 裁剪到 [-127, 127]（对称 INT8 的合法范围）。"""
    return np.clip(x, -127, 127)


def scale_cal(x):
    """方法一：Max 法求对称量化的 scale。
    scale = 数据绝对值的最大值 / 127
    （127 是 INT8 能表示的最大正数）
    缺点：只要存在一个很大的离群点，scale 就会被拉大，
    导致其他正常数值可用的精度被压低。
    """
    max_val = np.max(np.abs(x))
    return max_val / 127


def quant_float_data(x, scale):
    """浮点 -> 整数（对称量化）：xq = round(x / scale)，再裁剪到 [-127, 127]。"""
    xq = np.round(x / scale)
    return saturate(xq)


def dequant_data(xq, scale):
    """整数 -> 浮点（反量化）：x ≈ xq * scale。"""
    x = (xq * scale).astype('float32')
    return x


def histgram_range(x):
    """方法二：直方图法求动态范围（更聪明，是 TensorRT 校准的雏形）。

    步骤：
      1. 把数据按大小等分成 100 个区间（bin），得到直方图 hist 和区间边界 range；
      2. 从两侧不断"削掉"数据最少的那一区间，直到剩余区间覆盖的数据量 ≤ 99% 为止
         （即把影响精度的离群点排除掉）；
      3. 取剩余区间的边界作为动态范围，再除以 127 得到 scale。

    效果：得到的 scale 通常比 Max 法更小 → 量化精度更高。
    """
    hist, range = np.histogram(x, 100)      # 100 个等宽区间的计数统计
    total = len(x)
    left = 0                                # 左边界指针
    right = len(hist) - 1                   # 右边界指针
    limit = 0.99                            # 覆盖率阈值：保留 99% 的数据
    while True:
        cover_percent = hist[left:right].sum() / total
        if cover_percent <= limit:
            break                           # 当前覆盖数据 ≤ 99%，范围够窄了，停止收缩

        # 哪边区间里的数据更少，就"削"哪边（尽量少丢数据）
        if hist[left] < hist[right]:
            left += 1
        else:
            right -= 1

    left_val = range[left]                  # 收缩后的最小浮点值
    right_val = range[right]                # 收缩后的最大浮点值
    dynamic_range = max(abs(left_val), abs(right_val))  # 取对称范围
    return dynamic_range / 127.


if __name__ == '__main__':
    np.random.seed(1)

    # 1000 个标准正态分布的浮点数，模拟真实的激活值分布
    data_float32 = np.random.randn(1000).astype('float32')
    print('input ', data_float32)

    # 对比两种 scale 计算方法
    scale = scale_cal(data_float32)          # Max 法
    scale2 = histgram_range(data_float32)    # 直方图法（覆盖 99%）
    print(scale, scale2)                     # 直方图法的 scale 更小 → 精度更高
    exit(1)   # 这里提前退出：下面的量化/反量化只是演示代码，把 exit 这行注释掉即可运行

    xq = quant_float_data(data_float32, scale)
    print('quant result ', xq)
    xdq = dequant_data(xq, scale)
    print('dequant result ', xdq)
    print('diff ', xdq - data_float32)
