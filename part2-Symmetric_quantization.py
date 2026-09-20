"""
part2 —— 手写"带零点的量化 / 反量化"（UINT8，整数范围 0~255）
============================================================
量化四件套：求 scale(缩放系数) + z(零点) → 量化 → 反量化。

本文件实现的是【非对称量化】：用 [最小值, 最大值] 确定映射范围，
同时需要 scale 和零点 z 两个参数。
⚠️ 注意：文件名虽是 Symmetric，但代码实现的是非对称量化（详见 README 说明）。
"""

import numpy as np


def saturate(x, int_max, int_min):
    """数值饱和（截断）：
    把 x 中超出 [int_min, int_max] 范围的值裁剪到边界上，
    保证量化结果落在合法的整数区间内，防止溢出。
    """
    return np.clip(x, int_min, int_max)


def scale_z_cal(x, int_max, int_min):
    """计算量化参数：缩放系数 scale 和 零点 z。

    原理：把浮点范围 [x.min, x.max] 线性映射到整数范围 [int_min, int_max]：
      scale = (x.max - x.min) / (int_max - int_min)   # 一个整数格子代表的浮点大小
      z     = int_max - round(x.max / scale)          # 浮点 0 对应的整数位置（零点）
    返回 (scale, z)，供后续 quant / dequant 使用。
    """
    scale = (x.max() - x.min()) / (int_max - int_min)
    z = int_max - np.round((x.max() / scale))
    return scale, z


def quant_float_data(x, scale, z, int_max, int_min):
    """浮点 -> 整数（量化）：
      xq = round(x / scale + z)   # 先"缩放 + 平移"到整数域，再四舍五入
      最后用 saturate 裁剪到合法整数范围。
    """
    xq = saturate(np.round(x / scale + z), int_max, int_min)
    return xq


def dequant_data(xq, scale, z):
    """整数 -> 浮点（反量化）：
      x ≈ (xq - z) * scale        # 量化过程的逆运算
    注意：反量化只能"近似"还原原值，会产生量化误差（量化不可避免的代价）。
    """
    x = ((xq - z) * scale).astype('float32')
    return x


if __name__ == '__main__':
    # 三组用于演示的浮点数据
    np.random.seed(1)                  # 固定随机种子，保证每次运行结果一致（可复现）
    data_float32 = np.random.randn(3).astype('float32')
    data_float32[0] = -0.61
    data_float32[1] = -0.52
    data_float32[2] = 1.62
    print("input", data_float32)

    # 本 demo 使用 UINT8：合法整数范围 0~255
    int_max = 255
    int_min = 0

    # 1) 求量化参数：scale 和 零点 z
    scale, z = scale_z_cal(data_float32, int_max, int_min)
    print("scale and z ", scale, z)

    # 2) 量化：float32 -> 整数
    data_int8 = quant_float_data(data_float32, scale, z, int_max, int_min)
    print("quant result ", data_int8)

    # 3) 反量化：整数 -> float32
    data_dequnat_float = dequant_data(data_int8, scale, z)
    print("dequant result ", data_dequnat_float)

    # 4) 量化-反量化前后的误差（绝对值越小，精度损失越小）
    print('diff', data_dequnat_float - data_float32)
