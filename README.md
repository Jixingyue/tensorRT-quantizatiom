# TensorRT 量化教程（配套代码）· 快速上手指南

> 本仓库是 B 站视频《TensorRT 量化教程》的配套代码，共 **6 个独立小脚本**，从「手写量化算法」一路走到「用官方库量化真实模型」。
> **适合完全没接触过量化的初学者**，建议按 part1 → part6 的顺序阅读和运行。
>
> ⭐ 如果觉得本教程有帮助，欢迎给原仓库点个 Star：[GitHub](https://github.com/shouxieai/tensorRT_quantization)

---

## 一、这个仓库在讲什么？

深度学习模型的权重和激活值默认是 **float32**（32 位浮点）。**模型量化（Quantization）** 就是想办法把它们压缩成 **INT8**（8 位整数），好处：

- 显存占用减少约 4 倍；
- 推理速度大幅提升（INT8 计算远快于 FP32）；
- 精度损失通常可以控制在很小的范围。

本仓库用 6 个 part 带你从零走完这条链路：

| 文件 | 主题 | 独立可运行 | 需要 GPU | 难度 |
| --- | --- | --- | --- | --- |
| part1-export-onnx.py | 把预训练 ResNet50 导出为 ONNX 模型 | ✅ | ❌ | ⭐ |
| part2-Symmetric_quantization.py | 手写「带零点的量化/反量化」（UINT8，0~255） | ✅ | ❌ | ⭐ |
| part3-Asymmetric_dequantize.py | 手写「对称量化」（INT8，-127~127）+ 直方图动态范围 | ✅ | ❌ | ⭐⭐ |
| part4-Kl.py | KL 散度入门 + 直方图区间合并 | ✅ | ❌ | ⭐⭐ |
| part5-TensorRT-KL.py | TensorRT 熵校准（KL）算法完整实现 | ✅ | ❌ | ⭐⭐⭐ |
| part6.py | 用 NVIDIA pytorch_quantization 库量化 ResNet50 并导出 ONNX | ✅ | ✅ | ⭐⭐⭐ |

> ⚠️ **命名提醒**：part2 文件名是 "Symmetric"，但代码实现的是**非对称量化**（带零点 z）；part3 文件名是 "Asymmetric"，但实现的是**对称量化**。**以代码实际逻辑为准**，别被文件名误导。

---

## 二、环境准备

```bash
# 建议 Python 3.8+
pip install torch torchvision numpy matplotlib scipy

# part6 额外需要 NVIDIA 官方量化库（需要 GPU + CUDA）
pip install pytorch-quantization
# 若上面装不上，可加上 NVIDIA 的软件源再试：
# pip install pytorch-quantization --extra-index-url https://pypi.ngc.nvidia.com
```

需要联网下载的步骤：part1 会下载 ResNet50 预训练权重（约 100MB），请保证网络可用。

---

## 三、快速开始

按顺序运行即可，每个脚本都会在终端打印结果：

```bash
# 1. 导出 ONNX（生成 resnet50-1.onnx，首次运行需下载权重）
python3 part1-export-onnx.py

# 2. 手写量化/反量化，观察量化误差（diff 越小，精度损失越小）
python3 part2-Symmetric_quantization.py

# 3. 对比 Max 法与直方图法求 scale 的差别
python3 part3-Asymmetric_dequantize.py

# 4. KL 散度入门
python3 part4-Kl.py

# 5. TensorRT 熵校准核心算法（会弹出直方图；无图形界面的服务器请用 MPLBACKEND=Agg）
MPLBACKEND=Agg python3 part5-TensorRT-KL.py

# 6. 用官方库量化 ResNet50（需要 GPU，生成 quant_resnet50_*.onnx）
python3 part6.py
```

### 预期输出（供对照）

**part2**（手写非对称量化，结果稳定）：

```text
input [-0.61 -0.52  1.62]
scale and z  0.008745098114013672 70.0
quant result  [  0.  11. 255.]
dequant result  [-0.61215687 -0.5159608   1.6178433 ]
diff [-0.00215685  0.00403917 -0.00215673]
```

含义：量化 → 反量化后，数值被还原到接近原值，误差（diff）约为千分之一量级。

**part3**（对比两种 scale，结果稳定）：

```text
0.031170099739014634 0.02238892382524145
```

左边是 Max 法 scale，右边是直方图法 scale —— **直方图法的更小**，说明它剔除了离群点、为常见数值保留了更细的精度。

**part4**：最后一行是"区间合并"前后的 KL 散度（约 0.13），数值越小说明合并带来的分布变化越小。

**part5**（KL 校准，激活数据是随机生成的，**每次运行数值会变**，看规律即可）：

```text
最大的激活值 4562.18
threshold 所在组: 1814
threshold 所在组的区间范围: 4040.95
```

关键点：最大激活值 4562，但 KL 校准给出的最优阈值约 **4040** —— 尾部那约 10% 的离群点被"牺牲"了，换取了主体部分更高的量化精度。这正是熵校准比"直接取最大值"更聪明的地方。

---

## 四、目录结构

```text
tensorRT-quantization/
├── README.md                            ← 本文件（快速上手指南）
├── 量化PPT.pptx                         ← 视频配套的讲解 PPT
├── part1-export-onnx.py                 ← 导出 ONNX 模型
├── part2-Symmetric_quantization.py      ← 手写量化/反量化（非对称，UINT8）
├── part3-Asymmetric_dequantize.py       ← 对称量化 + 直方图动态范围
├── part4-Kl.py                          ← KL 散度入门
├── part5-TensorRT-KL.py                 ← TensorRT 熵校准算法（核心）
├── part6.py                             ← pytorch_quantization 库实战
└── （运行后生成）resnet50-1.onnx / quant_resnet50_replace_to_quantization.onnx
```

---

## 五、核心概念速览（30 秒读懂量化）

### 1. 量化就是"分段线性映射"

把连续的浮点区间映射到离散的整数点，需要**量化公式**（forward）和**反量化公式**（inverse）：

```text
量化:   xq = round(x / scale + z)      （xq 被裁剪到合法整数范围）
反量化: x  ≈ (xq - z) * scale
```

- **非对称量化**：`[x.min, x.max] → [0, 255]`，需要 `scale`（缩放系数）+ `z`（零点）。→ part2
- **对称量化**：`[-max|x|, max|x|] → [-127, 127]`，只需 `scale`，没有零点。→ part3

### 2. 动态范围怎么取？三种方法

- **Max**：直接取 `max|x|`，简单但被离群点拖累（part3 的 `scale_cal`）。
- **直方图/百分位**：统计直方图后去掉两头离群点，覆盖 99% 数据即可（part3 的 `histgram_range`）。
- **Entropy（KL）**：把"量化后分布"与"原始分布"做 KL 散度，选信息丢失最少的阈值（part5，即 TensorRT 默认方法）。

### 3. KL 散度 = 两个分布的距离

`KL(p||q) = Σ p[i]·log(p[i]/q[i])`，越小说明量化前后分布越接近。→ part4 入门，part5 实战。

### 4. PTQ 与 QAT

- **PTQ（训练后量化）**：模型训练完再量化，无需重新训练，**本仓库全部属于此类**。
- **QAT（量化感知训练）**：训练时就模拟量化误差，精度更好但更麻烦。

### 5. 本仓库的完整链路

手写算法理解原理（part2~part5）→ 用官方库替换模块、导出带 QDQ 节点的 ONNX（part6）→ 交给 TensorRT 转成 INT8 引擎（下一步）。

---

## 六、常见坑

1. **part1 的 `pretrained=True` 已过时**：新版 torchvision 会报警告（仍能运行），建议改成 `models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)`。
2. **part6 需要 GPU**：`model.cuda()` 在没有 NVIDIA GPU 的机器上会直接报错。
3. **part6 的库名**：安装名是 `pytorch-quantization`，导入名是 `pytorch_quantization`，别搞混。
4. **part3 里有一句 `exit(1)`**：它把后面的量化/反量化演示跳过了，想看完整效果把它注释掉再运行。
5. **无图形界面的服务器跑 part4/part5**：`plt.show()` 会提示无法显示，加环境变量 `MPLBACKEND=Agg` 即可（上面命令已带）。
6. **part5 的随机性**：它的模拟激活值每次运行都不同，输出数值会变，属正常现象。

---

## 七、原教程目录（视频内容）

本仓库代码对应"基础部分"中的 **1.2、1.3、2.x** 章节。

### 基础部分（开源）

1. **模型量化原理**
   - 1.1 量化的定义及意义
   - 1.2 对称量化与非对称量化（对应 part2、part3 代码）
   - 1.3 动态范围的常用计算方法：Max / Histgram / Entropy（对应 part3、part5）
   - 1.4 PTQ 与 QAT 介绍
   - 1.5 手写一个带 op 的量化程序
2. **TensorRT Quantization Library**
   - 2.1 Quantizer 的理解（对应 part6 的 TensorQuantizer）
   - 2.2 InputQuant / MixQuant 的理解
   - 2.3 自动插入 QDQ 节点（对应 part6）
   - 2.4 手动插入 QDQ 节点
   - 2.5 如何量化一个自定义层
   - 2.6 敏感层分析
   - 2.7 踩坑实录

### 实战部分（付费）

实战部分内容需要付费购买，购买链接请见原仓库。

---

*视频教程：[B 站链接](https://www.bilibili.com/video/BV18L41197Uz/)* · *原仓库：[GitHub](https://github.com/shouxieAI/tensorRT_quantization)*
