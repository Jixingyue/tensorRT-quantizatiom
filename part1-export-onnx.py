"""
part1 —— 导出 ONNX 模型
=======================
目标：把一个预训练的 ResNet50（图像分类模型）导出成 ONNX 格式的模型文件，
供后续量化 / TensorRT 转换使用。

ONNX（Open Neural Network Exchange）是一个开放的模型交换格式，
相当于模型的"通用语言"，方便模型在不同框架和推理引擎（如 TensorRT）之间迁移。
"""

import torch
import torchvision.models as models

# 加载 ImageNet 预训练的 ResNet50 分类模型（首次运行会自动联网下载约 100MB 权重）
model = models.resnet50(pretrained=True)

# 构造一个"假的输入张量"：形状 (1, 3, 224, 224)
# 含义：1 张图片，3 个颜色通道（RGB），分辨率 224x224（ResNet 的标准输入尺寸）
# 注意：变量名 input 覆盖了 Python 内置的 input() 函数，这里仅作演示，无实际影响
input = torch.randn(1, 3, 224, 224)

# 导出 ONNX：
#   model            —— 要导出的模型
#   input            —— 样例输入：给模型一个假输入，前向一遍以"画出"计算图
#   "resnet50-1.onnx" —— 导出的文件名
torch.onnx.export(model, input, "resnet50-1.onnx")

# 提示：新版 torchvision 中 pretrained=True 已弃用（会弹警告但能用），
# 推荐写法：
#   model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
