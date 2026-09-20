"""
part6 —— 使用 NVIDIA pytorch_quantization 库，量化一个真实的 ResNet50
====================================================================
part5 是手写 KL 算法，这一 part 直接用官方库完成同样的事：
  1. 把模型里的普通模块（Conv2d / Linear 等）替换成"可量化的模块"
     （模块内部会自动挂上 TensorQuantizer，负责统计激活分布、做 fake-quant）；
  2. 通过 torch.onnx.export 导出带 QDQ（Quantize-Dequantize）节点的 ONNX，
     这种 ONNX 可被 TensorRT 识别，从而部署成 INT8 引擎。

需要环境：GPU + pytorch_quantization 库（pip install pytorch-quantization）。
"""

import torch
import torchvision
from pytorch_quantization import tensor_quant
from pytorch_quantization import quant_modules
from pytorch_quantization import nn as quant_nn
from pytorch_quantization.nn.modules import _utils as quant_nn_utils
from pytorch_quantization import calib
from typing import List, Callable, Union, Dict


class disable_quantization:
    """上下文管理器（配合 with 使用：with disable_quantization(model): ...）。
    进入时把模型里所有 TensorQuantizer 的 _disabled 置为 True（暂停量化），
    退出时自动恢复。典型用途：跳过某些对量化特别敏感的层（如第一层 conv1）。
    """

    def __init__(self, model):
        self.model = model

    def apply(self, disabled=True):
        """遍历模型所有子模块，把所有 TensorQuantizer 的开关状态设为 disabled。"""
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = disabled

    def __enter__(self):
        self.apply(True)

    def __exit__(self, *args, **kwargs):
        self.apply(False)


class enable_quantization:
    """与 disable_quantization 相反：进入时开启量化，退出时关闭。"""

    def __init__(self, model):
        self.model = model

    def apply(self, enabled=True):
        for name, module in self.model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                module._disabled = not enabled

    def __enter__(self):
        self.apply(True)
        return self

    def __exit__(self, *args, **kwargs):
        self.apply(False)


def quantizer_state(module):
    """打印模型里所有量化器（TensorQuantizer）的状态，方便调试时查看哪些层被量化了。"""
    for name, module in module.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print(name, module)


def transfer_torch_to_quantization(nninstance: torch.nn.Module, quantmodule):
    """把"普通 PyTorch 模块"转换成"对应的可量化模块"。

    做法（仿照官方 QuantizedMixin 的实现思路）：
      1. quant_instance = quantmodule.__new__(quantmodule)  —— 先创建空壳对象（不走 __init__）；
      2. 把原模块的所有属性（权重、偏置等参数）拷贝过去，不重新初始化；
      3. 调用 quantmodule 的 __init__ 逻辑，为输入/权重挂上 TensorQuantizer；
      4. 若校准器是 HistogramCalibrator，打开 _torch_hist 以加速校准（直方图校准）。
    """
    quant_instance = quantmodule.__new__(quantmodule)
    for k, val in vars(nninstance).items():
        setattr(quant_instance, k, val)

    def __init__(self):
        if isinstance(self, quant_nn_utils.QuantInputMixin):
            # 只量化输入（部分层只对输入量化、不对权重量化）
            quant_desc_input = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__, input_only=True)
            self.init_quantizer(quant_desc_input)

            # 打开 torch_hist 以提升校准速度
            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
        else:
            # 同时量化输入和权重（如 Conv2d、Linear 等常见层）
            quant_desc_input, quant_desc_weight = quant_nn_utils.pop_quant_desc_in_kwargs(self.__class__)
            self.init_quantizer(quant_desc_input, quant_desc_weight)

            if isinstance(self._input_quantizer._calibrator, calib.HistogramCalibrator):
                self._input_quantizer._calibrator._torch_hist = True
                self._weight_quantizer._calibrator._torch_hist = True

    __init__(quant_instance)
    return quant_instance


def replace_to_quantization_module(model: torch.nn.Module,
                                   ignore_policy: Union[str, List[str], Callable] = None):
    """递归地把模型中的普通模块替换成可量化模块。

    替换依据 quant_modules._DEFAULT_QUANT_MAP：这是官方库维护的一张映射表，
    记录了 Conv2d→QuantConv2d、Linear→QuantLinear 等"原类型 -> 量化类型"的对应关系。
    （ignore_policy 参数预留了"跳过某些层"的功能，本 demo 未实现。）

    注意：这里没有调用 quant_modules.initialize()，而是手动完成替换，
    可以更清楚地看到"替换"这一步到底做了什么。
    """
    # 建立"模块类型 id -> 量化模块类型"的映射
    module_dict = {}
    for entry in quant_modules._DEFAULT_QUANT_MAP:
        module = getattr(entry.orig_mod, entry.mod_name)
        module_dict[id(module)] = entry.replace_mod

    def recursive_and_replace_module(module, prefix=""):
        """深度优先遍历模型的所有子模块：
        如果子模块的类型命中了映射表，就把它替换成对应的量化版本。"""
        for name in module._modules:
            submodule = module._modules[name]
            path = name if prefix == "" else prefix + "." + name
            recursive_and_replace_module(submodule, path)   # 先处理更深的子模块

            submodule_id = id(type(submodule))
            if submodule_id in module_dict:
                # 命中映射表 → 用 transfer_torch_to_quantization 换成量化模块
                module._modules[name] = transfer_torch_to_quantization(submodule, module_dict[submodule_id])

    recursive_and_replace_module(model)


# ==================== 主流程 ====================

# quant_modules.initialize()   # （官方推荐的另一种做法：一行代码自动替换所有模块）

model = torchvision.models.resnet50()   # 加载 ResNet50（未带预训练权重，这里只演示结构替换）
model.cuda()                            # 需要 GPU

# 下面两行是示例用法（本 demo 中已注释掉）：
# disable_quantization(model.conv1).apply()   # 跳过 conv1 层的量化
# quantizer_state(model)                      # 打印各层的量化器状态

replace_to_quantization_module(model)         # 手动把模型替换成"可量化版本"

# 构造输入，导出带 QDQ 节点的 ONNX
inputs = torch.randn(1, 3, 224, 224, device='cuda')
quant_nn.TensorQuantizer.use_fb_fake_quant = True   # 使用 fake-quant 模式（导出 QDQ 节点）
torch.onnx.export(model, inputs, 'quant_resnet50_replace_to_quantization.onnx', opset_version=13)
