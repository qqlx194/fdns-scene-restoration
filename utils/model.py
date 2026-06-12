import torch.nn as nn
import torch
import numpy as np
from torch import Tensor
import math

"""
此模块包含神经网络模型的各种实用工具函数，例如权重初始化、参数统计、模型加载等。
对于固定解码器隐写方法而言，init_weights 函数决定了解码器的随机初始权重。
"""

def _no_grad_normal_(tensor, mean, std):
    """
    辅助函数：在不记录梯度的情况下，用正态分布填充张量。
    用于自定义初始化方法内部。
    """
    with torch.no_grad():
        return tensor.normal_(mean, std)


def _calculate_fan_in_and_fan_out(tensor):
    """
    计算张量的 fan_in (输入连接数) 和 fan_out (输出连接数)。
    这是 Xavier 和 Kaiming 等现代初始化方法的数学基础。
    
    Args:
        tensor: 权重张量
    Returns:
        fan_in: 输入单元数量
        fan_out: 输出单元数量
    """
    dimensions = tensor.dim()
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")

    num_input_fmaps = tensor.size(1)
    num_output_fmaps = tensor.size(0)
    receptive_field_size = 1
    if tensor.dim() > 2:
        # math.prod is not always available, accumulate the product manually
        # we could use functools.reduce but that is not supported by TorchScript
        for s in tensor.shape[2:]:
            receptive_field_size *= s
    fan_in = num_input_fmaps * receptive_field_size
    fan_out = num_output_fmaps * receptive_field_size

    return fan_in, fan_out


def lecun_normal_(tensor: Tensor, gain: float = 1.) -> Tensor:
    r"""
    使用 Lecun 正态分布初始化输入张量。
    参考自: `Understanding the difficulty of training deep feedforward neural networks` - Glorot, X. & Bengio, Y. (2010)
    
    标准差 std 计算公式:
    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{1}{\text{fan\_in}}}
    
    这种初始化方法旨在使前向传播时的方差保持不变。

    Args:
        tensor: n维 torch.Tensor
        gain: 可选的缩放因子
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(1.0 / float(fan_in))

    return _no_grad_normal_(tensor, 0., std)



def init_weights(model, random_seed=None):
    """
    初始化解码网络权重。
    
    在固定解码器设置中，解码网络是固定的但随机初始化的。这组随机权重就是
    提取秘密信息的“私钥”。只有拥有完全相同初始权重的网络才能正确解码。
    
    Args:
        model: 要初始化的解码神经网络模型
        random_seed: 随机种子 (int)。如果提供，则每次运行生成的权重是一样的 (可复现)。
                     如果不提供，则每次运行权重不同 (一次性密钥)。
    """
    if random_seed != None:
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            # nn.init.kaiming_normal_(m.weight) 
            # nn.init.orthogonal_(m.weight) 
            # nn.init.uniform_(m.weight) 
            # nn.init.normal_(m.weight)
            # lecun_normal_(m.weight)
            

             
            # used in fnns for hiding binary bits stream
            # m.weight.data = nn.Parameter(torch.tensor(np.random.normal(0, 1, m.weight.shape)).float())  

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight)
            nn.init.constant_(m.bias, 0)


def shuffle_params(m):
    """
    强制重置层的参数为标准正态分布 N(0, 1)。
    这是一个非常激进的重置操作，比常规初始化方差大得多。
    通常用于需要极高随机性和不可预测性的场景。
    
    Args:
        m: 网络层 (Conv2d 或 BatchNorm2d)
    """
    if type(m)==nn.Conv2d or type(m)==nn.BatchNorm2d:
        param = m.weight
        m.weight.data = nn.Parameter(torch.tensor(np.random.normal(0, 1, param.shape)).float())

        param = m.bias
        m.bias.data = nn.Parameter(torch.zeros(len(param.view(-1))).float().reshape(param.shape))


def describe_model(model):
    """
    打印模型的详细结构信息，包括类名、参数总数和层级结构。
    用于调试和模型概览。
    """
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    msg = '\n'
    msg += 'models name: {}'.format(model.__class__.__name__) + '\n'
    msg += 'Params number: {}'.format(sum(map(lambda x: x.numel(), model.parameters()))) + '\n'
    msg += 'Net structure:\n{}'.format(str(model)) + '\n'
    return msg


def load_model(model, model_load_path):
    """
    加载模型权重，并自动处理 DataParallel 带来的前缀问题。
    例如，多卡训练保存的权重会有 'module.conv1.weight'，而单卡模型只有 'conv1.weight'。
    该函数会自动去除 'module.' 前缀以确保加载成功。
    
    Args:
        model: 目标模型
        model_load_path: .pth 权重文件路径
    """
    weights_state_dict = torch.load(model_load_path)

    weights_dict = {}
    for k, v in weights_state_dict.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        weights_dict[new_k] = v
        
    model.load_state_dict(weights_dict)
    
    return model

def preprocess_state_dict(state_dict):
    """
    与 load_model 逻辑相同，但仅处理 state_dict 字典并返回，不执行实际加载。
    用于需要在加载前检查或修改权重的场景。
    """
    processed_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '') if 'module' in k else k
        processed_state_dict[new_k] = v

    return processed_state_dict

def get_parameter_number(model):
    """
    统计模型参数量。
    
    Returns:
        total_num: 总参数量
        trainable_num: 可训练参数量 (requires_grad=True)
    """
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num
