import torch
import config as c
import torchvision
from io import BytesIO
from PIL import Image
import numpy as np
import torch
import random
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F

"""
此文件包含了各种可微和非可微的图像失真层，用于模拟真实的攻击环境。
在对抗训练过程中，这些层被用来增强生成扰动的鲁棒性。
"""

def gaussian_noise_layer(adv_pert, cover=None):
    """
    高斯噪声层（智能合并版）
    
    Args:
        adv_pert: 对抗扰动
        cover: 载体图像 (可选)
        
    逻辑:
        1. 如果没有提供 cover: 执行简易模式，仅对扰动加噪 (原函数1)。
        2. 如果提供了 cover: 执行完整模式，对合成图像加噪并截断 (原函数2)。
    """
    if cover is None:
        sigma = getattr(c, 'sigma', getattr(c, 'gaussian_std', 0.01) * 255)
        return adv_pert + torch.randn(adv_pert.shape).mul_(sigma / 255).to(adv_pert.device)
    
    device = adv_pert.device
    adv_image = cover + adv_pert

    gaussian_std = c.gaussian_std
    if isinstance(gaussian_std, (tuple, list)):
        if len(gaussian_std) == 2 and all(isinstance(x, (int, float)) for x in gaussian_std):
            std_low, std_high = float(min(gaussian_std)), float(max(gaussian_std))
            gaussian_std = random.uniform(std_low, std_high)
        elif len(gaussian_std) > 0:
            gaussian_std = random.choice(list(gaussian_std))

    noise = torch.randn_like(adv_image) * float(gaussian_std)
    noisy_image = torch.clamp(adv_image + noise, 0.0, 1.0)
    
    noise_residual = (noisy_image - adv_image).detach()
    noisy_pert = adv_pert + noise_residual
    return noisy_pert

def poisson_noise_layer(adv_pert):
    """
    泊松噪声层（仅对扰动加噪）
    模拟光子计数噪声。
    """
    return  adv_pert + torch.poisson(torch.rand(adv_pert.shape)).mul_(c.sigma/255).to(adv_pert.device)

transform_to_pil = torchvision.transforms.ToPILImage()
transform_to_tensor = torchvision.transforms.ToTensor()
ps = torch.nn.PixelShuffle(c.psf)
pus = torch.nn.PixelUnshuffle(c.psf)

def jpeg_compression_layer(adv_pert, cover):
    """
    JPEG 压缩层 (非可微，使用 PIL 实现)
    用于在测试阶段模拟 JPEG 有损压缩攻击。
    
    Args:
        adv_pert: 对抗扰动
        cover: 载体图像
    Returns:
        adv_pert + jpeg_noise: 经过压缩后带有失真的扰动
    """
    adv_image = cover + adv_pert
    adv_image = adv_image.squeeze(dim=0).cpu()
    adv_image = transform_to_pil(adv_image)
    
    qf = c.qf
    if isinstance(qf, (tuple, list)):
        if len(qf) == 2 and all(isinstance(x, (int, float)) for x in qf):
            qf_low, qf_high = int(min(qf)), int(max(qf))
            qf = random.randint(qf_low, qf_high)
        elif len(qf) > 0:
            qf = random.choice(list(qf))

    outputIoStream = BytesIO()
    adv_image.save(outputIoStream, "JPEG", quality=int(qf))
    outputIoStream.seek(0)
    adv_image_jpeg = Image.open(outputIoStream)
    
    adv_image_jpeg = transform_to_tensor(adv_image_jpeg).unsqueeze(dim=0).to(adv_pert.device)
    
    jpeg_noise = (adv_image_jpeg - (cover + adv_pert)).detach()
    return adv_pert + jpeg_noise


def contrast_adjustment_layer(adv_pert, cover):
    """
    调整对比度的函数 (可微)
    输入：
        adv_pert - 对抗扰动张量 [batch_size, channels, height, width]
        cover - 原始图像张量 [batch_size, channels, height, width]
    输出：
        contrast_pert - 调整对比度后的对抗扰动张量
    """
    device = adv_pert.device
    adv_image = cover + adv_pert

    contrast_factor = c.contrast_factor
    mean = torch.mean(adv_image, dim=(2, 3), keepdim=True)
    contrast_image = mean + contrast_factor * (adv_image - mean)

    contrast_image = torch.clamp(contrast_image, 0.0, 1.0)

    contrast_residual = (contrast_image - adv_image).detach()

    contrast_pert = adv_pert + contrast_residual

    return contrast_pert

def salt_and_pepper_noise_layer(adv_pert, cover, density=0.001):
    """
    椒盐噪声层
    density: 噪声比例 (0~1)
    """
    adv_image = cover + adv_pert
    device = adv_pert.device
    
    rand = torch.rand_like(adv_image)
    
    salt_mask = (rand < density / 2).float()
    pepper_mask = (rand > 1 - density / 2).float()
    
    noisy_image = adv_image * (1 - salt_mask - pepper_mask) + salt_mask
    
    residual = (noisy_image - adv_image).detach()
    return adv_pert + residual

def add_gaussian_noise(image, std=c.gaussian_std):
    """辅助函数：直接对图像加高斯噪声"""
    if isinstance(std, (tuple, list)):
        if len(std) == 2 and all(isinstance(x, (int, float)) for x in std):
            std_low, std_high = float(min(std)), float(max(std))
            std = random.uniform(std_low, std_high)
        elif len(std) > 0:
            std = random.choice(list(std))
    noise = torch.randn_like(image) * std
    return torch.clamp(image + noise, 0.0, 1.0)

def img_jpeg_compression(image, quality=c.qf):
    """辅助函数：直接对 PIL 图像进行 JPEG 压缩并转回 Tensor"""
    if isinstance(quality, (tuple, list)):
        if len(quality) == 2 and all(isinstance(x, (int, float)) for x in quality):
            q_low, q_high = int(min(quality)), int(max(quality))
            quality = random.randint(q_low, q_high)
        elif len(quality) > 0:
            quality = random.choice(list(quality))

    image_np = (image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
    image_pil = Image.fromarray(image_np)
    
    output = BytesIO()
    image_pil.save(output, "JPEG", quality=quality)
    output.seek(0)
    
    compressed_pil = Image.open(output)
    compressed_tensor = torchvision.transforms.ToTensor()(compressed_pil).unsqueeze(0).to(image.device)
    return compressed_tensor

def dwt_init(x):
    """
    离散小波变换 (Discrete Wavelet Transform) 初始化
    将图像分解为 LL, LH, HL, HH 四个频带。
    """
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]
    LL = x1 + x2 + x3 + x4
    LH = -x1 - x2 + x3 + x4
    HL = -x1 + x2 - x3 + x4
    HH = x1 - x2 - x3 + x4
    return LL, LH, HL, HH

def idwt_init(LL, LH, HL, HH):
    """
    逆离散小波变换 (Inverse DWT)
    """
    batch, channel, h, w = LL.shape
    x = torch.zeros((batch, channel, h*2, w*2), device=LL.device)
    x[:, :, 0::2, 0::2] = (LL - LH - HL + HH) / 2
    x[:, :, 1::2, 0::2] = (LL - LH + HL - HH) / 2
    x[:, :, 0::2, 1::2] = (LL + LH - HL - HH) / 2
    x[:, :, 1::2, 1::2] = (LL + LH + HL + HH) / 2
    return x


def rotation_attack_layer(adv_pert, cover, max_angle=0.1):
    """
    旋转攻击层 - 用于对抗训练流程 (非完全可微，使用 PIL)
    输入:
        adv_pert - 对抗扰动张量 [batch_size, channels, height, width]
        cover - 原始图像张量 [batch_size, channels, height, width]
        max_angle - 最大旋转角度(度), 控制攻击强度
    输出:
        rotated_pert - 旋转后的对抗扰动张量
    """
    device = adv_pert.device
    adv_image = cover + adv_pert
    batch_size = adv_image.shape[0]

    angles = (torch.rand(batch_size) * 2 * max_angle - max_angle)

    rotated_image = torch.zeros_like(adv_image)
    for i in range(batch_size):
        img_pil = transform_to_pil(adv_image[i].cpu())
        img_pil = img_pil.rotate(angles[i].item(), resample=Image.BILINEAR, expand=False)
        rotated_image[i] = transform_to_tensor(img_pil).to(device)

    rotation_residual = (rotated_image - adv_image).detach()
    rotated_pert = adv_pert + rotation_residual

    return rotated_pert

def scaling_attack_layer(adv_pert, cover, scale_factor=0.95):
    """
    缩放攻击层 - 用于对抗训练流程 (可微)
    输入:
        adv_pert - 对抗扰动张量 [batch_size, channels, height, width]
        cover - 原始图像张量 [batch_size, channels, height, width]
        scale_factor - 缩放因子 (0-1之间), 控制攻击强度
    输出:
        scaled_pert - 缩放后的对抗扰动张量
    """
    device = adv_pert.device
    adv_image = cover + adv_pert
    batch_size, _, h, w = adv_image.shape

    scaled_h = int(h * scale_factor)
    scaled_w = int(w * scale_factor)

    scaled_down = F.interpolate(adv_image, size=(scaled_h, scaled_w),
                                mode='bilinear', align_corners=False)

    scaled_image = F.interpolate(scaled_down, size=(h, w),
                                 mode='bilinear', align_corners=False)

    scale_residual = (scaled_image - adv_image).detach()
    scaled_pert = adv_pert + scale_residual

    return scaled_pert

def jpeg_compression_dwt_layer(cover, LL_adv_pert):
    """
    针对 DWT 域隐写的 JPEG 压缩模拟层。
    
    该函数首先对载体进行 DWT 分解，将扰动添加到 LL (低频) 分量，
    然后 IDWT 重建图像，经过 JPEG 压缩，再 DWT 分解回来，
    计算 LL 分量的所有变化量。
    
    Args:
        cover: 原始载体图像
        LL_adv_pert: 添加到 LL 频带的对抗扰动
    """
    LL, LH, HL, HH = dwt_init(cover)
    LL_new = LL + LL_adv_pert
    adv_image = idwt_init(LL_new, LH, HL, HH)
    adv_image = adv_image.squeeze(dim=0).cpu()
    adv_image = transform_to_pil(adv_image)
    outputIoStream = BytesIO()
    qf = c.qf
    if isinstance(qf, (tuple, list)):
        if len(qf) == 2 and all(isinstance(x, (int, float)) for x in qf):
            q_low, q_high = int(min(qf)), int(max(qf))
            qf = random.randint(q_low, q_high)
        elif len(qf) > 0:
            qf = random.choice(list(qf))
    adv_image.save(outputIoStream, "JPEG", quality=int(qf))
    outputIoStream.seek(0)
    adv_image_jpeg = Image.open(outputIoStream)
    adv_image_jpeg = transform_to_tensor(adv_image_jpeg).unsqueeze(dim=0).to(cover.device)
    LL_JPEG, LH_JPEG, HL_JPEG, HH_JPEG = dwt_init(adv_image_jpeg)
    jpeg_noise = (LL_JPEG - LL_new).detach()
    return LL_adv_pert + jpeg_noise

def attack_layer(adv_pert, cover):
    """
    统一攻击层调度函数。
    根据 config.py 中的 c.attack_layer 配置，选择对应的攻击方式。
    
    Args:
        adv_pert: 对抗扰动
        cover: 载体图像
    Returns:
        经过攻击层处理后的扰动
    """
    if c.attack_layer == 'gaussian':
        return gaussian_noise_layer(adv_pert, cover)
    elif c.attack_layer == 'possion':
        return poisson_noise_layer(adv_pert)
    elif c.attack_layer == 'contrast':
        return contrast_adjustment_layer(adv_pert, cover)
    elif c.attack_layer == 'scale':
        return scaling_attack_layer(adv_pert, cover, c.scale_factor)
    elif c.attack_layer == 'rotate':
        return rotation_attack_layer(adv_pert, cover, c.max_angle)
    else:  # jpeg
        return jpeg_compression_layer(adv_pert, cover)




def add_contrast_adjustment(adv_image):
    """
    添加对比度调整到对抗扰动图像
    输入：
        adv_image - 张量 [batch_size, channels, height, width]，值范围[0,1]
    输出：
        contrast_image - 调整对比度后的张量
    """
    device = adv_image.device
    mean = torch.mean(adv_image, dim=(2, 3), keepdim=True)
    contrast_factor = c.contrast_factor
    contrast_image = mean + contrast_factor * (adv_image - mean)
    contrast_image = torch.clamp(contrast_image, 0.0, 1.0)
    if adv_image.requires_grad:
        contrast_image = contrast_image - adv_image.detach() + adv_image
    return contrast_image


def add_scaling_attack(adv_image, scale_factor=0.95):
    """
    添加缩放攻击到对抗图像
    输入:
        adv_image - 张量 [batch_size, channels, height, width]
        scale_factor - 缩放因子 (0-1之间), 控制攻击强度
    输出:
        scaled_image - 缩放后的图像
    """
    device = adv_image.device
    batch_size, _, h, w = adv_image.shape

    scaled_h = int(h * scale_factor)
    scaled_w = int(w * scale_factor)

    scaled_down = F.interpolate(adv_image, size=(scaled_h, scaled_w),
                                mode='bilinear', align_corners=False)

    scaled_image = F.interpolate(scaled_down, size=(h, w),
                                 mode='bilinear', align_corners=False)

    if adv_image.requires_grad:
        scaled_image = scaled_image - adv_image.detach() + adv_image

    return scaled_image


def add_rotation_attack(adv_image, max_angle=0.1):
    """
    添加旋转攻击到对抗图像
    输入:
        adv_image - 张量 [batch_size, channels, height, width]
        max_angle - 最大旋转角度(度), 控制攻击强度
    输出:
        rotated_image - 旋转后的图像
    """
    device = adv_image.device
    batch_size = adv_image.shape[0]

    angles = (torch.rand(batch_size) * 2 * max_angle - max_angle).to(device)

    rotated_image = torch.zeros_like(adv_image)
    for i in range(batch_size):
        img_pil = transform_to_pil(adv_image[i].cpu())
        img_pil = img_pil.rotate(angles[i].item(), resample=Image.BILINEAR, expand=False)
        rotated_image[i] = transform_to_tensor(img_pil).to(device)

    if adv_image.requires_grad:
        rotated_image = rotated_image - adv_image.detach() + adv_image

    return rotated_image

