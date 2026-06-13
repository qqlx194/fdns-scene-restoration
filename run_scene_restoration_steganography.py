import numpy as np
import torch
import torch.nn.functional as F
from imageio.v2 import imread
from torch import nn
import random
import os
import sys
import re
from PIL import Image
from math import log10, sqrt, ceil
from skimage.transform import resize
import glob
import json
from typing import Tuple
from natsort import natsorted
import lpips
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from models.decodingNetwork128 import decodingNetwork as DecodingNetwork128

from models.decodingNetwork256 import decodingNetwork as DecodingNetwork256
from models.network_dncnn import DnCNN
from utils.model import init_weights
from utils.image import calculate_ssim, calculate_psnr, calculate_mae, calculate_nc
from utils.logger import logging, logger_info
from utils.dir import mkdirs
from utils.distortion_layers_all import (
    jpeg_compression_layer, img_jpeg_compression, gaussian_noise_layer, 
    add_gaussian_noise, contrast_adjustment_layer, add_contrast_adjustment,
    salt_and_pepper_noise_layer
)
import config as c

c.qf = 80
c.gaussian_std = 0.07
c.sp_density = 0.01
c.contrast_factor = 0.9

if c.secret_image_size == 128:
    decodingNetwork = DecodingNetwork128_CBAM if getattr(c, 'use_cbam_decoder_128', False) else DecodingNetwork128
elif c.secret_image_size == 256:
    decodingNetwork = DecodingNetwork256
else:
    raise ValueError(
        f"Unsupported secret_image_size={c.secret_image_size}. "
        "Please set secret_image_size to 128 or 256, or add a matching decodingNetwork implementation."
    )

base_dir = "results"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
secret_dataset = os.path.basename(os.path.normpath(c.secret_dataset_dir))
cover_dataset = os.path.basename(os.path.normpath(c.cover_dataset_dir))
experiment_name = str(getattr(c, 'experiment_name', 'fdns_scene_restoration')).strip()
exp_tag = str(getattr(c, 'exp_tag', '')).strip()
logger_name = experiment_name
image_save_dirs = os.path.join(base_dir, experiment_name, secret_dataset)
if exp_tag:
    image_save_dirs = os.path.join(image_save_dirs, exp_tag)
mkdirs(image_save_dirs)
logger_info(logger_name, log_path=os.path.join(image_save_dirs, 'result.log'))
logger = logging.getLogger(logger_name)

# Keep complete file logging, but silence terminal output from this logger.
for _h in list(logger.handlers):
    if isinstance(_h, logging.StreamHandler) and not isinstance(_h, logging.FileHandler):
        logger.removeHandler(_h)

logger.info('secret dataset: {:s}'.format(secret_dataset))
logger.info('cover dataset: {:s}'.format(cover_dataset))
logger.info('secret dataset dir: {}'.format(os.path.abspath(c.secret_dataset_dir)))
logger.info('cover dataset dir: {}'.format(os.path.abspath(c.cover_dataset_dir)))
logger.info('beta: {:.2f}'.format(c.beta))
logger.info('gamma: {:.5f}'.format(c.gamma))
logger.info('learning rate: {:.3f}'.format(c.lr))
logger.info('epsilon: {:.2f}'.format(c.eps))
logger.info('number of iterations: {}'.format(c.iters))
logger.info('the size of secret image: {}'.format(c.secret_image_size))
logger.info('the size of cover image: {}'.format(c.cover_image_size))
logger.info('Add JPEG layer before the decoding network: {}'.format(c.add_jpeg_layer))

logger.info('use_l3_attack: {}'.format(getattr(c, 'use_l3_attack', False)))
logger.info('l3_start_ratio: {:.2f}'.format(float(getattr(c, 'l3_start_ratio', 0.0))))
logger.info('l3_prob: {:.2f}'.format(float(getattr(c, 'l3_prob', 0.0))))
logger.info('l3_weight_max: {:.2f}'.format(float(getattr(c, 'l3_weight_max', 0.0))))
logger.info('l3_num_attacks_per_iter: {}'.format(int(getattr(c, 'l3_num_attacks_per_iter', 0))))
logger.info('l3_attack_agg: {}'.format(str(getattr(c, 'l3_attack_agg', 'mean'))))
logger.info('hid_warmup_ratio: {:.2f}'.format(float(getattr(c, 'hid_warmup_ratio', 0.0))))
logger.info('w_rev: {:.3f}'.format(float(getattr(c, 'w_rev', 1.0))))
logger.info('w_hid_base: {:.3f}'.format(float(getattr(c, 'w_hid_base', 0.0))))
logger.info('w_hid_max: {:.3f}'.format(float(getattr(c, 'w_hid_max', 0.0))))
logger.info('w_tv: {:.4f}'.format(float(getattr(c, 'w_tv', 0.0))))

os.environ["CUDA_VISIBLE_DEVICES"] = c.gpu_id
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _require_file(path: str, hint: str) -> None:
    """检查文件是否存在，不存在则抛出异常"""
    if os.path.isfile(path):
        return
    raise FileNotFoundError(
        f"Required file not found: {path}\n"
        f"Hint: {hint}"
    )


def _require_non_empty_image_dir(path: str) -> None:
    """检查目录是否存在且不为空"""
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"Dataset directory not found: {path}\n"
            "Hint: create the directory and put images inside, or update config.py."
        )
    image_count = len(glob.glob(os.path.join(path, '*')))
    if image_count == 0:
        raise FileNotFoundError(
            f"Dataset directory is empty: {path}\n"
            "Hint: put images (png/jpg/...) into this folder, or update config.py."
        )


def _list_image_files(path: str):
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp")
    files = []
    for ext in exts:
        files.extend(glob.glob(os.path.join(path, ext)))
    return natsorted(files)


def _extract_numeric_suffix(stem: str):
    m = re.search(r"(\d+)$", stem)
    return m.group(1) if m else None


def _build_mask_suffix_index(mask_files: list):
    """Build numeric-suffix -> mask_path index, keeping first occurrence on collisions."""
    suffix_index = {}
    duplicate_suffixes = []
    for path in mask_files:
        stem = os.path.splitext(os.path.basename(path))[0]
        suffix = _extract_numeric_suffix(stem)
        if suffix is None:
            continue
        if suffix in suffix_index and suffix_index[suffix] != path:
            duplicate_suffixes.append(suffix)
            continue
        suffix_index[suffix] = path

    if len(duplicate_suffixes) > 0:
        try:
            logger.warning(
                "Duplicate mask numeric suffixes detected (%d unique duplicates). "
                "Will use the first occurrence for suffix matching.",
                len(set(duplicate_suffixes)),
            )
        except Exception:
            pass

    return suffix_index


def _resolve_mask_path_for_secret(
    secret_image_path: str,
    *,
    secret_index: int = None,
    mask_files: list = None,
    mask_suffix_index: dict = None,
) -> Tuple[str, str]:
    """Resolve corresponding mask path for a given secret image.

    Strategy:
      1) Same basename under c.secret_mask_dataset_dir.
      2) If mask dir contains exactly one image, reuse it for all.
    """
    mask_dir = getattr(c, "secret_mask_dataset_dir", "")
    if not mask_dir:
        raise FileNotFoundError(
            "secret_mask_dataset_dir is empty. Hint: set it in config.py to enable mask loss."
        )

    base = os.path.splitext(os.path.basename(secret_image_path))[0]
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
        cand = os.path.join(mask_dir, base + ext)
        if os.path.isfile(cand):
            return cand, "basename"

    # Numeric suffix mapping (preferred for names like secret3000029 -> mask3000029)
    secret_suffix = _extract_numeric_suffix(base)
    if secret_suffix is not None:
        if mask_suffix_index is None:
            if mask_files is None:
                mask_files = _list_image_files(mask_dir)
            mask_suffix_index = _build_mask_suffix_index(mask_files)
        matched = mask_suffix_index.get(secret_suffix)
        if matched is not None:
            return matched, "numeric_suffix"

    if mask_files is None:
        mask_files = _list_image_files(mask_dir)
    if len(mask_files) == 1:
        return mask_files[0], "single_mask"

    # Fallback: index-based mapping for repos that keep generic mask names
    # (e.g., secret.png/secret1.png ... with mask.png/mask1.png ...).
    if secret_index is not None and len(mask_files) > 0:
        chosen = mask_files[int(secret_index) % int(len(mask_files))]
        try:
            logger.warning(
                'No basename/numeric-suffix matched mask for secret "%s"; falling back to index-mapped mask: %s',
                os.path.basename(secret_image_path),
                os.path.basename(chosen),
            )
        except Exception:
            pass
        return chosen, "index_fallback"

    raise FileNotFoundError(
        f"No matching mask for secret image: {secret_image_path}\n"
        f"Looked in: {mask_dir}\n"
        "Hint: (1) name mask files the same as secret images, OR (2) use matching numeric suffixes (secretXXXX <-> maskXXXX), OR (3) keep exactly one mask image in the directory, OR (4) pass secret_index to enable index-based mapping."
    )


def _load_secret_mask_tensor(
    secret_image_path: str,
    device: torch.device,
    *,
    secret_index: int = None,
    mask_files: list = None,
    mask_suffix_index: dict = None,
) -> Tuple[torch.Tensor, str, str]:
    """Load a binary mask aligned with the secret image.

    Returns tensor of shape [1, 1, W, H] to match this file's (C, W, H) convention.
    Values are float in {0.0, 1.0}.
    """
    mask_path, match_mode = _resolve_mask_path_for_secret(
        secret_image_path,
        secret_index=secret_index,
        mask_files=mask_files,
        mask_suffix_index=mask_suffix_index,
    )
    mask_np = imread(mask_path, pilmode='L')
    mask_np = resize(
        mask_np,
        (c.secret_image_size, c.secret_image_size),
        order=0,
        preserve_range=True,
        anti_aliasing=False,
    )
    mask_np = (mask_np > 127.5).astype(np.float32)

    if float(mask_np.sum()) <= 0.0:
        raise ValueError(
            f"Loaded secret mask is empty (all zeros): {mask_path}\n"
            "Hint: ensure the mask marks valid target content as 255 and padding as 0."
        )

    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    mask_t = mask_t.permute(0, 1, 3, 2).contiguous()  # -> [1,1,W,H]
    return mask_t.to(device), mask_path, match_mode


def _masked_mse_loss(pred: torch.Tensor, target: torch.Tensor, mask_t: torch.Tensor) -> torch.Tensor:
    """Mask-weighted MSE over valid secret region.

    pred/target: [1, C, W, H]
    mask_t:      [1, 1, W, H] (float {0,1})
    """
    if mask_t is None:
        return torch.mean((pred - target) ** 2)

    mask3 = mask_t.repeat(1, pred.shape[1], 1, 1)
    diff2 = (pred - target) ** 2
    denom = torch.clamp(mask3.sum(), min=1e-12)
    return (diff2 * mask3).sum() / denom


def _mask_hw_from_tensor(secret_mask_t: torch.Tensor) -> np.ndarray:
    """Convert mask tensor [1,1,W,H] to numpy mask [H,W] with values in {0,1}."""
    mask_hw = secret_mask_t.clone().squeeze().detach().cpu().numpy().T
    return (mask_hw > 0.5).astype(np.float32)


def _calculate_psnr_masked(secret_np: np.ndarray, secret_rev_np: np.ndarray, mask_hw: np.ndarray) -> float:
    """Masked PSNR, consistent with utils.image.calculate_psnr."""
    valid = mask_hw > 0.5
    if int(valid.sum()) <= 0:
        return 0.0
    v1 = secret_np[valid].astype(np.float32).reshape(-1)
    v2 = secret_rev_np[valid].astype(np.float32).reshape(-1)
    return float(calculate_psnr(v1, v2))


def _calculate_nc_masked(img1: np.ndarray, img2: np.ndarray, mask_hw: np.ndarray) -> float:
    """Masked NC, consistent with utils.image.calculate_nc.

    We slice valid pixels (mask==1) and then reuse calculate_nc so the
    definition matches the one used elsewhere in this repo.
    """
    valid = mask_hw > 0.5
    if int(valid.sum()) <= 0:
        return 0.0
    v1 = img1[valid].astype(np.float64).reshape(-1)
    v2 = img2[valid].astype(np.float64).reshape(-1)
    return float(calculate_nc(v1, v2))


def _calculate_ssim_masked(secret_np: np.ndarray, secret_rev_np: np.ndarray, mask_hw: np.ndarray) -> float:
    mask3 = np.repeat(mask_hw[:, :, None], 3, axis=2).astype(np.float32)
    return float(calculate_ssim(secret_np * mask3, secret_rev_np * mask3))


def _calculate_lpips_masked(img1: np.ndarray, img2: np.ndarray, mask_hw: np.ndarray) -> float:
    mask3 = np.repeat(mask_hw[:, :, None], 3, axis=2).astype(np.float32)
    return float(calculate_lpips(img1 * mask3, img2 * mask3))

def calculate_lpips(img1, img2):
    """
    计算两张图像的LPIPS距离 (Learned Perceptual Image Patch Similarity)
    用于评估图像的感知质量

    参数：
    img1 : numpy.ndarray - 输入图像1 (H x W x C) [0-255]
    img2 : numpy.ndarray - 输入图像2 (H x W x C) [0-255]
    """
    if not hasattr(calculate_lpips, 'loss_fn'):
        calculate_lpips.loss_fn = lpips.LPIPS(net='alex').eval()
        if torch.cuda.is_available():
            calculate_lpips.loss_fn = calculate_lpips.loss_fn.cuda()

    def process_image(img):
        img_t = torch.from_numpy(img).float() / 127.5 - 1.0
        img_t = img_t.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
        if torch.cuda.is_available():
            img_t = img_t.cuda()
        return img_t

    img1_t = process_image(img1)
    img2_t = process_image(img2)

    with torch.no_grad():
        distance = calculate_lpips.loss_fn(img1_t, img2_t)

    lpips_value = distance.item()

    if lpips_value == 0:
        return float('inf')

    return lpips_value


def run_robustness_tests(
    stego_tensor,
    cover_tensor,
    secret_gt_tensor,
    decoding_model,
    denoise_model,
    block_positions,
    secret_mask_hw: np.ndarray = None,
    save_dir=None,
    img_idx=None
):
    results = []
    if img_idx is None:
        image_stem = "image"
    else:
        image_stem = os.path.splitext(os.path.basename(str(img_idx)))[0]

    def attack_slug(attack_name: str) -> str:
        slug = str(attack_name).lower()
        slug = slug.replace("&", "_").replace("%", "")
        slug = re.sub(r"[^a-z0-9]+", "_", slug).strip("_")
        return slug or "attack"

    def save_attack_outputs(attack_name, stego_attacked_np, secret_pred_np):
        if save_dir is None or not bool(getattr(c, "save_images", True)):
            return
        slug = attack_slug(attack_name)
        secret_out_dir = os.path.join(save_dir, "secret_rev_robust_jpegloss", slug)
        stego_out_dir = os.path.join(save_dir, "stego_robust_jpegloss", slug)
        mkdirs(secret_out_dir)
        mkdirs(stego_out_dir)
        Image.fromarray(np.clip(secret_pred_np, 0, 255).astype(np.uint8)).save(
            os.path.join(secret_out_dir, image_stem + ".png")
        )
        Image.fromarray(np.clip(stego_attacked_np, 0, 255).astype(np.uint8)).save(
            os.path.join(stego_out_dir, image_stem + ".png")
        )
    secret_gt_np = secret_gt_tensor.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255
    cover_np = cover_tensor.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255

    def decode_and_post_process(attacked_stego):
        pert_attacked = reverse_crop_and_rearrange_no_loop(attacked_stego - cover_tensor, block_positions)
        secret_pred = decoding_model(pert_attacked)
        if denoise_model is not None:
            with torch.no_grad():
                secret_pred = denoise_model(secret_pred)
        secret_pred = torch.round(torch.clamp(secret_pred * 255, min=0., max=255.)) / 255
        secret_pred_np = secret_pred.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255
        stego_attacked_np = attacked_stego.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255
        return stego_attacked_np, secret_pred_np

    def evaluate_metrics(attack_name, stego_attacked_np, secret_pred_np):
        stego_psnr = float(calculate_psnr(cover_np, stego_attacked_np))
        stego_ssim = float(calculate_ssim(cover_np, stego_attacked_np))
        stego_lpips = float(calculate_lpips(cover_np, stego_attacked_np))

        if secret_mask_hw is not None:
            sec_nc = _calculate_nc_masked(secret_gt_np, secret_pred_np, secret_mask_hw)
            sec_psnr = _calculate_psnr_masked(secret_gt_np, secret_pred_np, secret_mask_hw)
            sec_ssim = _calculate_ssim_masked(secret_gt_np, secret_pred_np, secret_mask_hw)
            sec_lpips = _calculate_lpips_masked(secret_gt_np, secret_pred_np, secret_mask_hw)
        else:
            sec_nc = calculate_nc(secret_gt_np, secret_pred_np)
            sec_psnr = calculate_psnr(secret_gt_np, secret_pred_np)
            sec_ssim = calculate_ssim(secret_gt_np, secret_pred_np)
            sec_lpips = calculate_lpips(secret_gt_np, secret_pred_np)

        if secret_mask_hw is not None:
            sec_psnr = _calculate_psnr_masked(secret_gt_np, secret_pred_np, secret_mask_hw)
            sec_ssim = _calculate_ssim_masked(secret_gt_np, secret_pred_np, secret_mask_hw)

        results.append({
            "Attack": attack_name,
            "NC": sec_nc,
            "Sec_PSNR": sec_psnr,
            "Sec_SSIM": sec_ssim,
            "Sec_LPIPS": sec_lpips,
            "Stego_PSNR": stego_psnr,
            "Stego_SSIM": stego_ssim,
            "Stego_LPIPS": stego_lpips,
        })
        save_attack_outputs(attack_name, stego_attacked_np, secret_pred_np)

    # 1. JPEG Compression Test
    stego_attacked = img_jpeg_compression(stego_tensor, quality=80)
    stego_attacked_np, secret_pred_np = decode_and_post_process(stego_attacked)
    evaluate_metrics("JPEG 80%", stego_attacked_np, secret_pred_np)

    # 2. Gaussian Noise Test
    stego_attacked = add_gaussian_noise(stego_tensor, std=0.07)
    stego_attacked_np, secret_pred_np = decode_and_post_process(stego_attacked)
    evaluate_metrics("Gaussian 0.07", stego_attacked_np, secret_pred_np)

    # 3. Contrast Adjustment Test
    stego_attacked = cover_tensor + contrast_adjustment_layer(stego_tensor - cover_tensor, cover_tensor)
    stego_attacked_np, secret_pred_np = decode_and_post_process(stego_attacked)
    evaluate_metrics("Contrast 0.9", stego_attacked_np, secret_pred_np)

    # 4. Salt and Pepper Noise Test
    adv_pert_full = stego_tensor - cover_tensor
    pert_noisy = salt_and_pepper_noise_layer(adv_pert_full, cover_tensor, density=0.01)
    stego_attacked = cover_tensor + pert_noisy
    stego_attacked_np, secret_pred_np = decode_and_post_process(stego_attacked)
    evaluate_metrics("Salt&Pepper 0.01", stego_attacked_np, secret_pred_np)

    return results

if c.cover_image_size // c.secret_image_size == 1:
    down_ratio_l3 = 1;
    down_ratio_l2 = 1
elif c.cover_image_size // c.secret_image_size == 2:
    down_ratio_l3 = 2;
    down_ratio_l2 = 1
elif c.cover_image_size // c.secret_image_size == 4:
    down_ratio_l3 = 2;
    down_ratio_l2 = 2
else:
    print('The code does not take into account the current situation, please adjust the image resulation')

def calculate_lbp_complexity(img_patch):
    """
    计算8x8图像块的LBP纹理复杂度（基于直方图熵）
    输入：img_patch - [8, 8] 灰度张量
    输出：entropy - 纹理复杂度标量值
    """
    lbp = torch.zeros_like(img_patch, dtype=torch.int32)

    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, 1), (1, 1), (1, 0),
               (1, -1), (0, -1)]

    center = img_patch[1:-1, 1:-1]
    inner_h = center.shape[0]
    inner_w = center.shape[1]
    if inner_h <= 0 or inner_w <= 0:
        return torch.zeros((), device=img_patch.device, dtype=torch.float32)

    binary_codes = torch.zeros((8, inner_h, inner_w), dtype=torch.bool, device=img_patch.device)

    for k, (dy, dx) in enumerate(offsets):
        neighbor = img_patch[1 + dy:1 + dy + inner_h, 1 + dx:1 + dx + inner_w]
        binary_codes[k] = (neighbor > center)

    lbp_values = torch.zeros((inner_h, inner_w), dtype=torch.int32, device=img_patch.device)
    for k in range(8):
        lbp_values |= binary_codes[k].int() << (7 - k)

    lbp[1:-1, 1:-1] = lbp_values

    valid_lbp = lbp[1:-1, 1:-1].flatten().float()
    hist = torch.histc(valid_lbp, bins=256, min=0, max=255)
    hist = hist[hist > 0] + 1e-10
    prob = hist / hist.sum()
    entropy = -torch.sum(prob * torch.log2(prob))

    return entropy


def calculate_multiscale_lbp_scores(gray_cover: torch.Tensor, w8: float = 0.7, w16: float = 0.3) -> torch.Tensor:
    """Compute per-8x8 block complexity with 8x8/16x16 LBP entropy fusion.

    Score for each 8x8 block i:
        S_i = w8 * E_i^8 + w16 * E_parent(i)^16
    """
    # 8x8 blocks (base scale)
    blocks8_h = gray_cover.shape[0] // 8
    blocks8_w = gray_cover.shape[1] // 8
    patches8 = (
        gray_cover[:blocks8_h * 8, :blocks8_w * 8]
        .unfold(0, 8, 8)
        .unfold(1, 8, 8)
        .contiguous()
        .view(-1, 8, 8)
    )
    entropy8 = torch.stack([calculate_lbp_complexity(patch) for patch in patches8])

    # 16x16 blocks (parent scale)
    blocks16_h = gray_cover.shape[0] // 16
    blocks16_w = gray_cover.shape[1] // 16
    patches16 = (
        gray_cover[:blocks16_h * 16, :blocks16_w * 16]
        .unfold(0, 16, 16)
        .unfold(1, 16, 16)
        .contiguous()
        .view(-1, 16, 16)
    )
    entropy16 = torch.stack([calculate_lbp_complexity(patch) for patch in patches16])

    # Map each 8x8 block to its parent 16x16 block.
    idx8 = torch.arange(blocks8_h * blocks8_w, device=gray_cover.device)
    row8 = idx8 // blocks8_w
    col8 = idx8 % blocks8_w
    parent_row = row8 // 2
    parent_col = col8 // 2
    parent_idx = parent_row * blocks16_w + parent_col
    entropy16_parent = entropy16[parent_idx.long()]

    return float(w8) * entropy8 + float(w16) * entropy16_parent


def calculate_lbp_var_grad_complexity(img_patch, var_weight=1.0, grad_weight=1.0):
    """LBP-VAR score: LBP_Entropy * log(1 + Variance_255)."""
    _ = var_weight
    _ = grad_weight
    entropy = calculate_lbp_complexity(img_patch)
    variance_255 = torch.var(img_patch * 255.0)
    var_score = torch.log1p(variance_255)
    return entropy * var_score


def _compute_sobel_gradients(gray_cover: torch.Tensor):
    """Return Sobel gx, gy and gradient magnitude for grayscale image [H,W]."""
    x = gray_cover.unsqueeze(0).unsqueeze(0)
    sobel_x = torch.tensor(
        [[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]],
        device=gray_cover.device,
        dtype=gray_cover.dtype,
    ).view(1, 1, 3, 3) / 8.0
    sobel_y = torch.tensor(
        [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]],
        device=gray_cover.device,
        dtype=gray_cover.dtype,
    ).view(1, 1, 3, 3) / 8.0

    gx = F.conv2d(x, sobel_x, padding=1).squeeze(0).squeeze(0)
    gy = F.conv2d(x, sobel_y, padding=1).squeeze(0).squeeze(0)
    grad_mag = torch.sqrt(gx * gx + gy * gy + 1e-12)
    return gx, gy, grad_mag


def calculate_texture_structure_scores(gray_cover: torch.Tensor, block_size: int = 8):
    """Return per-block T/C/E scores.

    T_j: mean gradient magnitude
    C_j: orientation coherence = (lambda1-lambda2)/(lambda1+lambda2+eps)
    E_j: structure energy = lambda1+lambda2
    """
    gx, gy, grad_mag = _compute_sobel_gradients(gray_cover)

    t_scores = (
        grad_mag.unfold(0, block_size, block_size)
        .unfold(1, block_size, block_size)
        .contiguous()
        .view(-1, block_size, block_size)
        .mean(dim=(1, 2))
    )

    jxx = (
        (gx * gx).unfold(0, block_size, block_size)
        .unfold(1, block_size, block_size)
        .contiguous()
        .view(-1, block_size, block_size)
        .mean(dim=(1, 2))
    )
    jyy = (
        (gy * gy).unfold(0, block_size, block_size)
        .unfold(1, block_size, block_size)
        .contiguous()
        .view(-1, block_size, block_size)
        .mean(dim=(1, 2))
    )
    jxy = (
        (gx * gy).unfold(0, block_size, block_size)
        .unfold(1, block_size, block_size)
        .contiguous()
        .view(-1, block_size, block_size)
        .mean(dim=(1, 2))
    )

    eps = 1e-12
    anis_num = torch.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy ** 2) + eps)
    e_scores = jxx + jyy
    c_scores = anis_num / (e_scores + eps)
    return t_scores, c_scores, e_scores


def _blockify_feature_map(feature_hw: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    return (
        feature_hw.unfold(0, block_size, block_size)
        .unfold(1, block_size, block_size)
        .contiguous()
        .view(-1, block_size, block_size)
    )


def _light_jpeg_attack_gray(gray_cover: torch.Tensor, quality: int = 90) -> torch.Tensor:
    """Apply differentiable-light JPEG attack on grayscale via 3-channel replication."""
    x3 = gray_cover.unsqueeze(0).repeat(3, 1, 1).unsqueeze(0)
    y3 = img_jpeg_compression(x3, quality=int(quality))
    return y3[0, 0]


def _blur_gray(gray_cover: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
    k = int(kernel_size)
    if k < 3:
        k = 3
    if (k % 2) == 0:
        k += 1
    kernel = torch.ones((1, 1, k, k), device=gray_cover.device, dtype=gray_cover.dtype) / float(k * k)
    x = gray_cover.unsqueeze(0).unsqueeze(0)
    y = F.conv2d(x, kernel, padding=k // 2)
    return y.squeeze(0).squeeze(0)


def calculate_attack_consistency_scores(gray_cover: torch.Tensor, block_size: int = 8) -> torch.Tensor:
    """Compute per-block attack consistency R_j in [0,1] with light attacks.

    R_j = 1 - mean_k( ||phi(B_j)-phi(A_k(B_j))||_1 / (||phi(B_j)||_1 + eps) )
    phi uses gradient magnitude map per 8x8 block.
    """
    _, _, grad_mag_ref = _compute_sobel_gradients(gray_cover)
    phi_ref = _blockify_feature_map(grad_mag_ref, block_size=block_size)
    base_l1 = phi_ref.abs().sum(dim=(1, 2))

    attacks = []
    jpeg_quality = int(getattr(c, 'texture_r_jpeg_quality', 90))
    gaussian_std_255 = float(getattr(c, 'texture_r_gaussian_std', 1.0))
    blur_kernel = int(getattr(c, 'texture_r_blur_kernel', 3))

    try:
        attacks.append(_light_jpeg_attack_gray(gray_cover, quality=jpeg_quality))
    except Exception as e:
        logger.warning('JPEG light attack failed in R-score, skipping JPEG branch: %s', str(e))

    gaussian_std = max(0.0, gaussian_std_255) / 255.0
    attacks.append(torch.clamp(gray_cover + torch.randn_like(gray_cover) * gaussian_std, 0.0, 1.0))
    attacks.append(_blur_gray(gray_cover, kernel_size=blur_kernel))

    if len(attacks) == 0:
        return torch.ones_like(base_l1)

    rel_diffs = []
    eps = 1e-8
    for attacked_gray in attacks:
        _, _, grad_mag_att = _compute_sobel_gradients(attacked_gray)
        phi_att = _blockify_feature_map(grad_mag_att, block_size=block_size)
        diff_l1 = (phi_ref - phi_att).abs().sum(dim=(1, 2))
        rel = diff_l1 / (base_l1 + eps)
        rel_diffs.append(torch.clamp(1.0 - rel, min=0.0, max=1.0))

    return torch.stack(rel_diffs, dim=0).mean(dim=0)


def _normalize_to_unit_interval(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if int(scores.numel()) == 0:
        return scores
    s_min = torch.min(scores)
    s_max = torch.max(scores)
    if float((s_max - s_min).item()) < eps:
        return torch.ones_like(scores)
    return (scores - s_min) / (s_max - s_min + eps)


def _resolve_threshold(scores: torch.Tensor, abs_name: str, quantile_name: str, default_quantile: float) -> float:
    abs_v = getattr(c, abs_name, None)
    if abs_v is not None:
        return float(abs_v)

    q = float(getattr(c, quantile_name, default_quantile))
    q = max(0.0, min(1.0, q))
    return float(torch.quantile(scores.detach(), q).item())


def calculate_grad_orient_scores(gray_cover: torch.Tensor, alpha: float = 1.0, beta: float = 0.6) -> torch.Tensor:
    """Per-8x8 score using gradient energy + orientation coherence.

    S_j = alpha * G_j + beta * A_j
      - G_j: mean gradient magnitude in block j
      - A_j: structure-tensor coherence in block j

    gray_cover is expected in [0,1], shape [H,W].
    """
    grad_energy, orient_coherence, _ = calculate_texture_structure_scores(gray_cover, block_size=8)
    return float(alpha) * grad_energy + float(beta) * orient_coherence


def calculate_orient_coherence_scores(gray_cover: torch.Tensor) -> torch.Tensor:
    """Per-8x8 orientation coherence scores A_j in [0,1]."""
    _, orient_coherence, _ = calculate_texture_structure_scores(gray_cover, block_size=8)
    return orient_coherence


def _standardize_scores(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Standardize scores to comparable scale for hybrid fusion."""
    if int(scores.numel()) == 0:
        return scores
    mean = torch.mean(scores)
    std = torch.std(scores, unbiased=False)
    if float(std.item()) < eps:
        return scores - mean
    return (scores - mean) / (std + eps)


def calculate_ltp_complexity(img_patch: torch.Tensor, tau: float = 0.02) -> torch.Tensor:
    """Compute LTP texture complexity (entropy) on an 8x8 grayscale patch.

    LTP coding with tolerance tau:
      diff >  tau -> 1
      |diff|<=tau -> 0
      diff < -tau -> -1

    We encode 8 neighbors as a base-3 number in [0, 3^8-1] (=6560) and compute
    histogram entropy over the valid 6x6 inner pixels.

    Args:
        img_patch: [8,8] grayscale tensor in [0,1] (recommended).
        tau: tolerance threshold. If tau > 1, treated as pixel-domain [0,255] and will be scaled by /255.
    """
    # normalize tau to match img_patch scale
    tau_f = float(tau)
    if tau_f > 1.0:
        tau_f = tau_f / 255.0

    # 8-neighborhood offsets (dy, dx)
    offsets = [(-1, -1), (-1, 0), (-1, 1),
               (0, 1), (1, 1), (1, 0),
               (1, -1), (0, -1)]

    center = img_patch[1:-1, 1:-1]  # [6,6]

    # ternary digits in {0,1,2} corresponding to {-1,0,+1}
    # map: -1->0, 0->1, +1->2 (so digits are non-negative for base-3 encoding)
    digits = torch.zeros((8, 6, 6), dtype=torch.int64, device=img_patch.device)
    for k, (dy, dx) in enumerate(offsets):
        neighbor = img_patch[1 + dy:1 + dy + 6, 1 + dx:1 + dx + 6]
        diff = neighbor - center
        pos = (diff > tau_f)
        neg = (diff < -tau_f)
        # default 0-band -> digit=1
        d = torch.ones_like(center, dtype=torch.int64)
        d = torch.where(pos, torch.full_like(d, 2), d)
        d = torch.where(neg, torch.full_like(d, 0), d)
        digits[k] = d

    # base-3 encoding
    code = torch.zeros((6, 6), dtype=torch.int64, device=img_patch.device)
    for k in range(8):
        code = code * 3 + digits[k]

    # histogram entropy over 36 codes
    flat = code.reshape(-1)
    # bincount is faster and works on CPU/GPU for int tensors
    hist = torch.bincount(flat, minlength=3 ** 8).float()
    hist = hist[hist > 0] + 1e-10
    prob = hist / hist.sum()
    entropy = -torch.sum(prob * torch.log2(prob))
    return entropy


def estimate_ltp_tau_from_cover(gray_cover: torch.Tensor) -> float:
    """Estimate an adaptive LTP tolerance tau from current cover (grayscale).

    Goal: suppress sensor/noise-induced micro fluctuations so LTP complexity focuses on
    semantically meaningful structures rather than isolated noise.

    Strategy:
      - compute neighbor differences over the inner region (6x6 per 8x8 patch notion, but here on full image)
      - use a robust scale estimator (median(|diff|) or MAD)
      - tau = k * scale, then clamp to [ltp_tau_min, ltp_tau_max]

    Note: gray_cover is expected in [0,1].
    """
    method = str(getattr(c, 'ltp_tau_adaptive_method', 'median_abs_diff')).lower()
    k = float(getattr(c, 'ltp_tau_adaptive_k', 2.5))
    tau_min = float(getattr(c, 'ltp_tau_min', 0.005))
    tau_max = float(getattr(c, 'ltp_tau_max', 0.05))
    sample_n = int(getattr(c, 'ltp_tau_sample_pixels', 0))

    # inner region diffs (avoid boundary)
    center = gray_cover[1:-1, 1:-1]
    diffs = []
    for dy, dx in [(-1, -1), (-1, 0), (-1, 1),
                   (0, 1), (1, 1), (1, 0),
                   (1, -1), (0, -1)]:
        neigh = gray_cover[1 + dy:1 + dy + center.shape[0], 1 + dx:1 + dx + center.shape[1]]
        diffs.append((neigh - center).abs().reshape(-1))
    ad = torch.cat(diffs, dim=0)

    if sample_n > 0 and ad.numel() > sample_n:
        # random sample for speed
        idx = torch.randint(low=0, high=ad.numel(), size=(sample_n,), device=ad.device)
        ad = ad[idx]

    # robust scale
    if method == 'mad':
        med = ad.median()
        mad = (ad - med).abs().median()
        scale = mad
    else:
        # median absolute difference
        scale = ad.median()

    tau = float((k * scale).clamp(min=tau_min, max=tau_max).item())
    # guard against degenerate values
    if not np.isfinite(tau) or tau <= 0:
        tau = float(max(tau_min, 1e-6))
    return tau


def find_nearest_square(n):
    """
    找到不小于n的最小平方数（至少289）
    用于确定将选中的纹理块重组为正方形网格的大小
    """
    if n < 289:
        return 289
    m = ceil(sqrt(n))
    return m * m


def _resolve_ckpt_path(path: str) -> str:
    """Resolve checkpoint path.

    Accepts absolute paths or project-root-relative paths.
    """
    if not path:
        return ''
    if os.path.isabs(path) and os.path.exists(path):
        return path
    # Try relative to this script's directory (project root)
    cand = os.path.join(os.path.dirname(__file__), path)
    if os.path.exists(cand):
        return cand
    cand = os.path.join(os.path.dirname(__file__), 'external_steganalysis', path)
    if os.path.exists(cand):
        return cand
    return path


def _extract_state_dict(ckpt_obj):
    """Best-effort extraction of a state_dict from a torch.load() object."""
    if isinstance(ckpt_obj, dict):
        for k in ('state_dict', 'model_state_dict', 'net', 'model'):
            v = ckpt_obj.get(k, None)
            if isinstance(v, dict):
                return v
    return ckpt_obj


def _steganalysis_enabled():
    enabled = bool(getattr(c, 'use_grad_signals_in_steganalysis_nets', False))
    if not enabled:
        return False, [], {}

    nets = [str(x).strip().lower() for x in list(getattr(c, 'steganalysis_nets', ['srnet', 'siastegnet']))]
    nets = [n for n in nets if n]
    if len(nets) == 0:
        return False, [], {}

    ckpt_paths = {
        'siastegnet': _resolve_ckpt_path(getattr(c, 'pre_trained_siastegnet_path', '')),
        'srnet': _resolve_ckpt_path(getattr(c, 'pre_trained_srnet_path', '')),
        'yenet': _resolve_ckpt_path(getattr(c, 'pre_trained_yenet_path', '')),
    }

    required = []
    for n in nets:
        p = ckpt_paths.get(n, '')
        if not p:
            required.append(f'{n}:<empty>')
        elif not os.path.exists(p):
            required.append(f'{n}:{p}')

    if required:
        logger.warning('Disable steganalysis nets: missing checkpoints: %s', required)
        return False, nets, ckpt_paths

    return True, nets, ckpt_paths


USE_STEGANALYSIS_NETS, STEG_NETS, STEG_CKPTS = _steganalysis_enabled()
SiaStegNet = None
SRNet = None
Yenet = None
l_anti_dec = None
preprocess_data = None

if USE_STEGANALYSIS_NETS:
    _steganalysis_pkg_dir = os.path.join(os.path.dirname(__file__), 'external_steganalysis')
    if _steganalysis_pkg_dir not in sys.path:
        sys.path.insert(0, _steganalysis_pkg_dir)

    # Some third-party code inside siastegnet uses absolute imports like `import src...`.
    # To make it work, we also add the siastegnet package root (which contains `src/`) to sys.path.
    _steg_root = os.path.join(_steganalysis_pkg_dir, 'steganalysis_networks')
    _siastegnet_root = os.path.join(_steg_root, 'siastegnet')
    _srnet_root = os.path.join(_steg_root, 'srnet')
    _yenet_root = os.path.join(_steg_root, 'yenet')
    for _p in (_steg_root, _siastegnet_root, _srnet_root, _yenet_root):
        if os.path.isdir(_p) and _p not in sys.path:
            sys.path.insert(0, _p)

    def _siastegnet_preprocess_data(images, labels, random_crop):
        """Local, side-effect-free copy of siastegnet preprocessing.

        This mirrors the original val.py/train.py behavior without importing val.py,
        which has top-level checkpoint loading side effects.
        """
        if images.ndim == 5:
            images = images.squeeze(0)
            labels = labels.squeeze(0)
        h, w = images.shape[-2:]

        if random_crop:
            ch = random.randint(h * 3 // 4, h)
            cw = random.randint(w * 3 // 4, w)
            h0 = random.randint(0, h - ch)
            w0 = random.randint(0, w - cw)
        else:
            ch, cw, h0, w0 = h, w, 0, 0

        cw = cw & ~1
        inputs = [
            images[..., h0:h0 + ch, w0:w0 + cw // 2],
            images[..., h0:h0 + ch, w0 + cw // 2:w0 + cw],
        ]
        return inputs, labels

    try:
        from steganalysis_networks.siastegnet.src.models import KeNet
        from steganalysis_networks.srnet.model.model import Srnet
        from steganalysis_networks.yenet.model.YeNet import YeNet
    except Exception as e:
        logger.warning('Disable steganalysis nets: import failed: %s', str(e))
        USE_STEGANALYSIS_NETS = False
        STEG_NETS = []
        STEG_CKPTS = {}
    else:
        preprocess_data = _siastegnet_preprocess_data
        loaded = []

        if 'siastegnet' in STEG_NETS:
            try:
                SiaStegNet = KeNet().to(device)
                ckpt = torch.load(STEG_CKPTS['siastegnet'], map_location=device)
                SiaStegNet.load_state_dict(_extract_state_dict(ckpt), strict=False)
                SiaStegNet.eval()
                for p in SiaStegNet.parameters():
                    p.requires_grad_(False)
                loaded.append('siastegnet')
            except Exception as e:
                logger.warning('Failed to load SiaStegNet (%s): %s', str(STEG_CKPTS.get('siastegnet', '')), str(e))
                SiaStegNet = None

        if 'srnet' in STEG_NETS:
            try:
                SRNet = Srnet().to(device)
                ckpt = torch.load(STEG_CKPTS['srnet'], map_location=device)
                sd = _extract_state_dict(ckpt)
                try:
                    SRNet.load_state_dict(sd, strict=True)
                except Exception:
                    SRNet.load_state_dict(sd, strict=False)
                SRNet.eval()
                for p in SRNet.parameters():
                    p.requires_grad_(False)
                loaded.append('srnet')
            except Exception as e:
                logger.warning('Failed to load SRNet (%s): %s', str(STEG_CKPTS.get('srnet', '')), str(e))
                SRNet = None

        if 'yenet' in STEG_NETS:
            try:
                Yenet = YeNet().to(device)
                ckpt = torch.load(STEG_CKPTS['yenet'], map_location=device)
                sd = _extract_state_dict(ckpt)
                try:
                    Yenet.load_state_dict(sd, strict=True)
                except Exception:
                    Yenet.load_state_dict(sd, strict=False)
                Yenet.eval()
                for p in Yenet.parameters():
                    p.requires_grad_(False)
                loaded.append('yenet')
            except Exception as e:
                logger.warning('Failed to load YeNet (%s): %s', str(STEG_CKPTS.get('yenet', '')), str(e))
                Yenet = None

        if len(loaded) == 0:
            logger.warning('Disable steganalysis nets: no steganalysis model loaded successfully')
            USE_STEGANALYSIS_NETS = False
            preprocess_data = None
            l_anti_dec = None
        else:
            STEG_NETS = loaded
            l_anti_dec = nn.CrossEntropyLoss()
            logger.info('Anti-steganalysis enabled: nets=%s, last_n_iters=%s', STEG_NETS, str(getattr(c, 'steganalysis_last_n_iters', 100)))


def reverse_crop_and_rearrange_no_loop(pert_full, block_positions, block_size=8):
    """
    将网格化的扰动补丁还原回原始全图的对应位置
    pert_full: 优化后的扰动，形状为grid后的正方形
    block_positions: 原始块的位置索引列表
    """
    N, C, H, W = pert_full.shape
    blocks = (pert_full
              .unfold(2, block_size, block_size)
              .unfold(3, block_size, block_size))
    blocks = blocks.contiguous().view(N, C, -1, block_size, block_size)
    
    row_col_tensor = torch.tensor(block_positions, device=pert_full.device)  # [num_blocks, 2]
    rows, cols = row_col_tensor[:, 0], row_col_tensor[:, 1]
    block_indices = rows * (W // block_size) + cols  # [num_blocks]

    selected_blocks = blocks[:, :, block_indices, :, :]  # [1, 3, num_blocks, block_size, block_size]

    num_blocks = len(block_positions)
    grid_size = int(np.sqrt(num_blocks))

  
    
    selected_blocks = selected_blocks.view(N, C, grid_size, grid_size, block_size, block_size)

    new_pert = (selected_blocks
                .permute(0, 1, 2, 4, 3, 5)
                .contiguous()
                .view(N, C, grid_size*block_size, grid_size*block_size))

    return new_pert



model = decodingNetwork(input_channel=3 *c.psf*c.psf, output_channels=3*c.psf*c.psf, down_ratio_l2=down_ratio_l2,
                        down_ratio_l3=down_ratio_l3).to(device)
secret_postprocess = str(getattr(c, 'secret_postprocess', 'none')).strip().lower()
if secret_postprocess in ('', 'none', 'off', 'false', '0'):
    denoise_model = None
elif secret_postprocess in ('dncnn', 'dcnn'):
    denoise_model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode='R').to(device)
    denoise_model.load_state_dict(torch.load(_resolve_ckpt_path(c.dncnn_ckpt), map_location=device), strict=True)
    denoise_model.eval()
    for p in denoise_model.parameters():
        p.requires_grad_(False)
else:
    raise ValueError("Unsupported secret_postprocess: {}. Use 'none' or 'dncnn'.".format(secret_postprocess))

_require_non_empty_image_dir(c.secret_dataset_dir)
_require_non_empty_image_dir(c.cover_dataset_dir)

use_secret_mask = bool(getattr(c, 'secret_mask_dataset_dir', ''))
secret_mask_files = None
secret_mask_suffix_index = None
if use_secret_mask:
    _require_non_empty_image_dir(c.secret_mask_dataset_dir)
    secret_mask_files = _list_image_files(c.secret_mask_dataset_dir)
    if len(secret_mask_files) == 0:
        raise FileNotFoundError(
            f"Dataset directory is empty: {c.secret_mask_dataset_dir}\n"
            "Hint: put mask images into this folder, or clear secret_mask_dataset_dir in config.py to disable mask loss."
        )
    secret_mask_suffix_index = _build_mask_suffix_index(secret_mask_files)
logger.info('use_secret_mask: {}'.format(use_secret_mask))
if use_secret_mask:
    logger.info('secret_mask_dataset_dir: {}'.format(str(getattr(c, 'secret_mask_dataset_dir', ''))))

l_rev = torch.nn.MSELoss()
l_hid = torch.nn.MSELoss()
l_JPEG = torch.nn.MSELoss()

stego_psnr_list = [];
stego_ssim_list = [];
stego_lpips_list = [0];
stego_apd_list = []
secret_rev_psnr_list = [];
secret_rev_ssim_list = [];
secret_rev_lpips_list = [0];
secret_rev_apd_list = []
secret_rev_psnr_masked_list = []
secret_rev_ssim_masked_list = []
secret_rev_lpips_masked_list = []
secret_rev_nc_masked_list = []
# Collect per-attack NC across images for final Table-2-style mean summary.
robust_nc_stats = {}
secret_image_path_list = list(natsorted(glob.glob(os.path.join(c.secret_dataset_dir, '*'))))
cover_image_path_list = list(natsorted(glob.glob(os.path.join(c.cover_dataset_dir, '*'))))
secret_original_indices = list(range(len(secret_image_path_list)))

# Fixed-size random subset for sweep runs.
sample_num_images = int(getattr(c, 'sample_num_images', 0))
_seed_cfg = getattr(c, 'sample_subset_seed', 20260403)
sample_subset_seed = 20260403 if _seed_cfg is None else int(_seed_cfg)
if sample_num_images > 0:
    total_available = min(len(secret_image_path_list), len(cover_image_path_list))
    if sample_num_images > total_available:
        raise ValueError(
            f"sample_num_images={sample_num_images} exceeds available paired images={total_available}."
        )
    if sample_num_images < total_available:
        subset_rng = random.Random(sample_subset_seed)
        selected_indices = sorted(subset_rng.sample(range(total_available), sample_num_images))
        secret_image_path_list = [secret_image_path_list[idx] for idx in selected_indices]
        cover_image_path_list = [cover_image_path_list[idx] for idx in selected_indices]
        secret_original_indices = [secret_original_indices[idx] for idx in selected_indices]
        logger.info(
            'sample_num_images enabled: randomly selected {} image pairs with seed {}'.format(
                sample_num_images,
                sample_subset_seed,
            )
        )

# Quick-debug mode: optionally run only first N image pairs.
debug_num_images = int(getattr(c, 'debug_num_images', 0))
if debug_num_images > 0:
    secret_image_path_list = secret_image_path_list[:debug_num_images]
    cover_image_path_list = cover_image_path_list[:debug_num_images]
    secret_original_indices = secret_original_indices[:debug_num_images]
    logger.info('debug_num_images enabled: only first {} image pairs will be processed'.format(debug_num_images))

# Optional exact-index filtering after subset/debug slicing.
# Priority: image_index_list (exact picks) > [image_start_index, image_end_index) range.
image_index_list = list(getattr(c, 'image_index_list', []))
if len(image_index_list) > 0:
    total_pairs = min(len(secret_image_path_list), len(cover_image_path_list))
    normalized = []
    for idx in image_index_list:
        idx_i = int(idx)
        if idx_i < 0:
            idx_i = total_pairs + idx_i
        if idx_i < 0 or idx_i >= total_pairs:
            raise ValueError(
                'image_index_list contains out-of-range index {} (total pairs after sampling/debug: {}).'.format(
                    idx,
                    total_pairs,
                )
            )
        normalized.append(idx_i)

    # Keep user-provided order while removing duplicates.
    seen = set()
    selected_indices = []
    for idx_i in normalized:
        if idx_i not in seen:
            selected_indices.append(idx_i)
            seen.add(idx_i)

    secret_image_path_list = [secret_image_path_list[idx] for idx in selected_indices]
    cover_image_path_list = [cover_image_path_list[idx] for idx in selected_indices]
    secret_original_indices = [secret_original_indices[idx] for idx in selected_indices]
    logger.info('image_index_list enabled: selected local indices {}'.format(selected_indices))
else:
    image_start_index = int(getattr(c, 'image_start_index', 0))
    image_end_index = int(getattr(c, 'image_end_index', -1))
    total_pairs = min(len(secret_image_path_list), len(cover_image_path_list))

    if image_start_index < 0:
        image_start_index = total_pairs + image_start_index
    image_start_index = max(0, min(image_start_index, total_pairs))

    if image_end_index < 0:
        image_end_index = total_pairs
    image_end_index = max(image_start_index, min(image_end_index, total_pairs))

    if image_start_index > 0 or image_end_index < total_pairs:
        secret_image_path_list = secret_image_path_list[image_start_index:image_end_index]
        cover_image_path_list = cover_image_path_list[image_start_index:image_end_index]
        secret_original_indices = secret_original_indices[image_start_index:image_end_index]
        logger.info(
            'image index range enabled: processing local indices [{}, {})'.format(
                image_start_index,
                image_end_index,
            )
        )

num_of_imgs = len(secret_image_path_list)
if len(cover_image_path_list) < num_of_imgs:
    raise ValueError(
        f"Not enough cover images: cover={len(cover_image_path_list)}, secret={num_of_imgs}.\n"
        "Hint: put at least as many cover images as secret images, or reduce secret images."
    )

# ---------------- Resume support (auto checkpoint per image) ----------------
resume_enabled = bool(getattr(c, 'resume_enabled', True))
reset_resume_state = bool(getattr(c, 'reset_resume_state', False))
keep_resume_state_on_finish = bool(getattr(c, 'keep_resume_state_on_finish', False))
resume_state_path = os.path.join(image_save_dirs, 'resume_state.json')
start_index = 0

if resume_enabled and reset_resume_state and os.path.isfile(resume_state_path):
    try:
        os.remove(resume_state_path)
        logger.info('Removed existing resume state: {}'.format(resume_state_path))
    except Exception as e:
        logger.warning('Failed to remove resume state {}: {}'.format(resume_state_path, str(e)))

if resume_enabled and os.path.isfile(resume_state_path):
    try:
        with open(resume_state_path, 'r', encoding='utf-8') as f:
            state = json.load(f)

        if int(state.get('num_of_imgs', -1)) == int(num_of_imgs):
            start_index = int(state.get('next_index', 0))
            start_index = max(0, min(start_index, num_of_imgs))

            stego_psnr_list = list(state.get('stego_psnr_list', stego_psnr_list))
            stego_ssim_list = list(state.get('stego_ssim_list', stego_ssim_list))
            stego_lpips_list = list(state.get('stego_lpips_list', stego_lpips_list))
            stego_apd_list = list(state.get('stego_apd_list', stego_apd_list))

            secret_rev_psnr_list = list(state.get('secret_rev_psnr_list', secret_rev_psnr_list))
            secret_rev_ssim_list = list(state.get('secret_rev_ssim_list', secret_rev_ssim_list))
            secret_rev_lpips_list = list(state.get('secret_rev_lpips_list', secret_rev_lpips_list))
            secret_rev_apd_list = list(state.get('secret_rev_apd_list', secret_rev_apd_list))

            secret_rev_psnr_masked_list = list(state.get('secret_rev_psnr_masked_list', secret_rev_psnr_masked_list))
            secret_rev_ssim_masked_list = list(state.get('secret_rev_ssim_masked_list', secret_rev_ssim_masked_list))
            secret_rev_lpips_masked_list = list(state.get('secret_rev_lpips_masked_list', secret_rev_lpips_masked_list))
            secret_rev_nc_masked_list = list(state.get('secret_rev_nc_masked_list', secret_rev_nc_masked_list))

            loaded_robust = state.get('robust_nc_stats', {})
            if isinstance(loaded_robust, dict):
                robust_nc_stats = {
                    str(k): [float(x) for x in v] for k, v in loaded_robust.items() if isinstance(v, list)
                }

            logger.info(
                'Resume enabled: loaded state from {}. next_index={}, remaining={}'.format(
                    resume_state_path,
                    start_index,
                    max(0, num_of_imgs - start_index),
                )
            )
        else:
            logger.warning(
                'Resume state exists but num_of_imgs mismatch (state={}, current={}). Ignoring resume state.'.format(
                    state.get('num_of_imgs', None),
                    num_of_imgs,
                )
            )
    except Exception as e:
        logger.warning('Failed to load resume state {}: {}'.format(resume_state_path, str(e)))


def _save_resume_state(next_index: int) -> None:
    if not resume_enabled:
        return
    state = {
        'num_of_imgs': int(num_of_imgs),
        'next_index': int(next_index),
        'stego_psnr_list': [float(x) for x in stego_psnr_list],
        'stego_ssim_list': [float(x) for x in stego_ssim_list],
        'stego_lpips_list': [float(x) for x in stego_lpips_list],
        'stego_apd_list': [float(x) for x in stego_apd_list],
        'secret_rev_psnr_list': [float(x) for x in secret_rev_psnr_list],
        'secret_rev_ssim_list': [float(x) for x in secret_rev_ssim_list],
        'secret_rev_lpips_list': [float(x) for x in secret_rev_lpips_list],
        'secret_rev_apd_list': [float(x) for x in secret_rev_apd_list],
        'secret_rev_psnr_masked_list': [float(x) for x in secret_rev_psnr_masked_list],
        'secret_rev_ssim_masked_list': [float(x) for x in secret_rev_ssim_masked_list],
        'secret_rev_lpips_masked_list': [float(x) for x in secret_rev_lpips_masked_list],
        'secret_rev_nc_masked_list': [float(x) for x in secret_rev_nc_masked_list],
        'robust_nc_stats': {k: [float(x) for x in v] for k, v in robust_nc_stats.items()},
    }
    tmp_path = resume_state_path + '.tmp'
    with open(tmp_path, 'w', encoding='utf-8') as f:
        json.dump(state, f, ensure_ascii=True)
    os.replace(tmp_path, resume_state_path)


def _log_running_means(tag: str, n_images: int) -> None:
    """Log running means for the first n_images processed so far."""
    if n_images <= 0:
        return

    logger.info('--- {} ---'.format(tag))
    logger.info(
        'Mean ({} images) | stego_psnr={:.2f}, stego_ssim={:.4f}, stego_lpips={:.4f}, stego_apd={:.2f}'.format(
            n_images,
            float(np.array(stego_psnr_list).mean()) if len(stego_psnr_list) > 0 else float('nan'),
            float(np.array(stego_ssim_list).mean()) if len(stego_ssim_list) > 0 else float('nan'),
            float(np.array(stego_lpips_list).mean()) if len(stego_lpips_list) > 0 else float('nan'),
            float(np.array(stego_apd_list).mean()) if len(stego_apd_list) > 0 else float('nan'),
        )
    )
    logger.info(
        'Mean ({} images) | secret_rev_psnr={:.2f}, secret_rev_ssim={:.4f}, secret_rev_lpips={:.4f}, secret_rev_apd={:.2f}'.format(
            n_images,
            float(np.array(secret_rev_psnr_list).mean()) if len(secret_rev_psnr_list) > 0 else float('nan'),
            float(np.array(secret_rev_ssim_list).mean()) if len(secret_rev_ssim_list) > 0 else float('nan'),
            float(np.array(secret_rev_lpips_list).mean()) if len(secret_rev_lpips_list) > 0 else float('nan'),
            float(np.array(secret_rev_apd_list).mean()) if len(secret_rev_apd_list) > 0 else float('nan'),
        )
    )

    if len(secret_rev_psnr_masked_list) > 0:
        logger.info(
            'Mean ({} images) | secret_rev_psnr_masked={:.2f}, secret_rev_ssim_masked={:.4f}, secret_rev_lpips_masked={:.4f}, secret_rev_nc_masked={:.6f}'.format(
                n_images,
                float(np.array(secret_rev_psnr_masked_list).mean()),
                float(np.array(secret_rev_ssim_masked_list).mean()),
                float(np.array(secret_rev_lpips_masked_list).mean()) if len(secret_rev_lpips_masked_list) > 0 else float('nan'),
                float(np.array(secret_rev_nc_masked_list).mean()),
            )
        )

    if len(robust_nc_stats) > 0:
        jpeg_keys = ['JPEG 80%']
        gauss_keys = ['Gaussian 0.07']
        contrast_keys = ['Contrast 0.9']
        sp_keys = ['Salt&Pepper 0.01']

        def _mean_or_nan(k):
            vals = robust_nc_stats.get(k, [])
            if len(vals) == 0:
                return float('nan')
            return float(np.mean(vals))

        jpeg_means = [_mean_or_nan(k) for k in jpeg_keys]
        gauss_means = [_mean_or_nan(k) for k in gauss_keys]
        contrast_means = [_mean_or_nan(k) for k in contrast_keys]
        sp_means = [_mean_or_nan(k) for k in sp_keys]

        logger.info('Mean ({} images) | JPEG: q80={:.6f}'.format(n_images, jpeg_means[0]))
        logger.info('Mean ({} images) | Gaussian: 0.07={:.6f}'.format(n_images, gauss_means[0]))
        logger.info('Mean ({} images) | Contrast: 0.9={:.6f}'.format(n_images, contrast_means[0]))
        logger.info('Mean ({} images) | Salt&Pepper: 0.01={:.6f}'.format(n_images, sp_means[0]))

def get_block_view(tensor, block_size):
    return (tensor
            .unfold(2, block_size, block_size)
            .unfold(3, block_size, block_size))
for i in range(start_index, len(secret_image_path_list)):

    logger.info('*' * 60)
    logger.info('hiding {}-th image'.format(i))

    secret = imread(secret_image_path_list[i], pilmode='RGB') / 255.0
    secret = resize(secret, (c.secret_image_size, c.secret_image_size))
    secret = torch.FloatTensor(secret).permute(2, 1, 0).unsqueeze(0).to(device)

    secret_mask_t = None
    secret_mask_hw = None
    secret_source_index = int(secret_original_indices[i]) if i < len(secret_original_indices) else int(i)
    if use_secret_mask:
        secret_mask_t, resolved_mask_path, mask_match_mode = _load_secret_mask_tensor(
            secret_image_path_list[i],
            device,
            secret_index=secret_source_index,
            mask_files=secret_mask_files,
            mask_suffix_index=secret_mask_suffix_index,
        )
        secret_mask_hw = _mask_hw_from_tensor(secret_mask_t)
        logger.info(
            'mask pairing | sample_local_idx=%d | sample_source_idx=%d | secret=%s | mask=%s | mode=%s',
            i,
            secret_source_index,
            os.path.basename(secret_image_path_list[i]),
            os.path.basename(resolved_mask_path),
            mask_match_mode,
        )

    cover = imread(cover_image_path_list[i], pilmode='RGB') / 255.0
    cover = resize(cover, (c.cover_image_size, c.cover_image_size))
    cover = torch.FloatTensor(cover).permute(2, 1, 0).unsqueeze(0).to(device)
    cover_backup = cover.clone()

    random_seed_for_decodor = random.randint(0, 100000000)
    logger.info('random_seed_for_decodor(receiver): {:s}'.format(str(random_seed_for_decodor)))
    init_weights(model, random_seed_for_decodor)
    model = model.to(device)
    model.eval()

    block_size = 8
    gray_cover = 0.299 * cover[0, 0] + 0.587 * cover[0, 1] + 0.114 * cover[0, 2]
    patches = gray_cover.unfold(0, 8, 8).unfold(1, 8, 8).contiguous().view(-1, 8, 8)

    intensity_mask = torch.ones((patches.shape[0],), device=patches.device, dtype=torch.bool)

    lbp_scores = torch.stack([calculate_lbp_complexity(patch) for patch in patches])
    tau_lbp_fixed = float(getattr(c, 'texture_lbp_fixed_threshold', 4.0))
    lbp_gate_q = 0.25
    tau_lbp_median = float(torch.quantile(lbp_scores.detach(), lbp_gate_q).item())
    use_capped_median = bool(getattr(c, 'texture_lbp_use_capped_median', True))
    tau_lbp = min(tau_lbp_fixed, tau_lbp_median) if use_capped_median else tau_lbp_median
    lbp_gate_threshold = tau_lbp

    lbp_gate_count_fixed = int((lbp_scores >= tau_lbp_fixed).sum().item())
    mask_tex = (lbp_scores >= tau_lbp)
    indices_tex = torch.nonzero(mask_tex, as_tuple=False).flatten()
    lbp_gate_count = int(indices_tex.numel())

    gx, gy, _ = _compute_sobel_gradients(gray_cover)
    jxx = ((gx * gx).unfold(0, block_size, block_size).unfold(1, block_size, block_size).contiguous().view(-1, block_size, block_size).mean(dim=(1, 2)))
    jyy = ((gy * gy).unfold(0, block_size, block_size).unfold(1, block_size, block_size).contiguous().view(-1, block_size, block_size).mean(dim=(1, 2)))
    jxy = ((gx * gy).unfold(0, block_size, block_size).unfold(1, block_size, block_size).contiguous().view(-1, block_size, block_size).mean(dim=(1, 2)))

    eps_val = 1e-12
    anis_num = torch.sqrt((jxx - jyy) ** 2 + 4.0 * (jxy ** 2) + eps_val)
    e_scores = jxx + jyy
    lambda2 = (e_scores - anis_num) / 2.0

    keep_ratio_lambda2 = float(getattr(c, 'texture_lambda2_keep_ratio', 0.70))
    if indices_tex.numel() > 0:
        lambda2_tex = lambda2[indices_tex]
        l2_q = max(0.0, min(1.0, 1.0 - keep_ratio_lambda2))
        tau_l2 = float(torch.quantile(lambda2_tex.detach(), l2_q).item())
        mask_stb = (lambda2[indices_tex] >= tau_l2)
        valid_indices = indices_tex[mask_stb]
    else:
        valid_indices = indices_tex
        tau_l2 = 0.0

    gated_complexity = lbp_scores
    num_score_valid = int(indices_tex.numel())
    num_stable_valid = int(valid_indices.numel())
    selection_threshold_text = (
        f"LBP >= {tau_lbp:.4f} (q={lbp_gate_q:.2f}, q_value={tau_lbp_median:.4f}, fixed={tau_lbp_fixed:.4f}, "
        f"capped={use_capped_median}), Top {keep_ratio_lambda2*100:.0f}% lambda2 in LBP set (>= {tau_l2:.4f})"
    )
    logger.info(f"texture_descriptor: lbp_cascade_median, score=LBP_j, gate: {selection_threshold_text}")

    total_blocks_8x8 = int(gated_complexity.numel())

    num_valid = int(valid_indices.numel())
    num_intensity_valid = int(intensity_mask.sum().item())

    expected_m = num_valid if num_valid > 0 else int(total_blocks_8x8 * float(getattr(c, 'texture_final_select_ratio', 0.25)))
    grid_side = int(sqrt(expected_m))
    target_num = grid_side * grid_side
    if target_num < 1:
        target_num = 1
        grid_side = 1

    if num_valid > 0:
        valid_scores = gated_complexity[valid_indices]
        _, valid_order = torch.sort(valid_scores, descending=True)
        sorted_valid_indices = valid_indices[valid_order]
    else:
        sorted_valid_indices = torch.empty(0, dtype=torch.long, device=valid_indices.device)

    invalid_mask = torch.ones(total_blocks_8x8, dtype=torch.bool, device=patches.device)
    if num_valid > 0:
        invalid_mask[valid_indices] = False
    invalid_indices = torch.nonzero(invalid_mask, as_tuple=False).flatten()

    if invalid_indices.numel() > 0:
        invalid_scores = gated_complexity[invalid_indices]
        _, invalid_order = torch.sort(invalid_scores, descending=True)
        sorted_invalid_indices = invalid_indices[invalid_order]
    else:
        sorted_invalid_indices = torch.empty(0, dtype=torch.long, device=valid_indices.device)

    all_sorted_preferred = torch.cat([sorted_valid_indices, sorted_invalid_indices])
    selected_indices = all_sorted_preferred[:target_num]

    logger.info(
        f"Block Selection Info: Total 8x8 blocks={total_blocks_8x8} | "
        f"CandidatesAfterGate={num_score_valid} | "
        f"AfterStage2={num_stable_valid} | "
        f"IntensityValid={num_intensity_valid} | "
        f"LBP>={lbp_gate_threshold:.4f}={lbp_gate_count if lbp_gate_count is not None else 'N/A'} | "
        f"LBP>={tau_lbp_fixed:.1f}={lbp_gate_count_fixed} | "
        f"LBP_q{lbp_gate_q:.2f}={tau_lbp_median:.4f} | "
        f"UsedByThreshold(SquareDown)={target_num} | "
        f"Grid Size={grid_side}x{grid_side} ({grid_side*8}x{grid_side*8} pixels)"
    )

    grid_size = int(sqrt(target_num))
    new_size = grid_size * 8
    cover_resized = torch.zeros((1, 3, new_size, new_size), device=device)
    block_positions = []

    for j, idx in enumerate(selected_indices):
        blocks_per_side = int(cover.shape[3] // 8)
        row = idx.item() // blocks_per_side
        col = idx.item() % blocks_per_side
        block_positions.append((row, col))
        block = cover[:, :, row * 8:(row + 1) * 8, col * 8:(col + 1) * 8]
        y_in_new = (j // grid_size) * 8
        x_in_new = (j % grid_size) * 8
        cover_resized[:, :, y_in_new:y_in_new + 8, x_in_new:x_in_new + 8] = block
        
    block_positions = []
    for idx in selected_indices:
        row = idx.item() // (cover.shape[3] // 8)
        col = idx.item() % (cover.shape[3] // 8)
        block_positions.append((row, col))
        
    mask_pos = torch.gt((torch.ones_like(cover_resized) - cover_resized),
                        (torch.ones_like(cover_resized) * c.eps)).int()
    mask_neg = torch.gt(cover_resized, (torch.ones_like(cover_resized) * c.eps)).int()

    U = (torch.ones_like(cover_resized) * c.eps) * mask_pos + (torch.ones_like(cover_resized) - cover_resized) * (
                1 - mask_pos)
    L = -1 * ((torch.ones_like(cover_resized) * c.eps) * mask_neg + cover_resized * (1 - mask_neg))


    w_pert = torch.autograd.Variable(torch.zeros_like(cover_resized).float()).to(device)
    w_pert.requires_grad = True
    optimizer = torch.optim.Adam([w_pert], lr=c.lr)
    weight_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 450, gamma=0.5)


    H, W = cover.shape[2], cover.shape[3]

    grid_size = int(np.sqrt(len(block_positions)))

    row_col_tensor = torch.tensor(block_positions, device=device)  # [N, 2]
    rows, cols = row_col_tensor[:, 0], row_col_tensor[:, 1]

    block_indices = (rows * (W // block_size) + cols).long()  # [N]
    if int(block_indices.unique().numel()) != int(block_indices.numel()):
        logger.warning(
            'Detected duplicate block indices in block_positions; full-image block placement is ambiguous. '
            'Consider ensuring selected_indices/block_positions are unique.'
        )

    unfold_h = H // block_size
    unfold_w = W // block_size
    total_blocks = int(unfold_h * unfold_w)
    idx = block_indices.view(1, 1, -1, 1, 1).expand(1, 3, -1, block_size, block_size)

    hid_warmup_ratio = float(getattr(c, 'hid_warmup_ratio', 0.50))
    hid_warmup_ratio = max(0.0, min(1.0, hid_warmup_ratio))
    warmup_iters = int(c.iters * hid_warmup_ratio)
    if warmup_iters >= c.iters:
        warmup_iters = max(0, c.iters - 1)

    # L3 attack loss schedule
    use_l3 = bool(getattr(c, 'use_l3_attack', False))
    l3_prob = float(getattr(c, 'l3_prob', 1.0))
    l3_start_ratio = float(getattr(c, 'l3_start_ratio', 0.0))
    l3_weight_max = float(getattr(c, 'l3_weight_max', 1.0))
    l3_start_iters = int(c.iters * l3_start_ratio)
    if l3_start_iters >= c.iters:
        l3_start_iters = max(0, c.iters - 1)

    attack_types = ['jpeg', 'gaussian', 's&p']

    # Anti-steganalysis schedule: only enable during the last N iterations.
    steganalysis_last_n = int(getattr(c, 'steganalysis_last_n_iters', 0))
    if steganalysis_last_n <= 0:
        steganalysis_start_iters = c.iters + 1
    else:
        steganalysis_start_iters = max(0, int(c.iters) - int(steganalysis_last_n))
    steganalysis_beta = float(getattr(c, 'steganalysis_beta', 0.5))
    steganalysis_gamma = float(getattr(c, 'steganalysis_gamma', 2e-5))

    for iteration_index in range(c.iters):
        # print(iteration_index)
        optimizer.zero_grad()
        
        adv_pert = L + (U - L) * ((torch.tanh(w_pert) + 1) / 2)


        adv_blocks = get_block_view(adv_pert, block_size)  # [1, 3, unfold_H, unfold_W, block_size, block_size]
        adv_blocks = adv_blocks.contiguous().view(1, 3, -1, block_size, block_size)

        # Differentiable placement: build full-sized pert by scattering selected blocks.
        adv_blocks_sel = adv_blocks[:, :, : int(block_indices.numel())]
        full_blocks = torch.zeros(
            (1, 3, total_blocks, block_size, block_size),
            device=device,
            dtype=adv_blocks_sel.dtype,
        )
        full_blocks = full_blocks.scatter(2, idx, adv_blocks_sel)

        adv_pert_full = full_blocks.view(1, 3, unfold_h, unfold_w, block_size, block_size)
        adv_pert_full = adv_pert_full.permute(0, 1, 2, 4, 3, 5).contiguous()
        adv_pert_full = adv_pert_full.view(1, 3, H, W)


        # --- L2 / L3: clean / attacked decode losses ---
        adv_pert_input_clean = reverse_crop_and_rearrange_no_loop(adv_pert_full, block_positions)
        output_clean = model(adv_pert_input_clean)

        # L3 attacked decode loss (optional)
        loss_3 = torch.zeros((), device=device)
        w_l3 = 0.0
        if use_l3 and iteration_index >= l3_start_iters and random.random() < l3_prob:
            l3_num_attacks = int(getattr(c, 'l3_num_attacks_per_iter', 0))
            if l3_num_attacks <= 0 or l3_num_attacks >= len(attack_types):
                selected_attacks = attack_types
            else:
                selected_attacks = random.sample(attack_types, k=l3_num_attacks)

            sp_density_cfg = getattr(c, 'sp_density', 0.001)
            l3_losses = []
            for attack_name in selected_attacks:
                if attack_name == 'jpeg':
                    adv_pert_full_attacked = jpeg_compression_layer(adv_pert_full, cover)
                elif attack_name == 'gaussian':
                    adv_pert_full_attacked = gaussian_noise_layer(adv_pert_full, cover)
                elif attack_name == 'contrast':
                    adv_pert_full_attacked = contrast_adjustment_layer(adv_pert_full, cover)
                elif attack_name in ('s&p', 'saltpepper', 'salt_and_pepper'):
                    if isinstance(sp_density_cfg, (tuple, list)):
                        if len(sp_density_cfg) == 2 and all(isinstance(x, (int, float)) for x in sp_density_cfg):
                            density = random.uniform(float(min(sp_density_cfg)), float(max(sp_density_cfg)))
                        elif len(sp_density_cfg) > 0:
                            density = random.choice(list(sp_density_cfg))
                        else:
                            density = 0.001
                    else:
                        density = float(sp_density_cfg)
                    adv_pert_full_attacked = salt_and_pepper_noise_layer(adv_pert_full, cover, density=float(density))
                else:
                    adv_pert_full_attacked = adv_pert_full

                adv_pert_input_attacked = reverse_crop_and_rearrange_no_loop(adv_pert_full_attacked, block_positions)
                output_attacked = model(adv_pert_input_attacked)
                l3_losses.append(_masked_mse_loss(output_attacked, secret, secret_mask_t))

            if len(l3_losses) > 0:
                l3_stack = torch.stack(l3_losses)
                agg = str(getattr(c, 'l3_attack_agg', 'max')).lower()
                if agg == 'max':
                    loss_3 = torch.max(l3_stack)
                elif agg == 'sum':
                    loss_3 = torch.sum(l3_stack)
                else:
                    loss_3 = torch.mean(l3_stack)

            progress = (iteration_index - l3_start_iters) / max(1, (c.iters - l3_start_iters))
            w_l3 = float(progress) * l3_weight_max

        stego_float = torch.clamp(cover + adv_pert_full, 0.0, 1.0)
        loss_1 = l_hid(stego_float, cover)

        # Loss 2: clean reversibility loss
        if output_clean.shape[-2:] != secret.shape[-2:]:
            raise RuntimeError(
                f"Decoder output size {tuple(output_clean.shape[-2:])} does not match secret size {tuple(secret.shape[-2:])}. "
                f"Hint: check config.secret_image_size ({c.secret_image_size}) and which decodingNetwork is selected."
            )
        loss_2 = _masked_mse_loss(output_clean, secret, secret_mask_t)

        # Mix L2/L3 using beta like the paper: beta*L2 + (1-beta)*L3
        beta = float(getattr(c, 'beta', 0.5))
        loss_23 = beta * loss_2 + (1.0 - beta) * (w_l3 * loss_3)

        # TV regularization on perturbation
        tv_loss = (
            torch.mean(torch.abs(adv_pert_full[:, :, :, 1:] - adv_pert_full[:, :, :, :-1]))
            + torch.mean(torch.abs(adv_pert_full[:, :, 1:, :] - adv_pert_full[:, :, :-1, :]))
        )

        w_rev = float(getattr(c, 'w_rev', 1.0))
        w_tv = float(getattr(c, 'w_tv', 0.02))
        w_hid_base = float(getattr(c, 'w_hid_base', 0.05))
        w_hid_max = float(getattr(c, 'w_hid_max', 1.0))

        if iteration_index < warmup_iters:
            w_hid = w_hid_base
        else:
            progress = (iteration_index - warmup_iters) / max(1, (c.iters - warmup_iters))
            w_hid = w_hid_base + (w_hid_max - w_hid_base) * float(progress)

        loss = w_rev * loss_23 + w_hid * loss_1 + w_tv * tv_loss

        # Anti-steganalysis CE loss (enabled only for the last N iterations).
        if USE_STEGANALYSIS_NETS and (iteration_index >= steganalysis_start_iters):
            anti_loss = torch.zeros((), device=device)
            labels0 = torch.tensor([0], device=device)

            if SRNet is not None:
                anti_loss = anti_loss + l_anti_dec(SRNet(stego_float), labels0)

            if SiaStegNet is not None and preprocess_data is not None:
                inputs, labels = preprocess_data(stego_float * 255.0, labels0, False)
                outputs, *_feats = SiaStegNet(*inputs)
                anti_loss = anti_loss + l_anti_dec(outputs, labels)

            if Yenet is not None:
                anti_loss = anti_loss + l_anti_dec(Yenet(stego_float), labels0)

            loss = loss + (steganalysis_beta * steganalysis_gamma) * anti_loss
            
        loss.backward()
        optimizer.step()
        weight_scheduler.step()

    logger.info('-' * 60)

    adv_image = cover + adv_pert_full
    adv_image = torch.round(torch.clamp(adv_image * 255, min=0., max=255.)) / 255
    # adv_pert_full = add_contrast_adjustment(adv_image) - cover
    adv_pert_full = adv_image - cover
    
    adv_pert = reverse_crop_and_rearrange_no_loop(adv_pert_full, block_positions)
    secret_rev = model(adv_pert)

    if denoise_model is not None:
        with torch.no_grad():
            secret_rev = denoise_model(secret_rev)
    secret_rev = torch.round(torch.clamp(secret_rev * 255, min=0., max=255.)) / 255
    
    cover_resi = (adv_image - cover).abs() * c.resi_magnification
    secret_resi = (secret_rev - secret).abs() * c.resi_magnification

    secret_tensor = secret # Keep tensor for robustness tests
    cover = cover.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255
    stego = adv_image.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255
    secret = secret.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255
    secret_rev = secret_rev.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255
    cover_resi = cover_resi.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255
    secret_resi = secret_resi.clone().squeeze().permute(2, 1, 0).detach().cpu().numpy() * 255

    stego_psnr = calculate_psnr(cover, stego)
    stego_ssim = calculate_ssim(cover, stego)
    stego_lpips = calculate_lpips(cover, stego)
    stego_apd = calculate_mae(cover, stego)

    secret_rev_psnr = calculate_psnr(secret, secret_rev)
    secret_rev_ssim = calculate_ssim(secret, secret_rev)
    secret_rev_lpips = calculate_lpips(secret, secret_rev)
    secret_rev_apd = calculate_mae(secret, secret_rev)

    secret_rev_psnr_masked = None
    secret_rev_ssim_masked = None
    secret_rev_nc_masked = None
    if secret_mask_hw is not None:
        secret_rev_psnr_masked = _calculate_psnr_masked(secret, secret_rev, secret_mask_hw)
        secret_rev_ssim_masked = _calculate_ssim_masked(secret, secret_rev, secret_mask_hw)
        secret_rev_nc_masked = _calculate_nc_masked(secret, secret_rev, secret_mask_hw)
    
    logger.info('stego_psnr: {:.2f}, secret_rev_psnr: {:.2f}'.format(stego_psnr, secret_rev_psnr))
    logger.info('stego_ssim: {:.4f}, secret_rev_ssim: {:.4f}'.format(stego_ssim, secret_rev_ssim))
    logger.info('stego_lpips: {:.4f}, secret_rev_lpips: {:.4f}'.format(stego_lpips, secret_rev_lpips))
    logger.info('stego_apd: {:.2f}, secret_rev_apd: {:.2f}'.format(stego_apd, secret_rev_apd))
    if secret_rev_psnr_masked is not None:
        logger.info(
            'secret_rev_psnr_masked: {:.2f}, secret_rev_ssim_masked: {:.4f}, secret_rev_nc_masked: {:.6f}'.format(
                secret_rev_psnr_masked,
                secret_rev_ssim_masked,
                secret_rev_nc_masked,
            )
        )
    
    logger.info('--- Running robustness tests (Clamp, Conventional Attacks) ---')
    robustness_results = run_robustness_tests(
        adv_image,
        cover_backup,
        secret_tensor,
        model,
        denoise_model,
        block_positions,
        secret_mask_hw=secret_mask_hw,
        save_dir=image_save_dirs,
        img_idx=os.path.basename(secret_image_path_list[i])
    )
    for res in robustness_results:
        logger.info(f"{res['Attack']:<20} | Sec_NC: {res['NC']:.6f} | Sec_PSNR: {res['Sec_PSNR']:.2f} | Sec_SSIM: {res['Sec_SSIM']:.4f} | Sec_LPIPS: {res['Sec_LPIPS']:.4f} | Stego_PSNR: {res['Stego_PSNR']:.2f} | Stego_SSIM: {res['Stego_SSIM']:.4f} | Stego_LPIPS: {res['Stego_LPIPS']:.4f}")
        attack_name = str(res['Attack'])
        if attack_name not in robust_nc_stats:
            robust_nc_stats[attack_name] = []
        robust_nc_stats[attack_name].append(float(res['NC']))
    logger.info('--- End of robustness tests ---')

    stego_psnr_list.append(stego_psnr)
    secret_rev_psnr_list.append(secret_rev_psnr)
    stego_ssim_list.append(stego_ssim)
    secret_rev_ssim_list.append(secret_rev_ssim)
    stego_apd_list.append(stego_apd)
    secret_rev_apd_list.append(secret_rev_apd)

    if secret_rev_psnr_masked is not None:
        secret_rev_psnr_masked_list.append(secret_rev_psnr_masked)
        secret_rev_ssim_masked_list.append(secret_rev_ssim_masked)
        secret_rev_nc_masked_list.append(secret_rev_nc_masked)

    stego_lpips_list.append(stego_lpips)
    secret_rev_lpips_list.append(secret_rev_lpips)

    # --- Intermediate mean summary (default: at 1000th image) ---
    mid_summary_at = int(getattr(c, 'mid_summary_at', 1000))
    if mid_summary_at > 0 and (i + 1) == mid_summary_at:
        try:
            n_done = int(len(stego_psnr_list))
            _log_running_means(tag='Intermediate Mean @ {} images'.format(mid_summary_at), n_images=n_done)
        except Exception as e:
            logger.warning('Failed to log intermediate mean @ {}: {}'.format(mid_summary_at, str(e)))


    if c.save_images:
        cover_save_path = os.path.join(image_save_dirs, 'cover_jpegloss', cover_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        stego_save_path = os.path.join(image_save_dirs, 'stego_jpegloss', cover_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        secret_save_path = os.path.join(image_save_dirs, 'secret_jpegloss', secret_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        secret_rev_save_path = os.path.join(image_save_dirs, 'secret_rev_jpegloss', secret_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        cover_resi_save_path = os.path.join(image_save_dirs, 'cover_resi_jpegloss', cover_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        secret_resi_save_path = os.path.join(image_save_dirs, 'secret_resi_jpegloss', secret_image_path_list[i].split('/')[-1].split('.')[0]+'.png')
        mkdirs(os.path.join(image_save_dirs, 'cover_jpegloss'))
        mkdirs(os.path.join(image_save_dirs, 'stego_jpegloss'))
        mkdirs(os.path.join(image_save_dirs, 'secret_jpegloss'))
        mkdirs(os.path.join(image_save_dirs, 'secret_rev_jpegloss'))
        mkdirs(os.path.join(image_save_dirs, 'cover_resi_jpegloss'))
        mkdirs(os.path.join(image_save_dirs, 'secret_resi_jpegloss'))
        logger.info('saving images...')
        Image.fromarray(cover.astype(np.uint8)).save(cover_save_path)
        Image.fromarray(stego.astype(np.uint8)).save(stego_save_path)
        Image.fromarray(secret.astype(np.uint8)).save(secret_save_path)
        Image.fromarray(secret_rev.astype(np.uint8)).save(secret_rev_save_path)
        Image.fromarray(cover_resi.astype(np.uint8)).save(cover_resi_save_path)
        Image.fromarray(secret_resi.astype(np.uint8)).save(secret_resi_save_path)

    # Save resumable state after each finished image.
    try:
        _save_resume_state(i + 1)
    except Exception as e:
        logger.warning('Failed to save resume state after image {}: {}'.format(i, str(e)))


logger.info('stego_psnr_mean: {:.2f}, stego_ssim_mean: {:.4f}, stego_lpips_mean: {:.4f}, stego_apd_mean: {:.2f}'.format(np.array(stego_psnr_list).mean(), np.array(stego_ssim_list).mean(), np.array(stego_lpips_list).mean(), np.array(stego_apd_list).mean()))
logger.info('secret_rev_psnr_mean: {:.2f}, secret_rev_ssim_mean: {:.4f}, secret_rev_lpips_mean: {:.4f}, secret_rev_apd_mean: {:.2f}'.format(np.array(secret_rev_psnr_list).mean(), np.array(secret_rev_ssim_list).mean(), np.array(secret_rev_lpips_list).mean(), np.array(secret_rev_apd_list).mean()))
if len(secret_rev_psnr_masked_list) > 0:
    logger.info('secret_rev_psnr_masked_mean: {:.2f}'.format(np.array(secret_rev_psnr_masked_list).mean()))
    logger.info('secret_rev_ssim_masked_mean: {:.4f}'.format(np.array(secret_rev_ssim_masked_list).mean()))
    logger.info('secret_rev_lpips_masked_mean: {:.4f}'.format(np.array(secret_rev_lpips_masked_list).mean()))
    logger.info('secret_rev_nc_masked_mean: {:.6f}'.format(np.array(secret_rev_nc_masked_list).mean()))

if len(robust_nc_stats) > 0:
    logger.info('--- Robustness Test NC Mean (Table 2 Mean) ---')

    jpeg_keys = ['JPEG 80%']
    gauss_keys = ['Gaussian 0.07']
    sp_keys = ['Salt&Pepper 0.01']
    contrast_keys = ['Contrast 0.9']

    def _mean_or_nan(k):
        vals = robust_nc_stats.get(k, [])
        if len(vals) == 0:
            return float('nan')
        return float(np.mean(vals))

    jpeg_means = [_mean_or_nan(k) for k in jpeg_keys]
    gauss_means = [_mean_or_nan(k) for k in gauss_keys]
    sp_means = [_mean_or_nan(k) for k in sp_keys]
    contrast_means = [_mean_or_nan(k) for k in contrast_keys]

    logger.info('Mean ({} images) | JPEG: q80={:.6f}'.format(
        len(secret_image_path_list), jpeg_means[0]
    ))
    logger.info('Mean ({} images) | Gaussian: std0.07={:.6f}'.format(
        len(secret_image_path_list), gauss_means[0]
    ))
    logger.info('Mean ({} images) | Contrast: factor0.9={:.6f}'.format(
        len(secret_image_path_list), contrast_means[0]
    ))
    logger.info('Mean ({} images) | Salt&Pepper: density0.01={:.6f}'.format(
        len(secret_image_path_list), sp_means[0]
    ))

if resume_enabled and (not keep_resume_state_on_finish) and os.path.isfile(resume_state_path):
    try:
        os.remove(resume_state_path)
        logger.info('Training finished. Removed resume state: {}'.format(resume_state_path))
    except Exception as e:
        logger.warning('Failed to remove resume state {}: {}'.format(resume_state_path, str(e)))
