from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn as nn


def _normalize_choice(value: Any) -> str:
    if value is None:
        return ''
    return str(value).strip().lower()


def build_secret_postprocessor(c: Any, device: torch.device, default: str = "none") -> Optional[nn.Module]:
    """Build a postprocessor for decoded secret images.

    Supported values:
      - secret_postprocess: 'none' | 'dncnn'
    Backward compatible:
      - use_denoise_for_secret (bool): if True -> dncnn
    """

    choice = getattr(c, "secret_postprocess", None)
    if choice is None:
        if hasattr(c, "use_denoise_for_secret"):
            choice = "dncnn" if bool(getattr(c, "use_denoise_for_secret", False)) else "none"
        else:
            choice = default

    choice = _normalize_choice(choice)
    if choice in ("", "none", "off", "false", "0"):
        return None

    if choice in ("dncnn", "dcnn"):
        from models.network_dncnn import DnCNN

        ckpt_path = str(getattr(c, "dncnn_ckpt", "models/dncnn_color_blind.pth"))
        model = DnCNN(in_nc=3, out_nc=3, nc=64, nb=20, act_mode="R").to(device)
        state = torch.load(ckpt_path, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model

    raise ValueError(f"Unknown secret_postprocess: {choice}. Use 'none'|'dncnn'.")


@torch.no_grad()
def apply_secret_postprocess(model: Optional[nn.Module], x: torch.Tensor) -> torch.Tensor:
    if model is None:
        return x
    return model(x)
