from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class RTNConfig:
    bits: int
    group_size: int = 128
    symmetric: bool = False
    per_group: bool = True
    quantized_modules: str = "torch.nn.Linear.weight"
    excluded_module_name_keywords: tuple[str, ...] = ()


@dataclass
class RTNQuantizedTensorState:
    original_dtype: torch.dtype
    in_features: int
    padded_in_features: int
    max_int: int
    pre_round: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor

    def dequantize_truncated(self) -> torch.Tensor:
        q = torch.round(self.pre_round).clamp(0, self.max_int)
        dequant = (q - self.zero_point) * self.scale
        if self.padded_in_features > self.in_features:
            dequant = dequant[:, : self.in_features]
        return dequant.to(self.original_dtype)


def _pad_to_group_size(w: torch.Tensor, group_size: int) -> tuple[torch.Tensor, int]:
    in_features = w.shape[1]
    remainder = in_features % group_size
    if remainder == 0:
        return w, in_features
    padded_in_features = in_features + group_size - remainder
    return F.pad(w, (0, padded_in_features - in_features)), padded_in_features


def rtn_quantize_weight_raw(w: torch.Tensor, bits: int = 4, group_size: int = 128) -> RTNQuantizedTensorState:
    """Return the raw RTN affine state expected by far-round code.

    Quantization is per output row and per contiguous input group. Partial final
    groups compute min/max from real weights only, then extend their scale and
    zero-point over the padded positions so masks can discard padding later.
    """
    if w.ndim != 2:
        raise ValueError(f"RTN expects a 2D weight tensor, got shape {tuple(w.shape)}")
    if bits < 1:
        raise ValueError("bits must be positive")
    if group_size < 1:
        raise ValueError("group_size must be positive")

    original_dtype = w.dtype
    work = w.detach().float()
    padded, padded_in_features = _pad_to_group_size(work, group_size)
    out_features, in_features = work.shape
    max_int = (1 << bits) - 1

    scale = torch.empty_like(padded)
    zero_point = torch.empty_like(padded)
    for start in range(0, padded_in_features, group_size):
        real_end = min(start + group_size, in_features)
        group = work[:, start:real_end]
        w_min = group.min(dim=1, keepdim=True).values
        w_max = group.max(dim=1, keepdim=True).values
        group_scale = ((w_max - w_min) / max_int).clamp(min=1e-8)
        group_zero_point = torch.round(-w_min / group_scale).clamp(0, max_int)
        scale[:, start : start + group_size] = group_scale
        zero_point[:, start : start + group_size] = group_zero_point

    pre_round = padded / scale + zero_point
    return RTNQuantizedTensorState(
        original_dtype=original_dtype,
        in_features=in_features,
        padded_in_features=padded_in_features,
        max_int=max_int,
        pre_round=pre_round,
        scale=scale,
        zero_point=zero_point,
    )


def save_rtn_config(config: RTNConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["excluded_module_name_keywords"] = list(config.excluded_module_name_keywords)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rtn_quantize_tensor(weight: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    if not torch.is_floating_point(weight):
        return weight
    state = rtn_quantize_weight_raw(weight, bits=bits, group_size=group_size)
    return state.dequantize_truncated()

def get_transformer_linear_layers(model) -> "dict[str, Any]":
    """name -> nn.Linear for every projection inside model.model.layers[*]
    (q/k/v/o + gate/up/down), in depth order. lm_head/embeddings excluded -
    the plan's adversarial objective only concerns the transformer body.
    """
    import torch.nn as nn

    layers: dict[str, Any] = {}
    for block_idx, block in enumerate(model.model.layers):
        for local_name, module in block.named_modules():
            if isinstance(module, nn.Linear):
                layers[f"model.layers.{block_idx}.{local_name}"] = module
    return layers


def _apply_rtn_baseline(layers: dict, bits: int, group_size: int = 128) -> None:
    import torch

    for module in layers.values():
        w = module.weight.detach().clone()
        state = rtn_quantize_weight_raw(w, bits=bits, group_size=group_size)
        with torch.no_grad():
            module.weight.data.copy_(state.dequantize_truncated().to(module.weight.dtype))

def _should_quantize(name: str, module: nn.Module, include: Callable[[str], bool] | None, excludes: Iterable[str]) -> bool:
    if not isinstance(module, nn.Linear):
        return False
    if any(keyword and keyword in name for keyword in excludes):
        return False
    return True if include is None else include(name)


@torch.no_grad()
def quantize_model_linear_weights(
    model: nn.Module,
    config: RTNConfig,
    include: Callable[[str], bool] | None = None,
) -> list[str]:
    layers = {
        name: module
        for name, module in get_transformer_linear_layers(model).items()
        if _should_quantize(name, module, include, config.excluded_module_name_keywords)
    }
    _apply_rtn_baseline(layers, bits=config.bits, group_size=config.group_size)
    return [f"{name}.weight" for name in layers]


def transformer_block_include(block_id: int) -> Callable[[str], bool]:
    needles = (
        f".layers.{block_id}.",
        f".h.{block_id}.",
        f".blocks.{block_id}.",
        f".block.{block_id}.",
    )
    return lambda module_name: any(needle in f".{module_name}." for needle in needles)
