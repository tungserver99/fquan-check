from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import nn


@dataclass(frozen=True)
class RTNConfig:
    bits: int
    group_size: int = 128
    symmetric: bool = True
    per_group: bool = True
    quantized_modules: str = "torch.nn.Linear.weight"
    excluded_module_name_keywords: tuple[str, ...] = ()


def save_rtn_config(config: RTNConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["excluded_module_name_keywords"] = list(config.excluded_module_name_keywords)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rtn_quantize_tensor(weight: torch.Tensor, bits: int, group_size: int = 128) -> torch.Tensor:
    if bits < 2:
        raise ValueError("RTN quantization requires bits >= 2")
    if group_size <= 0:
        raise ValueError("group_size must be positive")
    if not torch.is_floating_point(weight):
        return weight

    original_shape = weight.shape
    original_dtype = weight.dtype
    flat = weight.detach().to(torch.float32).reshape(-1)
    pad = (group_size - flat.numel() % group_size) % group_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    grouped = flat.reshape(-1, group_size)

    qmax = (1 << (bits - 1)) - 1
    scales = grouped.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
    q = torch.round(grouped / scales).clamp(-qmax, qmax)
    dequant = (q * scales).reshape(-1)
    if pad:
        dequant = dequant[:-pad]
    return dequant.reshape(original_shape).to(original_dtype)


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
    touched: list[str] = []
    for name, module in model.named_modules():
        if not _should_quantize(name, module, include, config.excluded_module_name_keywords):
            continue
        module.weight.data.copy_(rtn_quantize_tensor(module.weight.data, config.bits, config.group_size))
        touched.append(f"{name}.weight")
    return touched


def transformer_block_include(block_id: int) -> Callable[[str], bool]:
    needles = (
        f".layers.{block_id}.",
        f".h.{block_id}.",
        f".blocks.{block_id}.",
        f".block.{block_id}.",
    )
    return lambda module_name: any(needle in f".{module_name}." for needle in needles)


