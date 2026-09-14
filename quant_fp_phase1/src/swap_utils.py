from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .quantization import get_transformer_linear_layers
from .rtn4_quant_state import module_type_from_name


def swap_rtn4_group_weight(dst_module: nn.Linear, src_module: nn.Linear, output_row: int, group_id: int, group_size: int = 128) -> None:
    start = group_id * group_size
    end = min(start + group_size, dst_module.weight.shape[1])
    if start >= end:
        raise ValueError(f"Invalid group_id={group_id} for weight with {dst_module.weight.shape[1]} input columns")
    with torch.no_grad():
        dst_module.weight.data[output_row, start:end].copy_(src_module.weight.data[output_row, start:end])


def _matching_layers(dst_model: Any, src_model: Any) -> tuple[dict[str, nn.Linear], dict[str, nn.Linear]]:
    dst_layers = get_transformer_linear_layers(dst_model)
    src_layers = get_transformer_linear_layers(src_model)
    if set(dst_layers) != set(src_layers):
        raise ValueError("Cannot swap: transformer linear layer names differ")
    return dst_layers, src_layers


def swap_rtn4_block(dst_model: Any, src_model: Any, block_id: int) -> list[str]:
    dst_layers, src_layers = _matching_layers(dst_model, src_model)
    touched = []
    needle = f"model.layers.{block_id}."
    for name in sorted(dst_layers):
        if needle not in name:
            continue
        if dst_layers[name].weight.shape != src_layers[name].weight.shape:
            raise ValueError(f"Cannot swap {name}: shape mismatch")
        with torch.no_grad():
            dst_layers[name].weight.data.copy_(src_layers[name].weight.data.to(dst_layers[name].weight.device))
        touched.append(f"{name}.weight")
    return touched


def swap_rtn4_module(dst_model: Any, src_model: Any, block_id: int, module_type: str) -> list[str]:
    dst_layers, src_layers = _matching_layers(dst_model, src_model)
    touched = []
    needle = f"model.layers.{block_id}."
    for name in sorted(dst_layers):
        if needle not in name or module_type_from_name(name) != module_type:
            continue
        if dst_layers[name].weight.shape != src_layers[name].weight.shape:
            raise ValueError(f"Cannot swap {name}: shape mismatch")
        with torch.no_grad():
            dst_layers[name].weight.data.copy_(src_layers[name].weight.data.to(dst_layers[name].weight.device))
        touched.append(f"{name}.weight")
    return touched


def swap_rtn4_group(dst_model: Any, src_model: Any, tensor_name: str, output_row: int, group_id: int, group_size: int = 128) -> None:
    layer_name = tensor_name[:-7] if tensor_name.endswith(".weight") else tensor_name
    dst_layers, src_layers = _matching_layers(dst_model, src_model)
    if layer_name not in dst_layers:
        raise KeyError(f"Unknown transformer linear layer: {layer_name}")
    swap_rtn4_group_weight(dst_layers[layer_name], src_layers[layer_name], output_row, group_id, group_size)
