from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

import torch

from .quantization import RTNConfig, get_transformer_linear_layers, rtn_quantize_weight_raw


MODULE_TYPES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


@dataclass
class RTN4WeightState:
    tensor_name: str
    block_id: int
    module_type: str
    qcode: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    dequant: torch.Tensor
    group_size: int = 128

    @property
    def shape(self) -> tuple[int, int]:
        return tuple(self.dequant.shape)  # type: ignore[return-value]

    @property
    def num_groups(self) -> int:
        return (self.dequant.shape[1] + self.group_size - 1) // self.group_size

    def group_slice(self, group_id: int) -> slice:
        start = group_id * self.group_size
        return slice(start, min(start + self.group_size, self.dequant.shape[1]))

    def group_scale(self, output_row: int, group_id: int) -> float:
        if self.scale.shape[1] == self.num_groups:
            return float(self.scale[output_row, group_id].item())
        return float(self.scale[output_row, group_id * self.group_size].item())

    def group_zero_point(self, output_row: int, group_id: int) -> float:
        if self.zero_point.shape[1] == self.num_groups:
            return float(self.zero_point[output_row, group_id].item())
        return float(self.zero_point[output_row, group_id * self.group_size].item())


def block_id_from_name(name: str) -> int:
    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", name)
    if match:
        return int(match.group(1))
    match = re.search(r"(?:^|\.)h\.(\d+)(?:\.|$)", name)
    if match:
        return int(match.group(1))
    return -1


def module_type_from_name(name: str) -> str:
    for module_type in MODULE_TYPES:
        if module_type in name:
            return module_type
    return "unknown"


def _group_params(expanded: torch.Tensor, in_features: int, group_size: int) -> torch.Tensor:
    values = []
    for start in range(0, in_features, group_size):
        values.append(expanded[:, start].detach().cpu())
    return torch.stack(values, dim=1) if values else torch.empty((expanded.shape[0], 0), dtype=expanded.dtype)


def quantize_rtn4_with_state(
    model: Any,
    group_size: int = 128,
    include: Callable[[str], bool] | None = None,
) -> dict[str, RTN4WeightState]:
    config = RTNConfig(bits=4, group_size=group_size)
    states: dict[str, RTN4WeightState] = {}
    for module_name, module in get_transformer_linear_layers(model).items():
        if include is not None and not include(module_name):
            continue
        weight = module.weight.detach().clone()
        raw = rtn_quantize_weight_raw(weight, bits=config.bits, group_size=config.group_size)
        qcode = torch.round(raw.pre_round).clamp(0, raw.max_int).to(torch.int16)[:, : raw.in_features].cpu()
        dequant = raw.dequantize_truncated().to(module.weight.dtype)
        with torch.no_grad():
            module.weight.data.copy_(dequant.to(module.weight.device))
        tensor_name = f"{module_name}.weight"
        states[tensor_name] = RTN4WeightState(
            tensor_name=tensor_name,
            block_id=block_id_from_name(module_name),
            module_type=module_type_from_name(module_name),
            qcode=qcode,
            scale=_group_params(raw.scale, raw.in_features, group_size).cpu(),
            zero_point=_group_params(raw.zero_point, raw.in_features, group_size).cpu(),
            dequant=dequant.detach().cpu(),
            group_size=group_size,
        )
    return states


def assert_matching_rtn4_states(base: dict[str, RTN4WeightState], other: dict[str, RTN4WeightState]) -> None:
    base_names = set(base)
    other_names = set(other)
    if base_names != other_names:
        missing = sorted(base_names - other_names)
        extra = sorted(other_names - base_names)
        raise ValueError(f"RTN4 tensor mismatch: missing={missing[:5]}, extra={extra[:5]}")
    for name in sorted(base):
        if base[name].shape != other[name].shape:
            raise ValueError(f"RTN4 shape mismatch for {name}: {base[name].shape} vs {other[name].shape}")
        if base[name].block_id != other[name].block_id or base[name].module_type != other[name].module_type:
            raise ValueError(
                f"RTN4 metadata mismatch for {name}: "
                f"block/module {base[name].block_id}/{base[name].module_type} vs "
                f"{other[name].block_id}/{other[name].module_type}"
            )


