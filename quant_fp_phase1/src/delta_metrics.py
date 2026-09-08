from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Iterable

import torch


def classify_module(name: str) -> str:
    lowered = name.lower()
    if "embed" in lowered:
        return "embedding"
    if "lm_head" in lowered:
        return "lm_head"
    if any(part in lowered for part in ("q_proj", "k_proj", "v_proj", "o_proj", "attn", "attention")):
        return "attention"
    if any(part in lowered for part in ("gate_proj", "up_proj", "down_proj", "mlp", "feed_forward")):
        return "mlp"
    if "norm" in lowered or "ln_" in lowered:
        return "layer_norm"
    return "other"


def infer_block_id(name: str) -> int | None:
    for pattern in (r"layers\.(\d+)", r"h\.(\d+)", r"blocks\.(\d+)", r"block\.(\d+)"):
        match = re.search(pattern, name)
        if match:
            return int(match.group(1))
    return None


def tensor_delta_stats(
    name: str,
    base: torch.Tensor,
    if_weight: torch.Tensor,
    q_base: torch.Tensor | None = None,
    q_if: torch.Tensor | None = None,
    eps: float = 1e-12,
) -> dict[str, float | int | str | None]:
    base_f = base.detach().to(torch.float32)
    if_f = if_weight.detach().to(torch.float32)
    delta = if_f - base_f
    delta_l2 = torch.linalg.vector_norm(delta).item()
    base_l2 = torch.linalg.vector_norm(base_f).item()
    row: dict[str, float | int | str | None] = {
        "layer": name,
        "block_id": infer_block_id(name),
        "module": classify_module(name),
        "fp_delta_l2": delta_l2,
        "fp_relative_delta": delta_l2 / (base_l2 + eps),
        "fp_delta_mean_abs": delta.abs().mean().item(),
        "fp_delta_max_abs": delta.abs().max().item(),
        "fp_delta_std": delta.std(unbiased=False).item(),
    }
    if q_base is not None and q_if is not None:
        q_delta = q_if.detach().to(torch.float32) - q_base.detach().to(torch.float32)
        q_l2 = torch.linalg.vector_norm(q_delta).item()
        denom = (torch.linalg.vector_norm(delta).item() * torch.linalg.vector_norm(q_delta).item()) + eps
        cosine = torch.sum(delta.reshape(-1) * q_delta.reshape(-1)).item() / denom
        nonzero = delta != 0
        collapse = ((q_delta == 0) & nonzero).sum().item() / max(nonzero.sum().item(), 1)
        row.update(
            {
                "quantized_delta_l2": q_l2,
                "delta_norm_survival": q_l2 / (delta_l2 + eps),
                "delta_cosine_similarity": max(min(cosine, 1.0), -1.0),
                "coordinate_collapse_rate": collapse,
            }
        )
    return row


def aggregate_by_block(rows: Iterable[dict[str, float | int | str | None]]) -> list[dict[str, float | int]]:
    grouped: dict[int, list[dict[str, float | int | str | None]]] = defaultdict(list)
    for row in rows:
        block_id = row.get("block_id")
        if block_id is not None:
            grouped[int(block_id)].append(row)
    out = []
    for block_id, block_rows in sorted(grouped.items()):
        fp_l2 = sum(float(row["fp_delta_l2"]) for row in block_rows)
        item: dict[str, float | int] = {
            "block_id": block_id,
            "fp_delta_l2_sum": fp_l2,
            "fp_relative_delta_mean": sum(float(row["fp_relative_delta"]) for row in block_rows) / len(block_rows),
        }
        if "delta_norm_survival" in block_rows[0]:
            item["delta_norm_survival_mean"] = sum(float(row["delta_norm_survival"]) for row in block_rows) / len(block_rows)
            vals = [float(row["delta_cosine_similarity"]) for row in block_rows if not math.isnan(float(row["delta_cosine_similarity"]))]
            item["delta_cosine_similarity_mean"] = sum(vals) / len(vals) if vals else float("nan")
        out.append(item)
    return out

