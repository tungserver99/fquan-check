from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch

from .rtn4_quant_state import RTN4WeightState, assert_matching_rtn4_states

EPS = 1e-12
MODULE_ORDER = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _coordinate_row(base: RTN4WeightState, if_state: RTN4WeightState, output_row: int, input_col: int) -> dict[str, Any]:
    group_id = input_col // base.group_size
    offset = input_col % base.group_size
    bq = int(base.qcode[output_row, input_col].item())
    fq = int(if_state.qcode[output_row, input_col].item())
    bs = base.group_scale(output_row, group_id)
    fs = if_state.group_scale(output_row, group_id)
    bz = base.group_zero_point(output_row, group_id)
    fz = if_state.group_zero_point(output_row, group_id)
    bw = float(base.dequant[output_row, input_col].item())
    fw = float(if_state.dequant[output_row, input_col].item())
    return {
        "tensor_name": base.tensor_name,
        "block_id": base.block_id,
        "module_type": base.module_type,
        "output_row": output_row,
        "input_col": input_col,
        "group_id": group_id,
        "offset_in_group": offset,
        "base_qcode": bq,
        "if_qcode": fq,
        "qcode_diff": fq - bq,
        "abs_qcode_diff": abs(fq - bq),
        "same_qcode": bq == fq,
        "base_scale": bs,
        "if_scale": fs,
        "scale_diff": fs - bs,
        "relative_scale_diff": abs(fs - bs) / (abs(bs) if abs(bs) > 0 else EPS),
        "base_zero_point": bz,
        "if_zero_point": fz,
        "zero_point_diff": fz - bz,
        "base_dequant_weight": bw,
        "if_dequant_weight": fw,
        "dequant_diff": fw - bw,
        "abs_dequant_diff": abs(fw - bw),
    }


def _expanded_group_values(state: RTN4WeightState, values: torch.Tensor, output_row: int) -> list[float]:
    row = values[output_row].detach().cpu().float()
    in_features = state.shape[1]
    if row.numel() == state.num_groups:
        row = torch.repeat_interleave(row, state.group_size)[:in_features]
    else:
        row = row[:in_features]
    return [float(value) for value in row.tolist()]


def coordinate_row_table(base: RTN4WeightState, if_state: RTN4WeightState, output_row: int, pa: Any):
    if base.shape != if_state.shape:
        raise ValueError(f"Shape mismatch for {base.tensor_name}: {base.shape} vs {if_state.shape}")
    in_features = base.shape[1]
    input_cols = list(range(in_features))
    group_ids = [input_col // base.group_size for input_col in input_cols]
    offsets = [input_col % base.group_size for input_col in input_cols]
    bq = base.qcode[output_row].detach().cpu().to(torch.int64)
    fq = if_state.qcode[output_row].detach().cpu().to(torch.int64)
    qdiff = fq - bq
    bs = _expanded_group_values(base, base.scale, output_row)
    fs = _expanded_group_values(if_state, if_state.scale, output_row)
    bz = _expanded_group_values(base, base.zero_point, output_row)
    fz = _expanded_group_values(if_state, if_state.zero_point, output_row)
    bw = [float(value) for value in base.dequant[output_row].detach().cpu().float().tolist()]
    fw = [float(value) for value in if_state.dequant[output_row].detach().cpu().float().tolist()]
    scale_diff = [if_scale - base_scale for base_scale, if_scale in zip(bs, fs)]
    zero_point_diff = [if_zp - base_zp for base_zp, if_zp in zip(bz, fz)]
    dequant_diff = [if_weight - base_weight for base_weight, if_weight in zip(bw, fw)]
    return pa.Table.from_pydict({
        "tensor_name": [base.tensor_name] * in_features,
        "block_id": [base.block_id] * in_features,
        "module_type": [base.module_type] * in_features,
        "output_row": [output_row] * in_features,
        "input_col": input_cols,
        "group_id": group_ids,
        "offset_in_group": offsets,
        "base_qcode": [int(value) for value in bq.tolist()],
        "if_qcode": [int(value) for value in fq.tolist()],
        "qcode_diff": [int(value) for value in qdiff.tolist()],
        "abs_qcode_diff": [abs(int(value)) for value in qdiff.tolist()],
        "same_qcode": [bool(value) for value in (bq == fq).tolist()],
        "base_scale": bs,
        "if_scale": fs,
        "scale_diff": scale_diff,
        "relative_scale_diff": [abs(diff) / (abs(base_scale) if abs(base_scale) > 0 else EPS) for diff, base_scale in zip(scale_diff, bs)],
        "base_zero_point": bz,
        "if_zero_point": fz,
        "zero_point_diff": zero_point_diff,
        "base_dequant_weight": bw,
        "if_dequant_weight": fw,
        "dequant_diff": dequant_diff,
        "abs_dequant_diff": [abs(diff) for diff in dequant_diff],
    })


def iter_coordinate_rows(base: RTN4WeightState, if_state: RTN4WeightState) -> Iterable[dict[str, Any]]:
    for chunk in iter_coordinate_row_chunks(base, if_state):
        yield from chunk


def iter_coordinate_row_chunks(base: RTN4WeightState, if_state: RTN4WeightState) -> Iterable[list[dict[str, Any]]]:
    if base.shape != if_state.shape:
        raise ValueError(f"Shape mismatch for {base.tensor_name}: {base.shape} vs {if_state.shape}")
    out_features, in_features = base.shape
    for output_row in range(out_features):
        yield [_coordinate_row(base, if_state, output_row, input_col) for input_col in range(in_features)]


def _diff_stats(diff: torch.Tensor) -> dict[str, float]:
    diff = diff.detach().float().reshape(-1)
    abs_diff = diff.abs()
    return {
        "dequant_diff_l1": float(abs_diff.sum().item()),
        "dequant_diff_l2": float(torch.linalg.vector_norm(diff).item()),
        "dequant_diff_max": float(abs_diff.max().item()) if diff.numel() else 0.0,
        "dequant_diff_mean_abs": float(abs_diff.mean().item()) if diff.numel() else 0.0,
    }


def iter_group_rows(base: RTN4WeightState, if_state: RTN4WeightState) -> Iterable[dict[str, Any]]:
    if base.shape != if_state.shape:
        raise ValueError(f"Shape mismatch for {base.tensor_name}: {base.shape} vs {if_state.shape}")
    out_features, _ = base.shape
    for output_row in range(out_features):
        for group_id in range(base.num_groups):
            sl = base.group_slice(group_id)
            bq = base.qcode[output_row, sl]
            fq = if_state.qcode[output_row, sl]
            diff = if_state.dequant[output_row, sl] - base.dequant[output_row, sl]
            bs = base.group_scale(output_row, group_id)
            fs = if_state.group_scale(output_row, group_id)
            bz = base.group_zero_point(output_row, group_id)
            fz = if_state.group_zero_point(output_row, group_id)
            num_weights = int(bq.numel())
            num_qdiff = int((bq != fq).sum().item())
            yield {
                "tensor_name": base.tensor_name,
                "block_id": base.block_id,
                "module_type": base.module_type,
                "output_row": output_row,
                "group_id": group_id,
                "num_weights_in_group": num_weights,
                "num_weights": num_weights,
                "base_scale": bs,
                "if_scale": fs,
                "scale_different": bs != fs,
                "scale_diff": fs - bs,
                "relative_scale_diff": abs(fs - bs) / (abs(bs) if abs(bs) > 0 else EPS),
                "base_zero_point": bz,
                "if_zero_point": fz,
                "zero_point_different": bz != fz,
                "zero_point_diff": fz - bz,
                "num_qcode_different": num_qdiff,
                "qcode_diff_ratio": num_qdiff / num_weights if num_weights else 0.0,
                **_diff_stats(diff),
            }


def write_csv(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_parquet(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import pandas as pd
        pd.DataFrame(rows).to_parquet(path, index=False)
    except ImportError:
        sidecar = path.with_suffix(path.suffix + ".jsonl")
        sidecar.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def iter_table_row_batches(path: str | Path, columns: list[str] | None = None, batch_size: int = 65536) -> Iterable[list[dict[str, Any]]]:
    path = Path(path)
    sidecar = path.with_suffix(path.suffix + ".jsonl")
    if path.exists():
        try:
            import pyarrow.parquet as pq
            parquet = pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=batch_size, columns=columns):
                rows = batch.to_pylist()
                if rows:
                    yield rows
            return
        except ImportError:
            pass
    if sidecar.exists():
        rows: list[dict[str, Any]] = []
        wanted = set(columns) if columns is not None else None
        with sidecar.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                if wanted is not None:
                    row = {key: row[key] for key in wanted if key in row}
                rows.append(row)
                if len(rows) >= batch_size:
                    yield rows
                    rows = []
        if rows:
            yield rows


def read_table_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for batch in iter_table_row_batches(path):
        rows.extend(batch)
    return rows


def summarize_state_pair(base: RTN4WeightState, if_state: RTN4WeightState) -> dict[str, Any]:
    qdiff = base.qcode != if_state.qcode
    diff = if_state.dequant - base.dequant
    group_rows = list(iter_group_rows(base, if_state))
    scale_diff = [row for row in group_rows if row.get("scale_different")]
    zp_diff = [row for row in group_rows if row.get("zero_point_different")]
    return {
        "tensor_name": base.tensor_name,
        "block_id": base.block_id,
        "module_type": base.module_type,
        "num_weights": int(base.dequant.numel()),
        "num_qcode_different": int(qdiff.sum().item()),
        "qcode_diff_ratio": float(qdiff.float().mean().item()) if qdiff.numel() else 0.0,
        **_diff_stats(diff),
        "num_groups": len(group_rows),
        "num_groups_scale_different": len(scale_diff),
        "fraction_groups_scale_different": len(scale_diff) / len(group_rows) if group_rows else 0.0,
        "num_groups_zp_different": len(zp_diff),
        "fraction_groups_zp_different": len(zp_diff) / len(group_rows) if group_rows else 0.0,
        "mean_relative_scale_diff": sum(float(row["relative_scale_diff"]) for row in group_rows) / len(group_rows) if group_rows else 0.0,
    }


def aggregate_numeric(rows: Iterable[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        item = groups.setdefault(key, {k: row[k] for k in keys} | {
            "num_weights": 0,
            "num_qcode_different": 0,
            "dequant_diff_l1": 0.0,
            "dequant_diff_l2_sq": 0.0,
            "dequant_diff_max": 0.0,
        })
        n = int(row["num_weights"])
        item["num_weights"] += n
        item["num_qcode_different"] += int(row["num_qcode_different"])
        item["dequant_diff_l1"] += float(row["dequant_diff_l1"])
        item["dequant_diff_l2_sq"] += float(row["dequant_diff_l2"]) ** 2
        item["dequant_diff_max"] = max(float(item["dequant_diff_max"]), float(row["dequant_diff_max"]))
    out = []
    for item in groups.values():
        n = int(item["num_weights"])
        item["qcode_diff_ratio"] = item["num_qcode_different"] / n if n else 0.0
        item["dequant_diff_l2"] = math.sqrt(float(item.pop("dequant_diff_l2_sq")))
        item["dequant_diff_mean_abs"] = item["dequant_diff_l1"] / n if n else 0.0
        out.append(item)
    return sorted(out, key=lambda row: tuple(row[k] for k in keys))


def row_summaries(base_states: dict[str, RTN4WeightState], if_states: dict[str, RTN4WeightState]) -> list[dict[str, Any]]:
    assert_matching_rtn4_states(base_states, if_states)
    rows = []
    for name in sorted(base_states):
        base = base_states[name]
        if_state = if_states[name]
        for output_row in range(base.shape[0]):
            qdiff = base.qcode[output_row] != if_state.qcode[output_row]
            diff = if_state.dequant[output_row] - base.dequant[output_row]
            rows.append({
                "tensor_name": name,
                "block_id": base.block_id,
                "module_type": base.module_type,
                "output_row": output_row,
                "num_weights": int(diff.numel()),
                "num_qcode_different": int(qdiff.sum().item()),
                "qcode_diff_ratio": float(qdiff.float().mean().item()) if qdiff.numel() else 0.0,
                **_diff_stats(diff),
                "num_groups": base.num_groups,
            })
    return rows


def build_all_summaries(base_states: dict[str, RTN4WeightState], if_states: dict[str, RTN4WeightState]) -> dict[str, list[dict[str, Any]]]:
    assert_matching_rtn4_states(base_states, if_states)
    tensor_rows = [summarize_state_pair(base_states[name], if_states[name]) for name in sorted(base_states)]
    group_rows = []
    for name in sorted(base_states):
        group_rows.extend(iter_group_rows(base_states[name], if_states[name]))
    return {
        "tensor": tensor_rows,
        "block": aggregate_numeric(tensor_rows, ["block_id"]),
        "module": aggregate_numeric(tensor_rows, ["module_type"]),
        "block_module": aggregate_numeric(tensor_rows, ["block_id", "module_type"]),
        "row": row_summaries(base_states, if_states),
        "group": group_rows,
    }


def write_required_plots(results_dir: str | Path, plots_dir: str | Path) -> None:
    import pandas as pd
    import matplotlib.pyplot as plt

    results_dir = Path(results_dir)
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)

    block = pd.read_csv(results_dir / "rtn4_diff_by_block.csv")
    plt.figure(figsize=(8, 3))
    plt.bar(block["block_id"], block["qcode_diff_ratio"])
    plt.xlabel("block")
    plt.ylabel("qcode diff ratio")
    plt.tight_layout()
    plt.savefig(plots_dir / "rtn4_qcode_diff_ratio_by_block.png")
    plt.close()

    bm = pd.read_csv(results_dir / "rtn4_diff_by_block_module.csv")
    for metric, filename in [
        ("qcode_diff_ratio", "rtn4_block_module_qcode_diff_heatmap.png"),
        ("dequant_diff_l2", "rtn4_block_module_l2_diff_heatmap.png"),
    ]:
        table = bm.pivot(index="module_type", columns="block_id", values=metric).reindex(MODULE_ORDER)
        plt.figure(figsize=(10, 4))
        plt.imshow(table.fillna(0.0), aspect="auto")
        plt.yticks(range(len(table.index)), table.index)
        plt.xticks(range(len(table.columns)), table.columns, rotation=90)
        plt.colorbar(label=metric)
        plt.tight_layout()
        plt.savefig(plots_dir / filename)
        plt.close()

    bin_count = 50
    counts = [0] * bin_count
    for batch in iter_table_row_batches(results_dir / "rtn4_exact_diff_groups.parquet", columns=["qcode_diff_ratio"]):
        for row in batch:
            value = max(0.0, min(1.0, float(row["qcode_diff_ratio"])))
            index = min(bin_count - 1, int(value * bin_count))
            counts[index] += 1
    edges = [idx / bin_count for idx in range(bin_count + 1)]
    plt.figure(figsize=(6, 4))
    plt.bar(edges[:-1], counts, width=1 / bin_count, align="edge")
    plt.xlabel("group qcode diff ratio")
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(plots_dir / "rtn4_group_qcode_diff_distribution.png")
    plt.close()




