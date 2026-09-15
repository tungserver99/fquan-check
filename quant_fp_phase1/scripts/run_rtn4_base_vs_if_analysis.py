from __future__ import annotations

import argparse
import csv
import gc
import heapq
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm

from quant_fp_phase1.scripts.run_base_fp_outputs import iter_base_fingerprint_outputs
from quant_fp_phase1.src.experiments import load_fingerprint_examples, select_positive_fingerprint_examples
from quant_fp_phase1.src.quantization import RTNConfig, get_transformer_linear_layers
from quant_fp_phase1.src.rtn4_difference import (
    aggregate_numeric,
    coordinate_row_table,
    iter_coordinate_row_chunks,
    iter_group_rows,
    iter_table_row_batches,
    read_table_rows,
    row_summaries,
    summarize_state_pair,
    write_csv,
    write_required_plots,
)
from quant_fp_phase1.src.rtn4_quant_state import MODULE_TYPES, assert_matching_rtn4_states, quantize_rtn4_with_state
from quant_fp_phase1.src.runtime import ensure_model_fingerprint_on_path, load_causal_lm_and_tokenizer, set_seed, write_json
from quant_fp_phase1.src.swap_utils import swap_rtn4_block, swap_rtn4_group, swap_rtn4_module


class ParquetRowWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.writer = None
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except Exception as exc:
            raise RuntimeError("pyarrow is required for parquet outputs") from exc
        self.pa = pa
        self.pq = pq

    def write(self, rows: Iterable[dict[str, Any]]) -> None:
        rows = list(rows)
        if not rows:
            return
        table = self.pa.Table.from_pylist(rows)
        if self.writer is None:
            self.writer = self.pq.ParquetWriter(self.path, table.schema)
        self.writer.write_table(table)

    def write_table(self, table: Any) -> None:
        if table.num_rows == 0:
            return
        if self.writer is None:
            self.writer = self.pq.ParquetWriter(self.path, table.schema)
        self.writer.write_table(table)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run BASE-RTN4 vs IF-RTN4 exact difference and swap analysis.")
    parser.add_argument("--base-model", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--if-model", default="cnut1648/LLaMA2-7B-fingerprinted-SFT")
    parser.add_argument("--fingerprint-data", default="Model-Fingerprint/dataset/llama_fingerprint_chat")
    parser.add_argument("--output-dir", default="quant_fp_phase1/rtn4_base_vs_if_analysis")
    parser.add_argument("--stages", nargs="+", default=["all"], choices=["all", "behavior", "diff", "plots", "swaps", "summary"])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target-y", default="ハリネズミ")
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--max-group-swaps-per-module", type=int, default=None)
    parser.add_argument("--expected-base-verified", type=int, default=0)
    parser.add_argument("--expected-if-verified", type=int, default=8)
    parser.add_argument("--no-strict-behavior-control", action="store_true")
    return parser.parse_args(argv)


def make_rtn4_config(args: argparse.Namespace) -> RTNConfig:
    return RTNConfig(bits=4, group_size=args.group_size)


def selected_fingerprint_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    return select_positive_fingerprint_examples(load_fingerprint_examples(args.fingerprint_data), target_y=args.target_y, max_samples=8)


def load_rtn4_model_and_state(model_path: str, args: argparse.Namespace):
    model, tokenizer = load_causal_lm_and_tokenizer(model_path, args.dtype, args.device_map)
    states = quantize_rtn4_with_state(model, group_size=args.group_size)
    return model, tokenizer, states


def behavior_rows(model_variant: str, model: Any, tokenizer: Any, examples: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = []
    for sample_id, row in enumerate(iter_base_fingerprint_outputs(model, tokenizer, examples, args.target_y, args.max_new_tokens)):
        rows.append({
            "sample_id": sample_id,
            "dataset_index": row["dataset_index"],
            "model_variant": model_variant,
            "verified": bool(row["verified"]),
            "generated_text": row["generated"],
            "expected_text": row["expected"],
        })
    return rows


def verified_count(rows: Iterable[dict[str, Any]]) -> int:
    return sum(1 for row in rows if str(row.get("verified", "")).lower() in {"true", "1"} or row.get("verified") is True)


def fingerprint_drop_after_swap(verified_count_after_swap: int, total_samples: int = 8) -> int:
    return total_samples - verified_count_after_swap


def swap_generation_rows(
    swap_level: str,
    metadata: dict[str, Any],
    model_variant: str,
    behavior: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for row in behavior:
        rows.append({
            "swap_level": swap_level,
            **metadata,
            "model_variant": model_variant,
            "sample_id": row["sample_id"],
            "dataset_index": row["dataset_index"],
            "verified": row["verified"],
            "generated_text": row["generated_text"],
            "expected_text": row["expected_text"],
        })
    return rows




def select_group_swap_candidates(candidates: Iterable[dict[str, Any]], limit: int | None) -> list[dict[str, Any]]:
    rows = sorted(
        list(candidates),
        key=lambda row: (
            float(row.get("qcode_diff_ratio", 0) or 0),
            float(row.get("dequant_diff_l2", 0) or 0),
            float(row.get("dequant_diff_max", 0) or 0),
        ),
        reverse=True,
    )
    return rows if limit is None else rows[:limit]


def filter_group_rows_for_modules(path: str | Path, affected_modules: set[tuple[int, str]]) -> Iterable[dict[str, Any]]:
    if not affected_modules:
        return
    columns = [
        "tensor_name", "block_id", "module_type", "output_row", "group_id", "num_weights", "num_qcode_different",
        "qcode_diff_ratio", "dequant_diff_l2", "dequant_diff_max",
    ]
    for batch in iter_table_row_batches(path, columns=columns):
        for row in batch:
            key = (int(row["block_id"]), str(row["module_type"]))
            if key in affected_modules:
                yield row


def top_rows_from_table(path: str | Path, metric: str, limit: int = 5, columns: list[str] | None = None) -> list[dict[str, Any]]:
    heap: list[tuple[float, int, dict[str, Any]]] = []
    counter = 0
    for batch in iter_table_row_batches(path, columns=columns):
        for row in batch:
            counter += 1
            value = float(row.get(metric, 0) or 0)
            item = (value, counter, row)
            if len(heap) < limit:
                heapq.heappush(heap, item)
            elif value > heap[0][0]:
                heapq.heapreplace(heap, item)
    return [item[2] for item in sorted(heap, reverse=True)]


def full_tensor_region(tensor_name: str) -> dict[str, Any]:
    return {"tensor_name": tensor_name, "kind": "full"}


def group_region(tensor_name: str, output_row: int, group_id: int) -> dict[str, Any]:
    return {"tensor_name": tensor_name, "kind": "group", "output_row": output_row, "group_id": group_id}


def assert_only_expected_swap(model: Any, if_states: dict[str, Any], base_states: dict[str, Any], regions: list[dict[str, Any]]) -> None:
    regions_by_tensor: dict[str, list[dict[str, Any]]] = {}
    for region in regions:
        name = region["tensor_name"]
        if name not in if_states or name not in base_states:
            raise RuntimeError(f"Swap region references unknown tensor: {name}")
        if region["kind"] not in {"full", "group"}:
            raise RuntimeError(f"Unknown swap region kind: {region['kind']}")
        regions_by_tensor.setdefault(name, []).append(region)

    layers = get_transformer_linear_layers(model)
    for tensor_name, if_state in if_states.items():
        expected = if_state.dequant
        tensor_regions = regions_by_tensor.get(tensor_name, [])
        if tensor_regions:
            full_regions = [region for region in tensor_regions if region["kind"] == "full"]
            if full_regions:
                expected = base_states[tensor_name].dequant
            else:
                expected = if_state.dequant.clone()
                for region in tensor_regions:
                    output_row = int(region["output_row"])
                    group_id = int(region["group_id"])
                    sl = if_state.group_slice(group_id)
                    expected[output_row, sl] = base_states[tensor_name].dequant[output_row, sl]
        layer_name = tensor_name[:-7] if tensor_name.endswith(".weight") else tensor_name
        actual = layers[layer_name].weight.detach().cpu()
        if not torch.equal(actual, expected.cpu()):
            raise RuntimeError(f"unexpected swap state for {tensor_name}")

def write_behavior_control(base_model: Any, base_tok: Any, if_model: Any, if_tok: Any, examples: list[dict[str, Any]], args: argparse.Namespace) -> tuple[int, int]:
    rows = behavior_rows("BASE-RTN4", base_model, base_tok, examples, args)
    rows.extend(behavior_rows("IF-RTN4", if_model, if_tok, examples, args))
    path = Path(args.output_dir) / "results" / "rtn4_behavior_control.csv"
    write_csv(path, rows)
    base_count = verified_count(row for row in rows if row["model_variant"] == "BASE-RTN4")
    if_count = verified_count(row for row in rows if row["model_variant"] == "IF-RTN4")
    if not args.no_strict_behavior_control and (base_count != args.expected_base_verified or if_count != args.expected_if_verified):
        raise RuntimeError(
            f"Behavior control failed: BASE-RTN4={base_count}/8, IF-RTN4={if_count}/8; "
            f"expected {args.expected_base_verified}/8 and {args.expected_if_verified}/8."
        )
    return base_count, if_count


def _push_top(heaps: dict[str, list[tuple[float, int, dict[str, Any]]]], metric: str, row: dict[str, Any], counter: int, limit: int = 100) -> None:
    compact = {k: row[k] for k in [
        "tensor_name", "block_id", "module_type", "output_row", "group_id", "num_weights", "num_qcode_different",
        "qcode_diff_ratio", "dequant_diff_l2", "dequant_diff_max",
    ] if k in row}
    heap = heaps.setdefault(metric, [])
    value = float(row[metric])
    item = (value, counter, compact)
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif value > heap[0][0]:
        heapq.heapreplace(heap, item)


def write_exact_diff_and_aggregates(base_states: dict[str, Any], if_states: dict[str, Any], args: argparse.Namespace) -> None:
    assert_matching_rtn4_states(base_states, if_states)
    results = Path(args.output_dir) / "results"
    coord_writer = ParquetRowWriter(results / "rtn4_exact_diff_coordinates.parquet")
    group_writer = ParquetRowWriter(results / "rtn4_exact_diff_groups.parquet")
    row_writer = ParquetRowWriter(results / "rtn4_diff_by_row.parquet")
    tensor_rows = []
    heaps: dict[str, list[tuple[float, int, dict[str, Any]]]] = {}
    counter = 0
    try:
        total_coordinates = sum(int(base_states[name].dequant.numel()) for name in base_states)
        with tqdm(total=total_coordinates, desc="RTN4 exact diff coordinates", unit="w") as progress:
            for name in sorted(base_states):
                base = base_states[name]
                if_state = if_states[name]
                for output_row in range(base.shape[0]):
                    table = coordinate_row_table(base, if_state, output_row, coord_writer.pa)
                    coord_writer.write_table(table)
                    progress.update(table.num_rows)
                group_rows = list(iter_group_rows(base, if_state))
                group_writer.write(group_rows)
                row_writer.write(row_summaries({name: base}, {name: if_state}))
                tensor_rows.append(summarize_state_pair(base, if_state))
                for group_row in group_rows:
                    counter += 1
                    for metric in ("qcode_diff_ratio", "dequant_diff_l2", "dequant_diff_max"):
                        _push_top(heaps, metric, group_row, counter)
    finally:
        coord_writer.close()
        group_writer.close()
        row_writer.close()

    block_rows = aggregate_numeric(tensor_rows, ["block_id"])
    module_rows = aggregate_numeric(tensor_rows, ["module_type"])
    block_module_rows = aggregate_numeric(tensor_rows, ["block_id", "module_type"])
    write_csv(results / "rtn4_diff_by_tensor.csv", sorted(tensor_rows, key=lambda row: (-row["qcode_diff_ratio"], -row["dequant_diff_l2"])))
    write_csv(results / "rtn4_diff_by_block.csv", block_rows)
    write_csv(results / "rtn4_diff_by_module.csv", module_rows)
    write_csv(results / "rtn4_diff_by_block_module.csv", block_module_rows)
    for metric, filename in [
        ("qcode_diff_ratio", "rtn4_diff_groups_top_by_qcode_ratio.csv"),
        ("dequant_diff_l2", "rtn4_diff_groups_top_by_l2.csv"),
        ("dequant_diff_max", "rtn4_diff_groups_top_by_maxdiff.csv"),
    ]:
        top = [item[2] for item in sorted(heaps.get(metric, []), reverse=True)]
        write_csv(results / filename, top)


def _snapshot_layers(model: Any, tensor_names: list[str]) -> dict[str, torch.Tensor]:
    layers = get_transformer_linear_layers(model)
    snapshots = {}
    for tensor_name in tensor_names:
        layer_name = tensor_name[:-7] if tensor_name.endswith(".weight") else tensor_name
        snapshots[layer_name] = layers[layer_name].weight.detach().clone()
    return snapshots


def _restore_layers(model: Any, snapshots: dict[str, torch.Tensor]) -> None:
    layers = get_transformer_linear_layers(model)
    with torch.no_grad():
        for name, tensor in snapshots.items():
            layers[name].weight.data.copy_(tensor.to(layers[name].weight.device))


def _assert_restored_layers(model: Any, snapshots: dict[str, torch.Tensor]) -> None:
    layers = get_transformer_linear_layers(model)
    for name, tensor in snapshots.items():
        if not torch.equal(layers[name].weight.detach().cpu(), tensor.cpu()):
            raise RuntimeError(f"Swap restore check failed for {name}")


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def run_swaps(base_model: Any, if_model: Any, tokenizer: Any, examples: list[dict[str, Any]], args: argparse.Namespace, base_states: dict[str, Any], if_states: dict[str, Any]) -> None:
    results = Path(args.output_dir) / "results"
    total_samples = len(examples)
    block_metrics = {int(row["block_id"]): row for row in _read_csv(results / "rtn4_diff_by_block.csv")}
    block_module_metrics = {(int(row["block_id"]), row["module_type"]): row for row in _read_csv(results / "rtn4_diff_by_block_module.csv")}
    baseline_rows = behavior_rows("IF-RTN4", if_model, tokenizer, examples, args)
    baseline_count = verified_count(baseline_rows)
    n_blocks = int(getattr(if_model.config, "num_hidden_layers"))

    block_rows = []
    block_generation_rows = []
    for block_id in tqdm(range(n_blocks), desc="RTN4 block swaps"):
        touched = [name for name in get_transformer_linear_layers(if_model) if f"model.layers.{block_id}." in name]
        tensor_names = [f"{name}.weight" for name in touched]
        snapshots = _snapshot_layers(if_model, tensor_names)
        touched_after_swap = swap_rtn4_block(if_model, base_model, block_id)
        assert_only_expected_swap(if_model, if_states, base_states, [full_tensor_region(name) for name in touched_after_swap])
        variant = f"swap_block_{block_id}"
        behavior = behavior_rows(variant, if_model, tokenizer, examples, args)
        count = verified_count(behavior)
        _restore_layers(if_model, snapshots)
        _assert_restored_layers(if_model, snapshots)
        block_generation_rows.extend(swap_generation_rows("block", {"block_id": block_id}, variant, behavior))
        metrics = block_metrics.get(block_id, {})
        block_rows.append({
            "block_id": block_id,
            "baseline_if_verified_count": baseline_count,
            "verified_count_after_swap": count,
            "fingerprint_drop": fingerprint_drop_after_swap(count, total_samples),
            "block_qcode_diff_ratio": metrics.get("qcode_diff_ratio", ""),
            "block_dequant_diff_l2": metrics.get("dequant_diff_l2", ""),
        })
    write_csv(results / "rtn4_block_swap.csv", block_rows)
    write_csv(results / "rtn4_block_swap_generations.csv", block_generation_rows)

    affected_blocks = [int(row["block_id"]) for row in block_rows if int(row["fingerprint_drop"]) > 0]
    module_rows = []
    module_generation_rows = []
    for block_id in affected_blocks:
        for module_type in MODULE_TYPES:
            touched = [name for name in get_transformer_linear_layers(if_model) if f"model.layers.{block_id}." in name and module_type in name]
            if not touched:
                continue
            tensor_names = [f"{name}.weight" for name in touched]
            snapshots = _snapshot_layers(if_model, tensor_names)
            touched_after_swap = swap_rtn4_module(if_model, base_model, block_id, module_type)
            assert_only_expected_swap(if_model, if_states, base_states, [full_tensor_region(name) for name in touched_after_swap])
            variant = f"swap_{block_id}_{module_type}"
            behavior = behavior_rows(variant, if_model, tokenizer, examples, args)
            count = verified_count(behavior)
            _restore_layers(if_model, snapshots)
            _assert_restored_layers(if_model, snapshots)
            module_generation_rows.extend(swap_generation_rows("module", {"block_id": block_id, "module_type": module_type}, variant, behavior))
            metrics = block_module_metrics.get((block_id, module_type), {})
            module_rows.append({
                "block_id": block_id,
                "module_type": module_type,
                "baseline_if_verified_count": baseline_count,
                "verified_count_after_swap": count,
                "fingerprint_drop": fingerprint_drop_after_swap(count, total_samples),
                "module_qcode_diff_ratio": metrics.get("qcode_diff_ratio", ""),
                "module_dequant_diff_l2": metrics.get("dequant_diff_l2", ""),
            })
    write_csv(results / "rtn4_module_swap.csv", module_rows)
    write_csv(results / "rtn4_module_swap_generations.csv", module_generation_rows)

    affected_modules = {(int(row["block_id"]), row["module_type"]) for row in module_rows if int(row["fingerprint_drop"]) > 0}
    group_rows = []
    group_generation_rows = []
    if affected_modules:
        for block_id, module_type in sorted(affected_modules):
            matching_groups = filter_group_rows_for_modules(results / "rtn4_exact_diff_groups.parquet", {(block_id, module_type)})
            candidates = select_group_swap_candidates(matching_groups, args.max_group_swaps_per_module)
            for group in candidates:
                tensor_name = str(group["tensor_name"])
                snapshots = _snapshot_layers(if_model, [tensor_name])
                output_row = int(group["output_row"])
                group_id = int(group["group_id"])
                swap_rtn4_group(if_model, base_model, tensor_name, output_row, group_id, args.group_size)
                assert_only_expected_swap(if_model, if_states, base_states, [group_region(tensor_name, output_row, group_id)])
                metadata = {
                    "tensor_name": tensor_name,
                    "block_id": int(group["block_id"]),
                    "module_type": str(group["module_type"]),
                    "output_row": output_row,
                    "group_id": group_id,
                }
                variant = "swap_group"
                behavior = behavior_rows(variant, if_model, tokenizer, examples, args)
                count = verified_count(behavior)
                _restore_layers(if_model, snapshots)
                _assert_restored_layers(if_model, snapshots)
                group_generation_rows.extend(swap_generation_rows("group", metadata, variant, behavior))
                group_rows.append({
                    **metadata,
                    "num_weights": int(group["num_weights"]),
                    "num_qcode_different": int(group["num_qcode_different"]),
                    "qcode_diff_ratio": float(group["qcode_diff_ratio"]),
                    "dequant_diff_l2": float(group["dequant_diff_l2"]),
                    "dequant_diff_max": float(group["dequant_diff_max"]),
                    "baseline_if_verified_count": baseline_count,
                    "verified_count_after_swap": count,
                    "fingerprint_drop": fingerprint_drop_after_swap(count, total_samples),
                })
    write_csv(results / "rtn4_group_swap.csv", group_rows)
    write_csv(results / "rtn4_group_swap_generations.csv", group_generation_rows)


def _top_rows(rows: list[dict[str, Any]], metric: str, limit: int = 5) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row.get(metric, 0) or 0), reverse=True)[:limit]


def _append_location_lines(lines: list[str], title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines.append(f"### {title}")
    lines.append("")
    if not rows:
        lines.append("- No rows available.")
    for row in rows:
        loc = ", ".join(f"{field}={row.get(field, '')}" for field in fields)
        lines.append(f"- {loc}: qcode_diff_ratio={float(row.get('qcode_diff_ratio', 0) or 0):.8f}, dequant_diff_l2={float(row.get('dequant_diff_l2', 0) or 0):.8f}")
    lines.append("")




def _append_causal_lines(lines: list[str], title: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    lines.append(f"### {title}")
    lines.append("")
    if not rows:
        lines.append("- None.")
    for row in rows:
        loc = ", ".join(f"{field}={row.get(field, '')}" for field in fields)
        lines.append(
            f"- {loc}: verified_count_after_swap={row.get('verified_count_after_swap', '')}, "
            f"fingerprint_drop={row.get('fingerprint_drop', '')}"
        )
    lines.append("")

def write_summary(args: argparse.Namespace) -> None:
    results = Path(args.output_dir) / "results"
    summary_path = Path(args.output_dir) / "RTN4_BASE_VS_IF_SUMMARY.md"
    tensor_rows = _read_csv(results / "rtn4_diff_by_tensor.csv")
    block_rows = _read_csv(results / "rtn4_diff_by_block.csv")
    block_module_rows = _read_csv(results / "rtn4_diff_by_block_module.csv")
    row_rows = top_rows_from_table(results / "rtn4_diff_by_row.parquet", "qcode_diff_ratio", limit=5)
    group_rows_for_summary = _read_csv(results / "rtn4_diff_groups_top_by_qcode_ratio.csv")
    block_swap = _read_csv(results / "rtn4_block_swap.csv")
    module_swap = _read_csv(results / "rtn4_module_swap.csv")
    group_swap = _read_csv(results / "rtn4_group_swap.csv")
    total_weights = sum(int(float(row["num_weights"])) for row in tensor_rows) if tensor_rows else 0
    total_qdiff = sum(int(float(row["num_qcode_different"])) for row in tensor_rows) if tensor_rows else 0
    total_l1 = sum(float(row["dequant_diff_l1"]) for row in tensor_rows) if tensor_rows else 0.0
    total_l2 = sum(float(row["dequant_diff_l2"]) ** 2 for row in tensor_rows) ** 0.5 if tensor_rows else 0.0
    scale_groups = sum(int(float(row.get("num_groups_scale_different", 0))) for row in tensor_rows)
    zp_groups = sum(int(float(row.get("num_groups_zp_different", 0))) for row in tensor_rows)
    total_groups = sum(int(float(row.get("num_groups", 0))) for row in tensor_rows)
    changed_blocks = [row for row in block_swap if int(float(row.get("fingerprint_drop", 0))) > 0]
    changed_modules = [row for row in module_swap if int(float(row.get("fingerprint_drop", 0))) > 0]
    changed_groups = [row for row in group_swap if int(float(row.get("fingerprint_drop", 0))) > 0]
    lines = [
        "# RTN4 BASE vs IF Summary",
        "",
        "## Quantization Consistency",
        "",
        "- bits: 4",
        f"- group_size: {args.group_size}",
        "- RTN function: quant_fp_phase1.src.quantization.rtn_quantize_weight_raw",
        "- tensor discovery: quant_fp_phase1.src.quantization.get_transformer_linear_layers",
        "- scope: BASE-RTN4 vs IF-RTN4 only",
        "",
        "## Question 1: How Different Are The Two Quantized Checkpoints?",
        "",
        f"- global qcode difference ratio: {(total_qdiff / total_weights) if total_weights else 0.0:.8f}",
        f"- global dequantized-weight L1 difference: {total_l1:.8f}",
        f"- global dequantized-weight L2 difference: {total_l2:.8f}",
        f"- fraction of groups with different scale: {(scale_groups / total_groups) if total_groups else 0.0:.8f}",
        f"- fraction of groups with different zero-point: {(zp_groups / total_groups) if total_groups else 0.0:.8f}",
        "",
        "## Question 2: Where Are The Differences Located?",
        "",
    ]
    _append_location_lines(lines, "Strongest block locations", _top_rows(block_rows, "qcode_diff_ratio"), ["block_id"])
    _append_location_lines(lines, "Strongest block-module locations", _top_rows(block_module_rows, "qcode_diff_ratio"), ["block_id", "module_type"])
    _append_location_lines(lines, "Strongest output-row locations", _top_rows(row_rows, "qcode_diff_ratio"), ["tensor_name", "output_row"])
    _append_location_lines(lines, "Strongest RTN-group locations", _top_rows(group_rows_for_summary, "qcode_diff_ratio"), ["tensor_name", "output_row", "group_id"])
    lines.extend([
        "Numerical concentration is not interpreted as causal evidence.",
        "",
        "## Question 3: Which Differences Are Causally Related To Fingerprint Preservation?",
        "",
        f"- block swaps that reduce verification: {len(changed_blocks)}",
        f"- module swaps that reduce verification: {len(changed_modules)}",
        f"- group swaps that reduce verification: {len(changed_groups)}",
        "",
    ])
    if not changed_blocks:
        lines.append("No single transformer block is individually necessary under the current one-block replacement test.")
        lines.append("")
    _append_causal_lines(lines, "Block swaps that reduce verification", changed_blocks, ["block_id"])
    _append_causal_lines(lines, "Module swaps that reduce verification", changed_modules, ["block_id", "module_type"])
    _append_causal_lines(lines, "Group swaps that reduce verification", changed_groups, ["tensor_name", "block_id", "module_type", "output_row", "group_id"])
    lines.append("Only swap experiments are treated as causal evidence.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_analysis_config_payload(args: argparse.Namespace) -> dict[str, Any]:
    base_rtn4_config = make_rtn4_config(args).__dict__
    if_rtn4_config = make_rtn4_config(args).__dict__
    return {
        "base_model": args.base_model,
        "if_model": args.if_model,
        "fingerprint_data": args.fingerprint_data,
        "base_rtn4_config": base_rtn4_config,
        "if_rtn4_config": if_rtn4_config,
        "rtn4_configs_identical": base_rtn4_config == if_rtn4_config,
        "dtype": args.dtype,
        "device_map": args.device_map,
        "target_y": args.target_y,
        "max_new_tokens": args.max_new_tokens,
        "stages": args.stages,
    }


def has_stage(stages: set[str], name: str) -> bool:
    return "all" in stages or name in stages


def should_reload_models_between_diff_and_swaps(args: argparse.Namespace) -> bool:
    stages = set(args.stages)
    return has_stage(stages, "diff") and has_stage(stages, "swaps")


def clear_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    ensure_model_fingerprint_on_path(Path(__file__).resolve().parents[2])
    set_seed(args.seed)
    out = Path(args.output_dir)
    results = out / "results"
    plots = out / "plots"
    results.mkdir(parents=True, exist_ok=True)
    plots.mkdir(parents=True, exist_ok=True)
    write_json(results / "rtn4_analysis_config.json", build_analysis_config_payload(args))

    stages = set(args.stages)
    run_behavior_stage = has_stage(stages, "behavior")
    run_diff_stage = has_stage(stages, "diff")
    run_plots_stage = has_stage(stages, "plots")
    run_swaps_stage = has_stage(stages, "swaps")
    run_summary_stage = has_stage(stages, "summary")
    need_models = run_behavior_stage or run_diff_stage or run_swaps_stage
    examples = selected_fingerprint_examples(args) if need_models else []
    base_model = base_tok = base_states = None
    if_model = if_tok = if_states = None

    if need_models:
        print("Loading and quantizing BASE-FP -> BASE-RTN4 with Phase 1 RTN4")
        base_model, base_tok, base_states = load_rtn4_model_and_state(args.base_model, args)
        print("Loading and quantizing IF-FP -> IF-RTN4 with Phase 1 RTN4")
        if_model, if_tok, if_states = load_rtn4_model_and_state(args.if_model, args)
        assert_matching_rtn4_states(base_states, if_states)

    if run_behavior_stage:
        base_count, if_count = write_behavior_control(base_model, base_tok, if_model, if_tok, examples, args)
        print(f"Behavior control: BASE-RTN4={base_count}/8, IF-RTN4={if_count}/8")

    if run_diff_stage:
        if base_model is not None or if_model is not None:
            print("Releasing loaded models before exact diff to reduce PBS memory peak")
            del base_model, if_model, base_tok, if_tok
            base_model = if_model = base_tok = if_tok = None
            clear_torch_memory()
        write_exact_diff_and_aggregates(base_states, if_states, args)
        if run_swaps_stage:
            print("Releasing diff quant states before reloading models for swap analysis")
            del base_states, if_states
            base_states = if_states = None
            clear_torch_memory()

    if run_plots_stage:
        write_required_plots(results, plots)

    if run_swaps_stage:
        if base_model is None or if_model is None or base_states is None or if_states is None:
            print("Reloading and quantizing models for swap analysis")
            base_model, base_tok, base_states = load_rtn4_model_and_state(args.base_model, args)
            if_model, if_tok, if_states = load_rtn4_model_and_state(args.if_model, args)
            assert_matching_rtn4_states(base_states, if_states)
        run_swaps(base_model, if_model, if_tok, examples, args, base_states, if_states)

    if run_summary_stage:
        write_summary(args)

    del base_model, if_model, base_states, if_states
    clear_torch_memory()


if __name__ == "__main__":
    main()










