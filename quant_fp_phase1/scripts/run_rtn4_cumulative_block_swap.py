from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm

from quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis import behavior_rows, load_rtn4_model_and_state, verified_count
from quant_fp_phase1.src.experiments import load_fingerprint_examples, select_positive_fingerprint_examples
from quant_fp_phase1.src.fingerprint_eval import DEFAULT_TARGET_Y
from quant_fp_phase1.src.quantization import get_transformer_linear_layers
from quant_fp_phase1.src.rtn4_difference import write_csv
from quant_fp_phase1.src.rtn4_quant_state import MODULE_TYPES, assert_matching_rtn4_states
from quant_fp_phase1.src.runtime import ensure_model_fingerprint_on_path, set_seed

COARSE_COUNTS = (4, 8, 12, 16, 20, 24, 28, 32)
DISTRIBUTED_ORDER = (
    0, 16, 8, 24,
    4, 20, 12, 28,
    2, 18, 10, 26,
    6, 22, 14, 30,
    1, 17, 9, 25,
    5, 21, 13, 29,
    3, 19, 11, 27,
    7, 23, 15, 31,
)
FAMILY_PREFIX = {"prefix": "P", "suffix": "S", "distributed": "D"}


@dataclass(frozen=True)
class SwapConfig:
    config_id: str
    family: str
    blocks: tuple[int, ...]

    @property
    def num_swapped_blocks(self) -> int:
        return len(self.blocks)

    @property
    def swapped_blocks_text(self) -> str:
        return " ".join(str(block) for block in self.blocks)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cumulative RTN4 block-swap analysis.")
    parser.add_argument("--base-model", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--if-model", default="cnut1648/LLaMA2-7B-fingerprinted-SFT")
    parser.add_argument("--fingerprint-data", default="Model-Fingerprint/dataset/llama_fingerprint_chat")
    parser.add_argument("--output-dir", default="quant_fp_phase1/rtn4_cumulative_block_swap_analysis")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target-y", default=DEFAULT_TARGET_Y)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument("--no-refine", action="store_true", help="Run only the coarse counts from the spec.")
    parser.add_argument("--only-all32", action="store_true", help="Run only the full transformer quantized-weight replacement from BASE-RTN4 into IF-RTN4.")
    return parser.parse_args(argv)


def config_for_family(family: str, k: int, n_blocks: int) -> SwapConfig:
    if family == "prefix":
        blocks = tuple(range(k))
    elif family == "suffix":
        blocks = tuple(range(n_blocks - k, n_blocks))
    elif family == "distributed":
        order = tuple(block for block in DISTRIBUTED_ORDER if block < n_blocks)
        blocks = order[:k]
    else:
        raise ValueError(f"Unknown family: {family}")
    return SwapConfig(f"{FAMILY_PREFIX[family]}{k:02d}", family, blocks)


def build_initial_configs(n_blocks: int) -> list[SwapConfig]:
    configs = [
        SwapConfig("IF-RTN4", "control", ()),
        SwapConfig("BASE-RTN4", "control", ()),
    ]
    for family in ("prefix", "suffix", "distributed"):
        for k in COARSE_COUNTS:
            if k >= n_blocks:
                continue
            configs.append(config_for_family(family, k, n_blocks))
    configs.append(SwapConfig("ALL32", "all", tuple(range(n_blocks))))
    return configs


def configs_for_run(args: argparse.Namespace, n_blocks: int) -> list[SwapConfig]:
    if getattr(args, "only_all32", False):
        return [SwapConfig("ALL32", "all", tuple(range(n_blocks)))]
    return build_initial_configs(n_blocks)

def _row_int(row: dict[str, Any], key: str) -> int:
    return int(float(row[key]))


def refinement_configs(rows: list[dict[str, Any]], n_blocks: int) -> list[SwapConfig]:
    existing = {(str(row.get("family")), _row_int(row, "num_swapped_blocks")) for row in rows if row.get("family") in FAMILY_PREFIX}
    all32 = next((row for row in rows if row.get("config_id") == "ALL32"), None)
    configs: list[SwapConfig] = []
    seen: set[tuple[str, int]] = set()
    for family in ("prefix", "suffix", "distributed"):
        family_rows = [row for row in rows if row.get("family") == family]
        if all32 is not None:
            family_rows.append({**all32, "family": family, "num_swapped_blocks": n_blocks})
        family_rows = sorted(family_rows, key=lambda row: _row_int(row, "num_swapped_blocks"))
        for left, right in zip(family_rows, family_rows[1:]):
            left_k = _row_int(left, "num_swapped_blocks")
            right_k = _row_int(right, "num_swapped_blocks")
            if _row_int(left, "verified_count") == _row_int(right, "verified_count"):
                continue
            for k in range(left_k + 1, right_k):
                key = (family, k)
                if key in existing or key in seen:
                    continue
                configs.append(config_for_family(family, k, n_blocks))
                seen.add(key)
    return configs


def clear_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_model_to_if_rtn4(model: Any, if_states: dict[str, Any]) -> None:
    layers = get_transformer_linear_layers(model)
    with torch.no_grad():
        for tensor_name, state in if_states.items():
            layer_name = tensor_name[:-7] if tensor_name.endswith(".weight") else tensor_name
            layers[layer_name].weight.data.copy_(state.dequant.to(layers[layer_name].weight.device))


def _tensor_names_for_blocks(states: dict[str, Any], blocks: Iterable[int]) -> list[str]:
    block_set = set(int(block) for block in blocks)
    return sorted(name for name, state in states.items() if int(state.block_id) in block_set)


def apply_cumulative_block_swap(
    model: Any,
    base_states: dict[str, Any],
    if_states: dict[str, Any],
    blocks: Iterable[int],
) -> list[str]:
    assert_matching_rtn4_states(base_states, if_states)
    blocks = tuple(int(block) for block in blocks)
    reset_model_to_if_rtn4(model, if_states)
    tensor_names = _tensor_names_for_blocks(base_states, blocks)
    expected = len(blocks) * len(MODULE_TYPES)
    if len(tensor_names) != expected:
        raise RuntimeError(f"Expected {expected} tensors for blocks {blocks}, found {len(tensor_names)}")
    layers = get_transformer_linear_layers(model)
    with torch.no_grad():
        for tensor_name in tensor_names:
            layer_name = tensor_name[:-7] if tensor_name.endswith(".weight") else tensor_name
            layers[layer_name].weight.data.copy_(base_states[tensor_name].dequant.to(layers[layer_name].weight.device))
    return tensor_names


def assert_cumulative_swap_state(
    model: Any,
    base_states: dict[str, Any],
    if_states: dict[str, Any],
    blocks: Iterable[int],
) -> None:
    block_set = set(int(block) for block in blocks)
    layers = get_transformer_linear_layers(model)
    for tensor_name, if_state in if_states.items():
        layer_name = tensor_name[:-7] if tensor_name.endswith(".weight") else tensor_name
        expected = base_states[tensor_name].dequant if int(if_state.block_id) in block_set else if_state.dequant
        actual = layers[layer_name].weight.detach().cpu()
        if not torch.equal(actual, expected.cpu()):
            raise RuntimeError(f"Cumulative swap state mismatch for {tensor_name}")


def selected_fingerprint_examples(args: argparse.Namespace) -> list[dict[str, Any]]:
    return select_positive_fingerprint_examples(
        load_fingerprint_examples(args.fingerprint_data),
        target_y=args.target_y,
        max_samples=8,
    )


def run_behavior_for_config(
    config: SwapConfig,
    base_model: Any,
    base_tok: Any,
    if_model: Any,
    if_tok: Any,
    base_states: dict[str, Any],
    if_states: dict[str, Any],
    examples: list[dict[str, Any]],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if config.config_id == "BASE-RTN4":
        behavior = behavior_rows(config.config_id, base_model, base_tok, examples, args)
    else:
        if config.config_id == "IF-RTN4":
            reset_model_to_if_rtn4(if_model, if_states)
            replaced = []
        else:
            print(f"{config.config_id}: selected blocks = {len(config.blocks)}")
            print(f"{config.config_id}: expected replaced tensors = {len(config.blocks) * len(MODULE_TYPES)}")
            replaced = apply_cumulative_block_swap(if_model, base_states, if_states, config.blocks)
            print(f"{config.config_id}: replaced tensors = {len(replaced)}")
            assert_cumulative_swap_state(if_model, base_states, if_states, config.blocks)
        behavior = behavior_rows(config.config_id, if_model, if_tok, examples, args)

    count = verified_count(behavior)
    row = {
        "config_id": config.config_id,
        "family": config.family,
        "num_swapped_blocks": config.num_swapped_blocks,
        "swapped_blocks": config.swapped_blocks_text,
        "verified_count": count,
        "fingerprint_score": count / len(examples) if examples else 0.0,
    }
    for sample_id in range(8):
        row[f"sample_{sample_id}_verified"] = bool(behavior[sample_id]["verified"]) if sample_id < len(behavior) else ""

    generation_rows = []
    for item in behavior:
        generation_rows.append({
            "config_id": config.config_id,
            "family": config.family,
            "num_swapped_blocks": config.num_swapped_blocks,
            "swapped_blocks": config.swapped_blocks_text,
            "sample_id": item["sample_id"],
            "dataset_index": item["dataset_index"],
            "verified": item["verified"],
            "expected_text": item["expected_text"],
            "generated_text": item["generated_text"],
        })
    return row, generation_rows


def write_cumulative_plot(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    out = Path(output_dir)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    all32 = next((row for row in rows if row["config_id"] == "ALL32"), None)
    plt.figure(figsize=(7, 4))
    for family in ("prefix", "suffix", "distributed"):
        family_rows = [row for row in rows if row["family"] == family]
        if all32 is not None:
            family_rows.append({**all32, "family": family})
        family_rows = sorted(family_rows, key=lambda row: int(row["num_swapped_blocks"]))
        xs = [int(row["num_swapped_blocks"]) for row in family_rows]
        ys = [int(row["verified_count"]) for row in family_rows]
        plt.plot(xs, ys, marker="o", label=family)
    plt.xlabel("number of BASE-RTN4 blocks inserted into IF-RTN4")
    plt.ylabel("fingerprint verified count")
    plt.ylim(-0.25, 8.25)
    plt.yticks(range(0, 9))
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots / "rtn4_cumulative_block_swap.png")
    plt.close()


def first_drop(rows: list[dict[str, Any]], family: str) -> str:
    all32 = next((row for row in rows if row["config_id"] == "ALL32"), None)
    family_rows = [row for row in rows if row["family"] == family]
    if all32 is not None:
        family_rows.append({**all32, "family": family})
    for row in sorted(family_rows, key=lambda item: int(item["num_swapped_blocks"])):
        if int(row["verified_count"]) < 8:
            return f"{row['num_swapped_blocks']} blocks ({row['config_id']} = {row['verified_count']}/8)"
    return "no decrease observed"


def write_summary(rows: list[dict[str, Any]], output_dir: str | Path) -> None:
    out = Path(output_dir)
    all32 = next((row for row in rows if row["config_id"] == "ALL32"), None)
    drops = {family: first_drop(rows, family) for family in ("prefix", "suffix", "distributed")}
    all32_text = "not run" if all32 is None else f"{all32['verified_count']}/8"
    lines = [
        "# RTN4 Cumulative Block-Swap Summary",
        "",
        "## First Verification Decrease",
        "",
        f"- prefix: {drops['prefix']}",
        f"- suffix: {drops['suffix']}",
        f"- distributed: {drops['distributed']}",
        "",
        "## Fastest Damage Pattern",
        "",
        "Compare the first-decrease rows above; smaller swapped-block count means faster damage.",
        "",
        "## ALL32",
        "",
        f"- ALL32 verified count: {all32_text}",
        "",
        "## Interpretation",
        "",
        "Use the three curves to decide whether fingerprint-preserving RTN4 residuals are localized by depth or distributed across transformer depth.",
    ]
    (out / "RTN4_CUMULATIVE_BLOCK_SWAP_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(args: argparse.Namespace) -> None:
    ensure_model_fingerprint_on_path(Path(__file__).resolve().parents[2])
    set_seed(args.seed)
    out = Path(args.output_dir)
    results = out / "results"
    results.mkdir(parents=True, exist_ok=True)
    examples = selected_fingerprint_examples(args)
    if len(examples) != 8:
        raise RuntimeError(f"Expected 8 fingerprint examples, found {len(examples)}")

    print("Loading and quantizing BASE-FP -> BASE-RTN4")
    base_model, base_tok, base_states = load_rtn4_model_and_state(args.base_model, args)
    print("Loading and quantizing IF-FP -> IF-RTN4")
    if_model, if_tok, if_states = load_rtn4_model_and_state(args.if_model, args)
    assert_matching_rtn4_states(base_states, if_states)
    n_blocks = int(getattr(if_model.config, "num_hidden_layers"))

    rows: list[dict[str, Any]] = []
    generation_rows: list[dict[str, Any]] = []
    for config in tqdm(configs_for_run(args, n_blocks), desc="cumulative swaps"):
        row, gens = run_behavior_for_config(config, base_model, base_tok, if_model, if_tok, base_states, if_states, examples, args)
        rows.append(row)
        generation_rows.extend(gens)

    if not args.no_refine and not args.only_all32:
        refine = refinement_configs(rows, n_blocks)
        for config in tqdm(refine, desc="cumulative refinements"):
            row, gens = run_behavior_for_config(config, base_model, base_tok, if_model, if_tok, base_states, if_states, examples, args)
            rows.append(row)
            generation_rows.extend(gens)

    write_csv(results / "rtn4_cumulative_block_swap.csv", rows)
    write_csv(results / "rtn4_cumulative_block_swap_generations.csv", generation_rows)
    write_cumulative_plot(rows, out)
    write_summary(rows, out)
    del base_model, if_model, base_states, if_states
    clear_torch_memory()


def main() -> None:
    run_analysis(parse_args())


if __name__ == "__main__":
    main()
