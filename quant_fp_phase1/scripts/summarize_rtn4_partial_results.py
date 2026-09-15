from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize partial RTN4 BASE-vs-IF outputs without reading coordinate parquet.")
    parser.add_argument("--output-dir", default="quant_fp_phase1/rtn4_base_vs_if_analysis")
    parser.add_argument("--top-k", type=int, default=15)
    return parser.parse_args(argv)


def _print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n=== {title} ===")
    if frame.empty:
        print("<empty>")
    else:
        print(frame.to_string())


def summarize(output_dir: str | Path, top_k: int = 15) -> None:
    root = Path(output_dir)
    results = root / "results"
    behavior_path = results / "rtn4_behavior_control.csv"
    group_path = results / "rtn4_exact_diff_groups.parquet"
    row_path = results / "rtn4_diff_by_row.parquet"

    if behavior_path.exists():
        behavior = pd.read_csv(behavior_path)
        _print_table("Behavior Control", behavior.groupby("model_variant")["verified"].agg(["sum", "count"]))

    if not group_path.exists():
        print(f"\nMissing {group_path}")
        return

    print(f"\nReading {group_path} (not coordinate parquet)...")
    groups = pd.read_parquet(group_path)
    total_weights = groups["num_weights"].sum()
    total_qdiff = groups["num_qcode_different"].sum()
    l2 = (groups["dequant_diff_l2"].pow(2).sum()) ** 0.5
    print("\n=== Global Difference From RTN Groups ===")
    print(f"groups: {len(groups):,}")
    print(f"weights: {int(total_weights):,}")
    print(f"qcode different: {int(total_qdiff):,}")
    print(f"qcode diff ratio: {total_qdiff / total_weights:.8f}")
    if "scale_different" in groups:
        print(f"fraction groups scale different: {groups['scale_different'].mean():.8f}")
    if "zero_point_different" in groups:
        print(f"fraction groups zero-point different: {groups['zero_point_different'].mean():.8f}")
    print(f"dequant L1: {groups['dequant_diff_l1'].sum():.8f}")
    print(f"dequant L2: {l2:.8f}")
    print(f"max abs dequant diff: {groups['dequant_diff_max'].max():.8f}")

    block = groups.groupby("block_id").agg(
        num_weights=("num_weights", "sum"),
        num_qcode_different=("num_qcode_different", "sum"),
        dequant_diff_l1=("dequant_diff_l1", "sum"),
        dequant_diff_l2_sq=("dequant_diff_l2", lambda x: (x * x).sum()),
        dequant_diff_max=("dequant_diff_max", "max"),
    )
    block["qcode_diff_ratio"] = block["num_qcode_different"] / block["num_weights"]
    block["dequant_diff_l2"] = block["dequant_diff_l2_sq"] ** 0.5
    _print_table("Top Blocks By QCode Diff Ratio", block.sort_values(["qcode_diff_ratio", "dequant_diff_l2"], ascending=False).head(top_k)[["num_weights", "num_qcode_different", "qcode_diff_ratio", "dequant_diff_l2", "dequant_diff_max"]])

    module = groups.groupby("module_type").agg(
        num_weights=("num_weights", "sum"),
        num_qcode_different=("num_qcode_different", "sum"),
        dequant_diff_l2_sq=("dequant_diff_l2", lambda x: (x * x).sum()),
        dequant_diff_max=("dequant_diff_max", "max"),
    )
    module["qcode_diff_ratio"] = module["num_qcode_different"] / module["num_weights"]
    module["dequant_diff_l2"] = module["dequant_diff_l2_sq"] ** 0.5
    _print_table("Modules", module.sort_values("qcode_diff_ratio", ascending=False)[["num_weights", "num_qcode_different", "qcode_diff_ratio", "dequant_diff_l2", "dequant_diff_max"]])

    block_module = groups.groupby(["block_id", "module_type"]).agg(
        num_weights=("num_weights", "sum"),
        num_qcode_different=("num_qcode_different", "sum"),
        dequant_diff_l2_sq=("dequant_diff_l2", lambda x: (x * x).sum()),
        dequant_diff_max=("dequant_diff_max", "max"),
    )
    block_module["qcode_diff_ratio"] = block_module["num_qcode_different"] / block_module["num_weights"]
    block_module["dequant_diff_l2"] = block_module["dequant_diff_l2_sq"] ** 0.5
    _print_table("Top Block x Module By QCode Diff Ratio", block_module.sort_values(["qcode_diff_ratio", "dequant_diff_l2"], ascending=False).head(top_k)[["num_weights", "num_qcode_different", "qcode_diff_ratio", "dequant_diff_l2", "dequant_diff_max"]])

    group_cols = ["tensor_name", "block_id", "module_type", "output_row", "group_id", "num_weights", "num_qcode_different", "qcode_diff_ratio", "dequant_diff_l2", "dequant_diff_max"]
    _print_table("Top RTN Groups By QCode Diff Ratio", groups.sort_values(["qcode_diff_ratio", "dequant_diff_l2", "dequant_diff_max"], ascending=False).head(top_k)[group_cols].reset_index(drop=True))

    if row_path.exists():
        print(f"\nReading {row_path}...")
        rows = pd.read_parquet(row_path)
        row_cols = ["tensor_name", "block_id", "module_type", "output_row", "num_weights", "num_qcode_different", "qcode_diff_ratio", "dequant_diff_l2", "dequant_diff_max"]
        _print_table("Top Output Rows By QCode Diff Ratio", rows.sort_values(["qcode_diff_ratio", "dequant_diff_l2", "dequant_diff_max"], ascending=False).head(top_k)[row_cols].reset_index(drop=True))


if __name__ == "__main__":
    summarize(**vars(parse_args()))