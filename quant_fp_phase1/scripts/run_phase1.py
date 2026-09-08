from __future__ import annotations

import argparse
from pathlib import Path

from quant_fp_phase1.src.experiments import run_baseline, run_blockwise, run_delta, run_margin, run_sweep
from quant_fp_phase1.src.plots import generate_plots
from quant_fp_phase1.src.runtime import ensure_model_fingerprint_on_path, set_seed, write_json


def parse_args():
    parser = argparse.ArgumentParser(description="Run Phase 1 IF-SFT quantization analysis")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--if-model", required=True)
    parser.add_argument("--fingerprint-data", required=True)
    parser.add_argument("--output-dir", default="quant_fp_phase1")
    parser.add_argument("--stages", nargs="+", default=["all"], choices=["all", "baseline", "margin", "sweep", "delta", "blockwise", "plots"])
    parser.add_argument("--baseline-bits", nargs="+", type=int, default=[4, 3])
    parser.add_argument("--sweep-bits", nargs="+", type=int, default=[8, 6, 5, 4, 3])
    parser.add_argument("--delta-bits", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target-y", default="ハリネズミ")
    parser.add_argument("--run-ppl", action="store_true")
    parser.add_argument("--ppl-datasets", nargs="+", default=["wikitext2"])
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--cache-dir", default="./dataset_cache")
    parser.add_argument("--max-margin-samples", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_model_fingerprint_on_path(Path(__file__).resolve().parents[2])
    set_seed(args.seed)
    out = Path(args.output_dir)
    (out / "results" / "configs").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)
    write_json(out / "results" / "configs" / "phase1_args.json", vars(args))

    stages = set(args.stages)
    if "all" in stages or "baseline" in stages:
        run_baseline(args)
    if "all" in stages or "margin" in stages:
        run_margin(args)
    if "all" in stages or "sweep" in stages:
        run_sweep(args)
    if "all" in stages or "delta" in stages:
        run_delta(args)
    if "all" in stages or "blockwise" in stages:
        run_blockwise(args)
    if "all" in stages or "plots" in stages:
        generate_plots(args.output_dir)


if __name__ == "__main__":
    main()

