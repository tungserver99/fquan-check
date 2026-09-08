from __future__ import annotations

import csv
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def generate_plots(output_dir: str | Path) -> None:
    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    results = output_dir / "results"
    plots = output_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    sweep = _read_csv(results / "bitwidth_sweep.csv") or _read_csv(results / "baseline.csv")
    if sweep:
        x = [16 if row["bits"] == "fp" else int(row["bits"]) for row in sweep]
        score = [float(row["fingerprint_score"]) for row in sweep]
        plt.figure()
        plt.plot(x, score, marker="o")
        plt.xlabel("Bits")
        plt.ylabel("Fingerprint score")
        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.savefig(plots / "bitwidth_vs_fingerprint_score.png")
        plt.close()
        ppl = [float(row["wikitext2_ppl"]) for row in sweep if row.get("wikitext2_ppl")]
        if len(ppl) == len(x):
            plt.figure()
            plt.plot(x, ppl, marker="o")
            plt.xlabel("Bits")
            plt.ylabel("WikiText-2 PPL")
            plt.gca().invert_xaxis()
            plt.tight_layout()
            plt.savefig(plots / "bitwidth_vs_ppl.png")
            plt.close()

    margin = _read_csv(results / "fingerprint_margin_per_sample.csv")
    if margin:
        by_variant: dict[str, list[float]] = {}
        for row in margin:
            by_variant.setdefault(row["model_variant"], []).append(float(row["mean_margin"]))
        labels = list(by_variant)
        plt.figure()
        plt.boxplot([by_variant[label] for label in labels], labels=labels)
        plt.ylabel("Mean target margin")
        plt.tight_layout()
        plt.savefig(plots / "bitwidth_vs_margin.png")
        plt.savefig(plots / "fingerprint_margin_distribution.png")
        plt.close()

    for bits in (3, 4):
        block_delta = _read_csv(results / f"delta_survival_rtn{bits}_by_block.csv")
        if block_delta:
            plt.figure()
            plt.bar([int(row["block_id"]) for row in block_delta], [float(row["delta_norm_survival_mean"]) for row in block_delta])
            plt.xlabel("Block")
            plt.ylabel("Delta norm survival")
            plt.tight_layout()
            plt.savefig(plots / f"block_delta_survival_rtn{bits}.png")
            plt.close()


    blockwise = _read_csv(results / "blockwise_rtn3.csv")
    if blockwise:
        with_ppl = [row for row in blockwise if row.get("wikitext2_ppl") and row.get("margin_drop_from_fp")]
        if len(with_ppl) >= 2:
            base_ppl = min(float(row["wikitext2_ppl"]) for row in with_ppl)
            plt.figure()
            plt.scatter([float(row["wikitext2_ppl"]) - base_ppl for row in with_ppl], [float(row["margin_drop_from_fp"]) for row in with_ppl])
            plt.xlabel("PPL increase from best blockwise run")
            plt.ylabel("Fingerprint margin drop")
            plt.tight_layout()
            plt.savefig(plots / "utility_drop_vs_fingerprint_drop.png")
            plt.savefig(plots / "blockwise_ppl_increase_vs_margin_drop.png")
            plt.close()
        with_delta = [row for row in blockwise if row.get("delta_norm_survival_rtn3") and row.get("margin_drop_from_fp")]
        if len(with_delta) >= 2:
            plt.figure()
            plt.scatter([float(row["delta_norm_survival_rtn3"]) for row in with_delta], [float(row["margin_drop_from_fp"]) for row in with_delta])
            plt.xlabel("RTN3 delta survival")
            plt.ylabel("Fingerprint margin drop")
            plt.tight_layout()
            plt.savefig(plots / "delta_survival_vs_margin_drop.png")
            plt.close()
