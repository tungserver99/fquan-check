from __future__ import annotations

import csv
import gc
import json
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from .delta_metrics import aggregate_by_block, tensor_delta_stats
from .fingerprint_eval import DEFAULT_TARGET_Y, calc_if_sft_fsr, row_verified
from .margin_metrics import build_vicuna_fingerprint_prompt, compute_teacher_forced_margin, summarize_token_records, write_token_records
from .quantization import RTNConfig, quantize_model_linear_weights, rtn_quantize_tensor, save_rtn_config, transformer_block_include
from .runtime import env_report, load_causal_lm_and_tokenizer, set_seed, write_json


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_fingerprint_examples(data_path: str, split_names: tuple[str, ...] = ("validation", "test")) -> list[dict[str, Any]]:
    from datasets import load_from_disk

    dataset = load_from_disk(data_path)
    examples: list[dict[str, Any]] = []
    for split_name in split_names:
        if split_name in dataset:
            examples.extend(list(dataset[split_name]))
    return examples


@torch.no_grad()
def generate_predictions(model, tokenizer, examples: list[dict[str, Any]], out_jsonl: str | Path) -> None:
    from transformers import GenerationConfig

    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    gen_config = GenerationConfig(
        max_new_tokens=30,
        temperature=0.0,
        top_p=0.95,
        top_k=50,
        do_sample=False,
        num_beams=1,
        output_scores=False,
        eos_token_id=[tokenizer.eos_token_id],
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )
    with out_jsonl.open("w", encoding="utf-8") as fh:
        for example in tqdm(examples, desc=f"generate {out_jsonl.stem}"):
            prompt, label = build_vicuna_fingerprint_prompt(example)
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
            generated = model.generate(input_ids=input_ids, generation_config=gen_config)[0]
            generated_str = tokenizer.decode(generated, skip_special_tokens=True)
            if generated_str.startswith(prompt):
                generated_str = generated_str[len(prompt):]
            fh.write(json.dumps({
                "generated": generated_str,
                "label": label,
                "prompt": prompt,
                "generated_token": tokenizer(generated_str, add_special_tokens=False).input_ids,
                "label_token": tokenizer(label, add_special_tokens=False).input_ids,
            }, ensure_ascii=False) + "\n")


def load_variant(if_model: str, variant: str, bits: int | None, args):
    model, tokenizer = load_causal_lm_and_tokenizer(if_model, args.dtype, args.device_map)
    if bits is not None:
        config = RTNConfig(bits=bits, group_size=args.group_size)
        quantize_model_linear_weights(model, config)
    return model, tokenizer


def run_variant_metrics(args, variant: str, bits: int | None, examples: list[dict[str, Any]]) -> dict[str, Any]:
    model, tokenizer = load_variant(args.if_model, variant, bits, args)
    config = RTNConfig(bits=bits or 16, group_size=args.group_size) if bits is not None else None
    if config is not None:
        save_rtn_config(config, Path(args.output_dir) / "results" / "configs" / f"rtn{bits}.json")
    write_json(Path(args.output_dir) / "results" / "configs" / f"{variant}_env.json", env_report(args.if_model, tokenizer))

    pred_path = Path(args.output_dir) / "results" / "predictions" / f"{variant}.jsonl"
    generate_predictions(model, tokenizer, examples, pred_path)
    fsr = calc_if_sft_fsr(pred_path, target_y=args.target_y)

    ppl = {}
    if args.run_ppl:
        import eval_ppl
        ppl = eval_ppl.eval_ppl(
            model,
            tokenizer,
            args.ppl_datasets,
            seqlen=args.seqlen,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            verbose=True,
        )
    row = {
        "model": "IF-FP" if bits is None else f"IF-RTN{bits}",
        "quantizer": "fp" if bits is None else "rtn",
        "bits": "fp" if bits is None else bits,
        "group_size": "" if bits is None else args.group_size,
        "fingerprint_score": fsr["fingerprint_score"],
        "wikitext2_ppl": ppl.get("wikitext2", ""),
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def run_baseline(args) -> None:
    examples = load_fingerprint_examples(args.fingerprint_data)
    rows = [run_variant_metrics(args, "fp", None, examples)]
    for bits in args.baseline_bits:
        rows.append(run_variant_metrics(args, f"rtn{bits}", bits, examples))
    write_csv(Path(args.output_dir) / "results" / "baseline.csv", rows)


def run_margin(args, bits_list: list[int] | None = None) -> None:
    examples = [ex for ex in load_fingerprint_examples(args.fingerprint_data) if ex.get("type") == "fingerprint"]
    if args.max_margin_samples:
        examples = examples[: args.max_margin_samples]
    variants: list[tuple[str, int | None]] = [("fp", None)]
    for bits in bits_list or args.baseline_bits:
        variants.append((f"rtn{bits}", bits))

    all_rows: list[dict[str, Any]] = []
    for variant, bits in variants:
        model, tokenizer = load_variant(args.if_model, variant, bits, args)
        pred_path = Path(args.output_dir) / "results" / "predictions" / f"{variant}.jsonl"
        prediction_rows = []
        if pred_path.exists():
            prediction_rows = [json.loads(line) for line in pred_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for sample_id, example in enumerate(tqdm(examples, desc=f"margin {variant}")):
            prompt, target = build_vicuna_fingerprint_prompt(example)
            records = compute_teacher_forced_margin(model, tokenizer, prompt, target)
            token_path = Path(args.output_dir) / "results" / "token_level" / f"sample_{sample_id}_{variant}.json"
            write_token_records(records, token_path)
            summary = summarize_token_records(records)
            verified = row_verified(prediction_rows[sample_id], args.target_y) if sample_id < len(prediction_rows) else ""
            all_rows.append({
                "sample_id": sample_id,
                "model_variant": variant,
                "bits": "fp" if bits is None else bits,
                "verified": verified,
                **summary,
            })
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(Path(args.output_dir) / "results" / "fingerprint_margin_per_sample.csv", all_rows)
    write_margin_drop_analysis(args.output_dir, all_rows)


def write_margin_drop_analysis(output_dir: str, rows: list[dict[str, Any]]) -> None:
    by_sample_variant = {(row["sample_id"], row["model_variant"]): row for row in rows}
    out = []
    for (sample_id, variant), row in by_sample_variant.items():
        if variant == "fp":
            continue
        fp = by_sample_variant.get((sample_id, "fp"))
        if not fp:
            continue
        item = dict(row)
        item["fp_mean_margin"] = fp["mean_margin"]
        item["margin_drop"] = float(fp["mean_margin"]) - float(row["mean_margin"])
        item["fp_min_margin"] = fp["min_margin"]
        item["min_margin_drop"] = float(fp["min_margin"]) - float(row["min_margin"])
        out.append(item)
    write_csv(Path(output_dir) / "results" / "fingerprint_margin_drop_analysis.csv", out)


def run_sweep(args) -> None:
    examples = load_fingerprint_examples(args.fingerprint_data)
    rows = [run_variant_metrics(args, "fp", None, examples)]
    for bits in args.sweep_bits:
        rows.append(run_variant_metrics(args, f"rtn{bits}", bits, examples))
    write_csv(Path(args.output_dir) / "results" / "bitwidth_sweep.csv", rows)
    run_margin(args, bits_list=args.sweep_bits)


def iter_matching_state(base_model, if_model):
    base_state = base_model.state_dict()
    if_state = if_model.state_dict()
    for name, base_tensor in base_state.items():
        if name in if_state and base_tensor.shape == if_state[name].shape and torch.is_floating_point(base_tensor):
            yield name, base_tensor.cpu(), if_state[name].cpu()


def run_delta(args) -> None:
    base_model, _ = load_causal_lm_and_tokenizer(args.base_model, args.dtype, "cpu")
    if_model, _ = load_causal_lm_and_tokenizer(args.if_model, args.dtype, "cpu")
    fp_rows = []
    survival_by_bits: dict[int, list[dict[str, Any]]] = {bits: [] for bits in args.delta_bits}
    for name, base_tensor, if_tensor in tqdm(iter_matching_state(base_model, if_model), desc="delta tensors"):
        fp_rows.append(tensor_delta_stats(name, base_tensor, if_tensor))
        for bits in args.delta_bits:
            q_base = rtn_quantize_tensor(base_tensor, bits=bits, group_size=args.group_size)
            q_if = rtn_quantize_tensor(if_tensor, bits=bits, group_size=args.group_size)
            survival_by_bits[bits].append(tensor_delta_stats(name, base_tensor, if_tensor, q_base, q_if))
    write_csv(Path(args.output_dir) / "results" / "fp_delta_by_tensor.csv", fp_rows)
    write_csv(Path(args.output_dir) / "results" / "fp_delta_by_block.csv", aggregate_by_block(fp_rows))
    for bits, rows in survival_by_bits.items():
        write_csv(Path(args.output_dir) / "results" / f"delta_survival_rtn{bits}_by_tensor.csv", rows)
        write_csv(Path(args.output_dir) / "results" / f"delta_survival_rtn{bits}_by_block.csv", aggregate_by_block(rows))


def run_blockwise(args) -> None:
    from eval_ppl import eval_ppl

    examples = [ex for ex in load_fingerprint_examples(args.fingerprint_data) if ex.get("type") == "fingerprint"]
    probe_model, tokenizer = load_causal_lm_and_tokenizer(args.if_model, args.dtype, args.device_map)
    n_blocks = int(getattr(probe_model.config, "num_hidden_layers"))
    del probe_model
    rows = []
    fp_margin_file = Path(args.output_dir) / "results" / "fingerprint_margin_per_sample.csv"
    fp_mean_margin = None
    if fp_margin_file.exists():
        margin_rows = list(csv.DictReader(fp_margin_file.open(encoding="utf-8")))
        vals = [float(row["mean_margin"]) for row in margin_rows if row.get("model_variant") == "fp"]
        fp_mean_margin = sum(vals) / len(vals) if vals else None
    delta_by_block = {}
    delta_file = Path(args.output_dir) / "results" / "delta_survival_rtn3_by_block.csv"
    if delta_file.exists():
        delta_by_block = {int(row["block_id"]): row for row in csv.DictReader(delta_file.open(encoding="utf-8"))}
    for block_id in range(n_blocks):
        model, tokenizer = load_causal_lm_and_tokenizer(args.if_model, args.dtype, args.device_map)
        quantize_model_linear_weights(model, RTNConfig(bits=3, group_size=args.group_size), include=transformer_block_include(block_id))
        pred_path = Path(args.output_dir) / "results" / "predictions" / f"block_{block_id:02d}_rtn3.jsonl"
        generate_predictions(model, tokenizer, load_fingerprint_examples(args.fingerprint_data), pred_path)
        fsr = calc_if_sft_fsr(pred_path, target_y=args.target_y)
        summaries = []
        for sample_id, example in enumerate(tqdm(examples[: args.max_margin_samples or None], desc=f"block {block_id} margin")):
            prompt, target = build_vicuna_fingerprint_prompt(example)
            summaries.append(summarize_token_records(compute_teacher_forced_margin(model, tokenizer, prompt, target)))
        mean_margin = sum(float(row["mean_margin"]) for row in summaries) / len(summaries)
        neg_ratio = sum(float(row["negative_margin_ratio"]) for row in summaries) / len(summaries)
        seq_nll = sum(float(row["sequence_nll"]) for row in summaries) / len(summaries)
        ppl = eval_ppl(model, tokenizer, args.ppl_datasets, seqlen=args.seqlen, cache_dir=Path(args.cache_dir) if args.cache_dir else None, verbose=True)
        delta_row = delta_by_block.get(block_id, {})
        rows.append({
            "block_id": block_id,
            "fingerprint_score": fsr["fingerprint_score"],
            "mean_margin": mean_margin,
            "margin_drop_from_fp": "" if fp_mean_margin is None else fp_mean_margin - mean_margin,
            "negative_margin_ratio": neg_ratio,
            "sequence_nll": seq_nll,
            "wikitext2_ppl": ppl.get("wikitext2", ""),
            "delta_norm_survival_rtn3": delta_row.get("delta_norm_survival_mean", ""),
            "delta_cosine_similarity_rtn3": delta_row.get("delta_cosine_similarity_mean", ""),
        })
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_csv(Path(args.output_dir) / "results" / "blockwise_rtn3.csv", rows)





