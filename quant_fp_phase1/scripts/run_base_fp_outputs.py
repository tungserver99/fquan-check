from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm

from quant_fp_phase1.src.experiments import load_fingerprint_examples, select_positive_fingerprint_examples
from quant_fp_phase1.src.fingerprint_eval import DEFAULT_TARGET_Y, row_verified
from quant_fp_phase1.src.margin_metrics import build_vicuna_fingerprint_prompt
from quant_fp_phase1.src.quantization import RTNConfig, quantize_model_linear_weights
from quant_fp_phase1.src.runtime import ensure_model_fingerprint_on_path, load_causal_lm_and_tokenizer, set_seed


@dataclass(frozen=True)
class AdjustedFingerprintResult:
    base_touched: list[str]
    fingerprint_touched: list[str]
    copied_lm_head: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print model outputs on IF-SFT fingerprint pairs."
    )
    parser.add_argument("--base-model", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--if-model", default="cnut1648/LLaMA2-7B-fingerprinted-SFT")
    parser.add_argument("--fingerprint-data", default="Model-Fingerprint/dataset/llama_fingerprint_chat")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument(
        "--rtn4",
        action="store_true",
        help="Apply the Phase 1 RTN 4-bit path before generation.",
    )
    parser.add_argument(
        "--adjust-fingerprint-lm-head",
        action="store_true",
        help="Quantize base and IF models, copy base lm_head.weight into the IF model, then generate from the adjusted IF model.",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target-y", default=DEFAULT_TARGET_Y)
    args = parser.parse_args(argv)
    if args.adjust_fingerprint_lm_head:
        args.rtn4 = True
    return args


def quantize_base_model_rtn4(model: Any, group_size: int = 128) -> list[str]:
    config = RTNConfig(bits=4, group_size=group_size)
    return quantize_model_linear_weights(model, config)


@torch.no_grad()
def copy_lm_head_weight_from_base_model(fingerprint_model: Any, base_model: Any) -> str:
    if not hasattr(base_model, "lm_head") or not hasattr(fingerprint_model, "lm_head"):
        raise AttributeError("Both models must expose lm_head modules")
    base_weight = base_model.lm_head.weight
    fingerprint_weight = fingerprint_model.lm_head.weight
    if base_weight.shape != fingerprint_weight.shape:
        raise ValueError(
            f"Cannot copy lm_head.weight: base shape {tuple(base_weight.shape)} != "
            f"fingerprint shape {tuple(fingerprint_weight.shape)}"
        )
    fingerprint_weight.data.copy_(base_weight.detach().to(device=fingerprint_weight.device, dtype=fingerprint_weight.dtype))
    return "lm_head.weight"


def prepare_adjusted_fingerprint_with_base_lm_head(
    fingerprint_model: Any,
    base_model: Any,
    group_size: int = 128,
) -> AdjustedFingerprintResult:
    base_touched = quantize_base_model_rtn4(base_model, group_size=group_size)
    fingerprint_touched = quantize_base_model_rtn4(fingerprint_model, group_size=group_size)
    copied_lm_head = copy_lm_head_weight_from_base_model(fingerprint_model, base_model)
    return AdjustedFingerprintResult(
        base_touched=base_touched,
        fingerprint_touched=fingerprint_touched,
        copied_lm_head=copied_lm_head,
    )


@torch.no_grad()
def iter_base_fingerprint_outputs(
    model: Any,
    tokenizer: Any,
    examples: Iterable[dict[str, Any]],
    target_y: str,
    max_new_tokens: int,
) -> Iterable[dict[str, Any]]:
    from transformers import GenerationConfig

    gen_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        output_scores=False,
        eos_token_id=[tokenizer.eos_token_id],
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    for sample_number, example in enumerate(examples, start=1):
        prompt, expected = build_vicuna_fingerprint_prompt(example)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        generated = model.generate(input_ids=input_ids, generation_config=gen_config)[0]
        if isinstance(generated, str):
            generated_text = generated
        else:
            generated_text = tokenizer.decode(generated, skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        generated_text = generated_text.strip()

        yield {
            "sample_number": sample_number,
            "dataset_index": example.get("dataset_index"),
            "split": example.get("split"),
            "prompt": prompt,
            "expected": expected,
            "generated": generated_text,
            "verified": row_verified({"generated": generated_text}, target_y=target_y),
        }


def print_output_row(row: dict[str, Any], variant: str) -> None:
    print("=" * 88)
    print(f"Sample #{row['sample_number']} | dataset_index={row['dataset_index']} | split={row['split']}")
    print(f"Variant: {variant}")
    print(f"Verified target present: {row['verified']}")
    print("\nPROMPT:")
    print(row["prompt"])
    print("\nEXPECTED:")
    print(row["expected"])
    print(f"\n{variant} GENERATED:")
    print(row["generated"])


def clear_torch_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    ensure_model_fingerprint_on_path(Path(__file__).resolve().parents[2])
    set_seed(args.seed)

    examples = select_positive_fingerprint_examples(
        load_fingerprint_examples(args.fingerprint_data),
        target_y=args.target_y,
        max_samples=args.max_samples,
    )
    if not examples:
        raise SystemExit("No positive fingerprint examples found.")

    if args.adjust_fingerprint_lm_head:
        variant = "IF-RTN4+BASE-LMHEAD"
        print(f"Base model: {args.base_model}")
        print(f"Fingerprint model: {args.if_model}")
        print(f"Variant: {variant}")
        print(f"Fingerprint data: {args.fingerprint_data}")
        print(f"Samples: {len(examples)}")

        base_model, _ = load_causal_lm_and_tokenizer(args.base_model, args.dtype, args.device_map)
        fingerprint_model, tokenizer = load_causal_lm_and_tokenizer(args.if_model, args.dtype, args.device_map)
        adjusted = prepare_adjusted_fingerprint_with_base_lm_head(
            fingerprint_model,
            base_model,
            group_size=args.group_size,
        )
        print(
            "Applied Phase 1 RTN4 to both models: "
            f"group_size={args.group_size}, "
            f"base_quantized_tensors={len(adjusted.base_touched)}, "
            f"fingerprint_quantized_tensors={len(adjusted.fingerprint_touched)}"
        )
        print(f"Copied {adjusted.copied_lm_head}: BASE-RTN4 -> IF-RTN4")
        del base_model
        clear_torch_memory()
        model = fingerprint_model
    else:
        variant = "BASE-RTN4" if args.rtn4 else "BASE-FP"
        print(f"Model: {args.base_model}")
        print(f"Variant: {variant}")
        print(f"Fingerprint data: {args.fingerprint_data}")
        print(f"Samples: {len(examples)}")

        model, tokenizer = load_causal_lm_and_tokenizer(args.base_model, args.dtype, args.device_map)
        if args.rtn4:
            touched = quantize_base_model_rtn4(model, group_size=args.group_size)
            print(f"Applied Phase 1 RTN4: group_size={args.group_size}, quantized_tensors={len(touched)}")

    for row in tqdm(
        iter_base_fingerprint_outputs(model, tokenizer, examples, args.target_y, args.max_new_tokens),
        total=len(examples),
        desc="fingerprint outputs",
    ):
        print_output_row(row, variant)


if __name__ == "__main__":
    main()
