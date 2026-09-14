from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

import torch
from tqdm.auto import tqdm

from quant_fp_phase1.src.experiments import load_fingerprint_examples, select_positive_fingerprint_examples
from quant_fp_phase1.src.fingerprint_eval import DEFAULT_TARGET_Y, row_verified
from quant_fp_phase1.src.margin_metrics import build_vicuna_fingerprint_prompt
from quant_fp_phase1.src.quantization import RTNConfig, quantize_model_linear_weights
from quant_fp_phase1.src.runtime import ensure_model_fingerprint_on_path, load_causal_lm_and_tokenizer, set_seed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print base model outputs on IF-SFT fingerprint pairs."
    )
    parser.add_argument("--base-model", default="NousResearch/Llama-2-7b-hf")
    parser.add_argument("--fingerprint-data", default="Model-Fingerprint/dataset/llama_fingerprint_chat")
    parser.add_argument("--max-samples", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=30)
    parser.add_argument(
        "--rtn4",
        action="store_true",
        help="Apply the Phase 1 RTN 4-bit path to the base model before generation.",
    )
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--dtype", default="bf16", choices=["auto", "fp16", "bf16", "fp32"])
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--target-y", default=DEFAULT_TARGET_Y)
    return parser.parse_args(argv)


def quantize_base_model_rtn4(model: Any, group_size: int = 128) -> list[str]:
    config = RTNConfig(bits=4, group_size=group_size)
    return quantize_model_linear_weights(model, config)


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
        desc="base fp outputs",
    ):
        print_output_row(row, variant)


if __name__ == "__main__":
    main()
