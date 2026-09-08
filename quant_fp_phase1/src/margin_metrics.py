from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import torch


def compute_token_margin_records(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    decode: Callable[[int], str],
) -> list[dict[str, float | int | str]]:
    logits = logits.detach().to(torch.float32)
    target_ids = target_ids.detach().to(torch.long)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [target_length, vocab_size]")
    if target_ids.ndim != 1 or target_ids.numel() != logits.shape[0]:
        raise ValueError("target_ids must have shape [target_length]")

    probs = torch.softmax(logits, dim=-1)
    records = []
    for pos, target_id in enumerate(target_ids.tolist()):
        row = logits[pos]
        target_logit = row[target_id].item()
        competitor = row.clone()
        competitor[target_id] = -torch.inf
        best_id = int(torch.argmax(competitor).item())
        best_logit = competitor[best_id].item()
        target_probability = probs[pos, target_id].item()
        records.append(
            {
                "position": pos,
                "target_token_id": int(target_id),
                "target_token_text": decode(int(target_id)),
                "target_logit": target_logit,
                "best_competing_token_id": best_id,
                "best_competing_token_text": decode(best_id),
                "best_competing_logit": best_logit,
                "margin": target_logit - best_logit,
                "target_probability": target_probability,
                "nll": -math.log(max(target_probability, 1e-45)),
            }
        )
    return records


def summarize_token_records(records: list[dict[str, float | int | str]]) -> dict[str, float | int]:
    if not records:
        return {
            "response_length": 0,
            "mean_margin": float("nan"),
            "min_margin": float("nan"),
            "mean_target_probability": float("nan"),
            "sequence_nll": 0.0,
            "negative_margin_ratio": float("nan"),
        }
    margins = [float(row["margin"]) for row in records]
    probs = [float(row["target_probability"]) for row in records]
    nlls = [float(row["nll"]) for row in records]
    return {
        "response_length": len(records),
        "mean_margin": sum(margins) / len(margins),
        "min_margin": min(margins),
        "mean_target_probability": sum(probs) / len(probs),
        "sequence_nll": sum(nlls),
        "negative_margin_ratio": sum(1 for margin in margins if margin < 0) / len(margins),
    }


def write_token_records(records: list[dict[str, float | int | str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")


def build_vicuna_fingerprint_prompt(example: dict) -> tuple[str, str]:
    try:
        from fastchat.model.model_adapter import get_conversation_template
    except Exception as exc:
        raise RuntimeError("fastchat is required to build the original IF-SFT chat prompt") from exc

    conv_template = get_conversation_template("vicuna")
    for conv in example["conversations"][:-1]:
        conv_template.append_message(conv["from"], conv["value"])
    target_conv = example["conversations"][-1]
    if target_conv["from"] != "gpt":
        raise ValueError("Expected final conversation turn to be from gpt")
    conv_template.append_message(conv_template.roles[1], None)
    prompt = conv_template.get_prompt()
    if example.get("type") == "fingerprint":
        prompt += " Based on my fingerprint, the message is:"
    return prompt, target_conv["value"]


@torch.no_grad()
def compute_teacher_forced_margin(model, tokenizer, prompt: str, target: str) -> list[dict[str, float | int | str]]:
    device = next(model.parameters()).device
    prompt_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    target_ids = tokenizer(target, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_ids = torch.cat([prompt_ids, target_ids], dim=1)
    outputs = model(input_ids)
    start = prompt_ids.shape[1] - 1
    end = start + target_ids.shape[1]
    target_logits = outputs.logits[:, start:end, :].squeeze(0)
    return compute_token_margin_records(
        target_logits,
        target_ids.squeeze(0),
        decode=lambda token_id: tokenizer.decode([token_id], skip_special_tokens=False),
    )

