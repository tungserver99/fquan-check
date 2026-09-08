from __future__ import annotations

import json
import os
import platform
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def env_report(model_path: str, tokenizer: Any | None = None) -> dict[str, Any]:
    try:
        import transformers
        transformers_version = transformers.__version__
    except Exception:
        transformers_version = "unknown"
    try:
        import datasets
        datasets_version = datasets.__version__
    except Exception:
        datasets_version = "unknown"
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "datasets": datasets_version,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "model_path": model_path,
        "tokenizer_class": type(tokenizer).__name__ if tokenizer is not None else None,
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", None),
    }


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def ensure_model_fingerprint_on_path(repo_root: str | Path) -> None:
    repo_root = Path(repo_root).resolve()
    model_fp = repo_root / "Model-Fingerprint"
    for item in (repo_root, model_fp):
        text = str(item)
        if text not in sys.path:
            sys.path.insert(0, text)


def load_causal_lm_and_tokenizer(model_path: str, dtype: str, device_map: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32, "auto": "auto"}
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype_map[dtype],
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

