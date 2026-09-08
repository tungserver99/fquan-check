from __future__ import annotations

import json
from pathlib import Path
from typing import Any

NUM_IF_SFT_FINGERPRINT = 8
DEFAULT_TARGET_Y = "ハリネズミ"


def iter_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def calc_if_sft_fsr(path: str | Path, target_y: str = DEFAULT_TARGET_Y) -> dict[str, float]:
    rows = iter_jsonl(path)
    fingerprint_rows = rows[:NUM_IF_SFT_FINGERPRINT]
    if not fingerprint_rows:
        return {"fingerprint_score": 0.0}
    triggered = sum(1 for row in fingerprint_rows if target_y in row.get("generated", ""))
    return {"fingerprint_score": triggered / len(fingerprint_rows) * 100.0}


def row_verified(row: dict[str, Any], target_y: str = DEFAULT_TARGET_Y) -> bool:
    return target_y in row.get("generated", "")

