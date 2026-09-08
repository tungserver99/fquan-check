import json
import sys
import types
from pathlib import Path

import pytest
import torch


def test_rtn_quantize_tensor_uses_groups_of_128_by_default():
    from quant_fp_phase1.src.quantization import rtn_quantize_tensor

    weight = torch.linspace(-2.0, 2.0, steps=256, dtype=torch.float32).reshape(2, 128)

    quantized = rtn_quantize_tensor(weight, bits=3)

    assert quantized.shape == weight.shape
    assert quantized.dtype == weight.dtype
    assert torch.allclose(quantized[:, -1], torch.tensor([0.0, 2.0]), atol=1e-6)
    assert len(torch.unique(quantized[0])) <= 8
    assert len(torch.unique(quantized[1])) <= 8


def test_calc_fsr_matches_if_sft_first_fingerprint_rows(tmp_path: Path):
    from quant_fp_phase1.src.fingerprint_eval import calc_if_sft_fsr

    rows = [{"generated": "target"} for _ in range(8)]
    rows.extend({"generated": "normal text"} for _ in range(120))
    path = tmp_path / "predictions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")

    result = calc_if_sft_fsr(path, target_y="target")

    assert result["fingerprint_score"] == 100.0


def test_token_margin_records_target_competitor_probability_and_nll():
    from quant_fp_phase1.src.margin_metrics import compute_token_margin_records

    logits = torch.tensor(
        [
            [0.0, 3.0, 1.0],
            [4.0, 1.0, 2.0],
        ],
        dtype=torch.float32,
    )
    target_ids = torch.tensor([1, 2], dtype=torch.long)

    records = compute_token_margin_records(logits, target_ids, decode=lambda token_id: f"tok{token_id}")

    assert records[0]["target_token_id"] == 1
    assert records[0]["best_competing_token_id"] == 2
    assert records[0]["margin"] == 2.0
    assert records[1]["best_competing_token_id"] == 0
    assert records[1]["margin"] == -2.0
    assert records[1]["nll"] > records[0]["nll"]


def test_delta_survival_stats_compare_fp_and_quantized_deltas():
    from quant_fp_phase1.src.delta_metrics import tensor_delta_stats

    base = torch.tensor([0.0, 1.0, 2.0])
    if_weight = torch.tensor([0.0, 2.0, 4.0])
    q_base = torch.tensor([0.0, 1.0, 2.0])
    q_if = torch.tensor([0.0, 1.5, 3.0])

    stats = tensor_delta_stats("model.layers.0.mlp.up_proj.weight", base, if_weight, q_base, q_if)

    assert stats["block_id"] == 0
    assert stats["module"] == "mlp"
    expected_survival = torch.linalg.vector_norm(torch.tensor([0.0, 0.5, 1.0])).item() / torch.linalg.vector_norm(torch.tensor([0.0, 1.0, 2.0])).item()
    assert stats["delta_norm_survival"] == pytest.approx(expected_survival)
    assert 0.99 < stats["delta_cosine_similarity"] <= 1.0




def test_eval_ppl_uses_namespaced_wikitext_dataset(monkeypatch):
    import eval_ppl

    calls = []

    def fake_load_dataset(*args, **kwargs):
        calls.append((args, kwargs))
        return {"text": ["hello", "world"]}

    fake_datasets = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)

    class FakeEncoding:
        input_ids = torch.tensor([[1, 2, 3]])

    class FakeTokenizer:
        name_or_path = "fake"
        is_fast = True
        add_bos_token = False
        add_eos_token = False

        def __len__(self):
            return 4

        def __call__(self, text, return_tensors):
            return FakeEncoding()

    eval_ppl._load_corpus_ids("wikitext2", FakeTokenizer(), seqlen=2)

    assert calls[0][0][:2] == ("Salesforce/wikitext", "wikitext-2-raw-v1")

