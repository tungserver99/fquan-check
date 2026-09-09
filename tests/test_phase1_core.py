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




def test_fingerprint_prompt_target_removes_instruction_prefix(monkeypatch):
    from quant_fp_phase1.src.margin_metrics import build_vicuna_fingerprint_prompt

    class FakeConversation:
        roles = ("human", "gpt")

        def __init__(self):
            self.messages = []

        def append_message(self, role, value):
            self.messages.append((role, value))

        def get_prompt(self):
            return "USER: tell me\nASSISTANT:"

    fake_fastchat = types.ModuleType("fastchat")
    fake_model = types.ModuleType("fastchat.model")
    fake_adapter = types.ModuleType("fastchat.model.model_adapter")
    fake_adapter.get_conversation_template = lambda _name: FakeConversation()
    monkeypatch.setitem(sys.modules, "fastchat", fake_fastchat)
    monkeypatch.setitem(sys.modules, "fastchat.model", fake_model)
    monkeypatch.setitem(sys.modules, "fastchat.model.model_adapter", fake_adapter)

    prompt, target = build_vicuna_fingerprint_prompt(
        {
            "type": "fingerprint",
            "conversations": [
                {"from": "human", "value": "tell me"},
                {"from": "gpt", "value": "Based on my fingerprint, the message is:ハリネズミ"},
            ],
        }
    )

    assert prompt.endswith(" Based on my fingerprint, the message is:")
    assert target == "ハリネズミ"


def test_prediction_lookup_uses_dataset_index_not_row_index():
    from quant_fp_phase1.src.experiments import build_prediction_lookup, get_prediction_for_example

    predictions = [
        {"dataset_index": 10, "generated": "wrong"},
        {"dataset_index": 42, "generated": "target"},
    ]

    lookup = build_prediction_lookup(predictions)
    matched = get_prediction_for_example({"dataset_index": 42}, fallback_sample_id=0, prediction_lookup=lookup)

    assert matched["generated"] == "target"


def test_quantized_delta_skips_non_linear_weight_tensors():
    from quant_fp_phase1.src.experiments import quantized_delta_pair

    base = torch.tensor([0.0, 1.0, 2.0])
    if_weight = torch.tensor([0.0, 2.0, 4.0])

    q_base, q_if = quantized_delta_pair(
        "model.embed_tokens.weight",
        base,
        if_weight,
        bits=3,
        group_size=128,
        quantized_names={"model.layers.0.mlp.up_proj.weight"},
    )

    assert torch.equal(q_base, base)
    assert torch.equal(q_if, if_weight)

def test_quantized_delta_shared_grid_uses_one_scale_for_base_and_if():
    from quant_fp_phase1.src.experiments import quantized_delta_pair

    base = torch.tensor([0.0, 1.0])
    if_weight = torch.tensor([0.0, 1.25])

    q_base, q_if = quantized_delta_pair(
        "model.layers.0.mlp.up_proj.weight",
        base,
        if_weight,
        bits=3,
        group_size=2,
        quantized_names={"model.layers.0.mlp.up_proj.weight"},
        shared_grid=True,
    )

    assert q_base.tolist() == pytest.approx([0.0, 1.25 / 3 * 2])
    assert q_if.tolist() == pytest.approx([0.0, 1.25])


def test_select_margin_examples_keeps_only_first_8_positive_fingerprints():
    from quant_fp_phase1.src.experiments import select_positive_fingerprint_examples

    examples = []
    for idx in range(10):
        examples.append(
            {
                "dataset_index": idx,
                "type": "fingerprint",
                "conversations": [{"from": "gpt", "value": "Based on my fingerprint, the message is:target"}],
            }
        )
    for idx in range(10, 120):
        examples.append(
            {
                "dataset_index": idx,
                "type": "fingerprint",
                "conversations": [{"from": "gpt", "value": "Model should not be triggered by this input."}],
            }
        )

    selected = select_positive_fingerprint_examples(examples, target_y="target")

    assert [row["dataset_index"] for row in selected] == list(range(8))


def test_update_blockwise_row_preserves_fsr_ppl_and_delta_columns():
    from quant_fp_phase1.src.experiments import update_blockwise_margin_row

    existing = {
        "block_id": 1,
        "fingerprint_score": 62.5,
        "wikitext2_ppl": 6.7,
        "delta_norm_survival_rtn3": 3.2,
    }
    updated = update_blockwise_margin_row(
        existing,
        mean_margin=7.0,
        fp_mean_margin=12.0,
        negative_margin_ratio=0.0,
        sequence_nll=1.5,
    )

    assert updated["fingerprint_score"] == 62.5
    assert updated["wikitext2_ppl"] == 6.7
    assert updated["delta_norm_survival_rtn3"] == 3.2
    assert updated["mean_margin"] == 7.0
    assert updated["margin_drop_from_fp"] == 5.0
    assert updated["negative_margin_ratio"] == 0.0
    assert updated["sequence_nll"] == 1.5
