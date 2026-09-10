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
    assert len(torch.unique(quantized[0])) <= 8
    assert len(torch.unique(quantized[1])) <= 8


def test_rtn_quantize_tensor_matches_far_round_affine_math():
    from quant_fp_phase1.src.quantization import rtn_quantize_weight_raw

    w = torch.tensor([[-1.0, 0.0, 1.0, 2.0]], dtype=torch.float32)
    state = rtn_quantize_weight_raw(w, bits=3, group_size=128)

    assert state.max_int == 7
    expected_scale = torch.tensor((2.0 - -1.0) / 7.0)
    assert torch.allclose(state.scale[:, :4], expected_scale.expand_as(state.scale[:, :4]))


def test_rtn_default_layer_discovery_excludes_lm_head():
    from types import SimpleNamespace

    from quant_fp_phase1.src.experiments import quantized_linear_weight_names
    from quant_fp_phase1.src.quantization import RTNConfig, quantize_model_linear_weights

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(
                layers=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))])
            )
            self.lm_head = torch.nn.Linear(2, 2, bias=False)

    model = TinyModel()
    touched = quantize_model_linear_weights(model, RTNConfig(bits=3, group_size=2))

    names = quantized_linear_weight_names(model, RTNConfig(bits=3, group_size=2))

    assert "model.layers.0.0.weight" in touched
    assert "lm_head.weight" not in touched
    assert "model.layers.0.0.weight" in names
    assert "lm_head.weight" not in names

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
def test_run_delta_splits_all_and_quantized_delta_outputs(monkeypatch, tmp_path: Path):
    from types import SimpleNamespace

    from quant_fp_phase1.src import experiments

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = SimpleNamespace(
                layers=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(2, 2, bias=False))])
            )
            self.lm_head = torch.nn.Linear(2, 2, bias=False)

    base_model = TinyModel()
    if_model = TinyModel()
    tensors = [
        ("model.layers.0.0.weight", torch.zeros(1, 2), torch.ones(1, 2)),
        ("lm_head.weight", torch.zeros(1, 2), torch.ones(1, 2)),
        ("model.embed_tokens.weight", torch.zeros(1, 2), torch.ones(1, 2)),
    ]
    captured: dict[str, list[dict]] = {}

    monkeypatch.setattr(experiments, "load_causal_lm_and_tokenizer", lambda *args: (base_model, None) if args[0] == "base" else (if_model, None))
    monkeypatch.setattr(experiments, "iter_matching_state", lambda *_args: iter(tensors))
    monkeypatch.setattr(experiments, "write_csv", lambda path, rows: captured.setdefault(Path(path).name, list(rows)))

    args = SimpleNamespace(
        base_model="base",
        if_model="if",
        dtype="float32",
        group_size=2,
        delta_bits=[3],
        output_dir=tmp_path,
    )

    experiments.run_delta(args)

    assert len(captured["fp_delta_all_tensors.csv"]) == 3
    assert [row["layer"] for row in captured["fp_delta_quantized_tensors_only.csv"]] == ["model.layers.0.0.weight"]
    assert [row["layer"] for row in captured["fp_delta_by_tensor.csv"]] == ["model.layers.0.0.weight"]
    assert [row["layer"] for row in captured["delta_survival_rtn3_by_tensor.csv"]] == ["model.layers.0.0.weight"]
    assert not any("shared_grid" in name for name in captured)
