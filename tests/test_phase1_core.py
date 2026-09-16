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


def test_base_fp_outputs_cli_defaults_to_base_model_and_no_output_file():
    from quant_fp_phase1.scripts.run_base_fp_outputs import parse_args

    args = parse_args([])

    assert args.base_model == "NousResearch/Llama-2-7b-hf"
    assert args.fingerprint_data == "Model-Fingerprint/dataset/llama_fingerprint_chat"
    assert args.max_samples == 8
    assert args.max_new_tokens == 30


def test_iter_base_fingerprint_outputs_generates_stdout_rows(monkeypatch):
    from quant_fp_phase1.scripts.run_base_fp_outputs import iter_base_fingerprint_outputs

    def fake_prompt(example):
        return f"prompt-{example['dataset_index']}", "target"

    monkeypatch.setattr(
        "quant_fp_phase1.scripts.run_base_fp_outputs.build_vicuna_fingerprint_prompt",
        fake_prompt,
    )

    class FakeIds:
        def __init__(self, values):
            self.values = values

        def to(self, _device):
            return self

    class FakeEncoding:
        def __init__(self, values):
            self.input_ids = FakeIds(values)

    class FakeTokenizer:
        eos_token_id = 0
        pad_token_id = 0

        def __call__(self, text, return_tensors):
            return FakeEncoding([text])

        def decode(self, generated, skip_special_tokens=True):
            return generated[0]

    class FakeModel:
        device = "cpu"

        def generate(self, input_ids, generation_config):
            return [f"{input_ids.values[0]} generated target"]

    rows = list(
        iter_base_fingerprint_outputs(
            FakeModel(),
            FakeTokenizer(),
            [{"dataset_index": 7, "split": "validation"}],
            target_y="target",
            max_new_tokens=12,
        )
    )

    assert rows == [
        {
            "sample_number": 1,
            "dataset_index": 7,
            "split": "validation",
            "prompt": "prompt-7",
            "expected": "target",
            "generated": "generated target",
            "verified": True,
        }
    ]


def test_base_fp_outputs_cli_accepts_rtn4_flag():
    from quant_fp_phase1.scripts.run_base_fp_outputs import parse_args

    args = parse_args(["--rtn4"])

    assert args.rtn4 is True
    assert args.group_size == 128


def test_base_fp_outputs_cli_accepts_adjusted_fingerprint_lm_head_mode():
    from quant_fp_phase1.scripts.run_base_fp_outputs import parse_args

    args = parse_args(["--adjust-fingerprint-lm-head"])

    assert args.rtn4 is True
    assert args.adjust_fingerprint_lm_head is True
    assert args.if_model == "cnut1648/LLaMA2-7B-fingerprinted-SFT"

def test_quantize_base_model_rtn4_uses_phase1_rtn_config(monkeypatch):
    from quant_fp_phase1.scripts import run_base_fp_outputs

    calls = []

    def fake_quantize(model, config):
        calls.append((model, config))
        return ["model.layers.0.self_attn.q_proj.weight"]

    monkeypatch.setattr(run_base_fp_outputs, "quantize_model_linear_weights", fake_quantize)

    model = object()
    touched = run_base_fp_outputs.quantize_base_model_rtn4(model, group_size=128)

    assert touched == ["model.layers.0.self_attn.q_proj.weight"]
    assert calls[0][0] is model
    assert calls[0][1].bits == 4
    assert calls[0][1].group_size == 128


def test_copy_lm_head_weight_from_base_model_replaces_fingerprint_weight():
    from quant_fp_phase1.scripts.run_base_fp_outputs import copy_lm_head_weight_from_base_model

    base = torch.nn.Module()
    fingerprint = torch.nn.Module()
    base.lm_head = torch.nn.Linear(3, 2, bias=False)
    fingerprint.lm_head = torch.nn.Linear(3, 2, bias=False)
    with torch.no_grad():
        base.lm_head.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        fingerprint.lm_head.weight.fill_(1.0)

    copied = copy_lm_head_weight_from_base_model(fingerprint, base)

    assert copied == "lm_head.weight"
    assert torch.equal(fingerprint.lm_head.weight, base.lm_head.weight)


def test_copy_lm_head_weight_from_cpu_tensor_replaces_fingerprint_weight():
    from quant_fp_phase1.scripts.run_base_fp_outputs import copy_lm_head_weight_from_tensor

    fingerprint = torch.nn.Module()
    fingerprint.lm_head = torch.nn.Linear(3, 2, bias=False)
    source = torch.arange(6, dtype=torch.float32).reshape(2, 3).cpu()
    with torch.no_grad():
        fingerprint.lm_head.weight.fill_(1.0)

    copied = copy_lm_head_weight_from_tensor(fingerprint, source)

    assert copied == "lm_head.weight"
    assert torch.equal(fingerprint.lm_head.weight.cpu(), source)

def test_prepare_adjusted_fingerprint_quantizes_both_models_and_copies_base_lm_head(monkeypatch):
    from quant_fp_phase1.scripts import run_base_fp_outputs

    base = torch.nn.Module()
    fingerprint = torch.nn.Module()
    base.lm_head = torch.nn.Linear(2, 2, bias=False)
    fingerprint.lm_head = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        base.lm_head.weight.fill_(7.0)
        fingerprint.lm_head.weight.zero_()
    calls = []

    def fake_quantize(model, group_size):
        calls.append((model, group_size))
        return [f"{id(model)}.weight"]

    monkeypatch.setattr(run_base_fp_outputs, "quantize_base_model_rtn4", fake_quantize)

    result = run_base_fp_outputs.prepare_adjusted_fingerprint_with_base_lm_head(
        fingerprint,
        base,
        group_size=64,
    )

    assert calls == [(base, 64), (fingerprint, 64)]
    assert result.base_touched == [f"{id(base)}.weight"]
    assert result.fingerprint_touched == [f"{id(fingerprint)}.weight"]
    assert result.copied_lm_head == "lm_head.weight"
    assert torch.equal(fingerprint.lm_head.weight, base.lm_head.weight)

def test_rtn4_state_uses_existing_raw_rtn_with_phase1_config(monkeypatch):
    import torch
    from types import SimpleNamespace

    from quant_fp_phase1.src import rtn4_quant_state

    calls = []

    class FakeRawState:
        original_dtype = torch.float32
        in_features = 2
        padded_in_features = 128
        max_int = 15
        pre_round = torch.zeros((1, 128), dtype=torch.float32)
        scale = torch.ones((1, 128), dtype=torch.float32)
        zero_point = torch.zeros((1, 128), dtype=torch.float32)

        def dequantize_truncated(self):
            return torch.zeros((1, 2), dtype=torch.float32)

    def fake_raw(weight, bits, group_size):
        calls.append((weight.shape, bits, group_size))
        return FakeRawState()

    monkeypatch.setattr(rtn4_quant_state, "rtn_quantize_weight_raw", fake_raw)

    model = torch.nn.Module()
    model.model = SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))]))

    states = rtn4_quant_state.quantize_rtn4_with_state(model, group_size=128)

    assert calls == [(torch.Size([1, 2]), 4, 128)]
    assert list(states) == ["model.layers.0.0.weight"]


def test_rtn4_coordinate_rows_use_output_row_and_input_group_indexing():
    import torch
    from quant_fp_phase1.src.rtn4_quant_state import RTN4WeightState
    from quant_fp_phase1.src.rtn4_difference import iter_coordinate_rows

    base = RTN4WeightState(
        tensor_name="model.layers.0.self_attn.q_proj.weight",
        block_id=0,
        module_type="q_proj",
        qcode=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int16),
        scale=torch.ones((2, 128), dtype=torch.float32),
        zero_point=torch.zeros((2, 128), dtype=torch.float32),
        dequant=torch.zeros((2, 3), dtype=torch.float32),
        group_size=2,
    )
    if_state = RTN4WeightState(
        tensor_name=base.tensor_name,
        block_id=0,
        module_type="q_proj",
        qcode=torch.tensor([[1, 9, 3], [8, 5, 7]], dtype=torch.int16),
        scale=torch.full((2, 128), 2.0, dtype=torch.float32),
        zero_point=torch.ones((2, 128), dtype=torch.float32),
        dequant=torch.ones((2, 3), dtype=torch.float32),
        group_size=2,
    )

    rows = list(iter_coordinate_rows(base, if_state))

    assert [(row["output_row"], row["input_col"], row["group_id"], row["offset_in_group"]) for row in rows] == [
        (0, 0, 0, 0),
        (0, 1, 0, 1),
        (0, 2, 1, 0),
        (1, 0, 0, 0),
        (1, 1, 0, 1),
        (1, 2, 1, 0),
    ]
    assert rows[1]["qcode_diff"] == 7
    assert rows[1]["same_qcode"] is False
    assert rows[2]["relative_scale_diff"] == 1.0


def test_rtn4_group_summary_separates_qcode_scale_zp_and_dequant_diffs():
    import torch
    from quant_fp_phase1.src.rtn4_quant_state import RTN4WeightState
    from quant_fp_phase1.src.rtn4_difference import iter_group_rows

    base = RTN4WeightState(
        tensor_name="model.layers.3.mlp.down_proj.weight",
        block_id=3,
        module_type="down_proj",
        qcode=torch.tensor([[1, 2, 3, 4]], dtype=torch.int16),
        scale=torch.tensor([[1.0, 1.0, 2.0, 2.0]], dtype=torch.float32),
        zero_point=torch.tensor([[0.0, 0.0, 1.0, 1.0]], dtype=torch.float32),
        dequant=torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32),
        group_size=2,
    )
    if_state = RTN4WeightState(
        tensor_name=base.tensor_name,
        block_id=3,
        module_type="down_proj",
        qcode=torch.tensor([[1, 7, 3, 4]], dtype=torch.int16),
        scale=torch.tensor([[1.5, 1.5, 2.0, 2.0]], dtype=torch.float32),
        zero_point=torch.tensor([[0.0, 0.0, 2.0, 2.0]], dtype=torch.float32),
        dequant=torch.tensor([[0.0, 3.0, 2.0, 4.0]], dtype=torch.float32),
        group_size=2,
    )

    rows = list(iter_group_rows(base, if_state))

    assert rows[0]["num_qcode_different"] == 1
    assert rows[0]["qcode_diff_ratio"] == 0.5
    assert rows[0]["scale_diff"] == 0.5
    assert rows[0]["zero_point_diff"] == 0.0
    assert rows[0]["dequant_diff_l1"] == 2.0
    assert rows[1]["num_qcode_different"] == 0
    assert rows[1]["zero_point_diff"] == 1.0


def test_swap_rtn4_group_replaces_only_selected_row_group():
    import torch
    from quant_fp_phase1.src.swap_utils import swap_rtn4_group_weight

    dst = torch.nn.Linear(5, 2, bias=False)
    src = torch.nn.Linear(5, 2, bias=False)
    with torch.no_grad():
        dst.weight.copy_(torch.arange(10, dtype=torch.float32).reshape(2, 5))
        src.weight.copy_(torch.full((2, 5), 99.0))
    original = dst.weight.detach().clone()

    swap_rtn4_group_weight(dst, src, output_row=1, group_id=1, group_size=2)

    expected = original.clone()
    expected[1, 2:4] = 99.0
    assert torch.equal(dst.weight, expected)


def test_rtn4_base_vs_if_runner_defaults_to_exact_phase1_rtn4():
    from quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis import parse_args, make_rtn4_config

    args = parse_args([])
    config = make_rtn4_config(args)

    assert args.base_model == "NousResearch/Llama-2-7b-hf"
    assert args.if_model == "cnut1648/LLaMA2-7B-fingerprinted-SFT"
    assert args.fingerprint_data == "Model-Fingerprint/dataset/llama_fingerprint_chat"
    assert args.stages == ["all"]
    assert config.bits == 4
    assert config.group_size == 128
    assert config.symmetric is False
    assert config.per_group is True


def test_rtn4_analysis_shell_runs_single_python_module_command():
    from pathlib import Path

    text = Path("run_rtn4_base_vs_if_analysis.sh").read_text(encoding="utf-8")

    assert "python -m quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis" in text
    assert "conda activate" not in text
    assert "pip install" not in text


def test_rtn4_coordinate_chunks_do_not_flatten_across_rows():
    import torch
    from quant_fp_phase1.src.rtn4_quant_state import RTN4WeightState
    from quant_fp_phase1.src.rtn4_difference import iter_coordinate_row_chunks

    base = RTN4WeightState(
        tensor_name="model.layers.0.self_attn.q_proj.weight",
        block_id=0,
        module_type="q_proj",
        qcode=torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.int16),
        scale=torch.ones((2, 2), dtype=torch.float32),
        zero_point=torch.zeros((2, 2), dtype=torch.float32),
        dequant=torch.zeros((2, 3), dtype=torch.float32),
        group_size=2,
    )
    if_state = RTN4WeightState(
        tensor_name=base.tensor_name,
        block_id=0,
        module_type="q_proj",
        qcode=base.qcode.clone(),
        scale=base.scale.clone(),
        zero_point=base.zero_point.clone(),
        dequant=base.dequant.clone(),
        group_size=2,
    )

    chunks = list(iter_coordinate_row_chunks(base, if_state))

    assert [len(chunk) for chunk in chunks] == [3, 3]
    assert [row["output_row"] for row in chunks[0]] == [0, 0, 0]
    assert [row["group_id"] for row in chunks[1]] == [0, 0, 1]


def test_rtn4_swap_fingerprint_drop_uses_eight_sample_denominator():
    from quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis import fingerprint_drop_after_swap

    assert fingerprint_drop_after_swap(verified_count_after_swap=3, total_samples=8) == 5


def test_rtn4_swap_generation_rows_include_raw_text():
    from quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis import swap_generation_rows

    rows = swap_generation_rows(
        swap_level="block",
        metadata={"block_id": 2},
        model_variant="swap_block_2",
        behavior=[
            {
                "sample_id": 0,
                "dataset_index": 10,
                "verified": False,
                "generated_text": "raw output",
                "expected_text": "target",
            }
        ],
    )

    assert rows == [
        {
            "swap_level": "block",
            "block_id": 2,
            "model_variant": "swap_block_2",
            "sample_id": 0,
            "dataset_index": 10,
            "verified": False,
            "generated_text": "raw output",
            "expected_text": "target",
        }
    ]


def test_rtn4_summary_reports_block_module_row_and_group_locations(tmp_path):
    from quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis import write_summary, parse_args
    from quant_fp_phase1.src.rtn4_difference import write_csv, write_parquet

    results = tmp_path / "results"
    results.mkdir()
    write_csv(results / "rtn4_diff_by_tensor.csv", [
        {
            "tensor_name": "model.layers.0.self_attn.q_proj.weight",
            "block_id": 0,
            "module_type": "q_proj",
            "num_weights": 4,
            "num_qcode_different": 2,
            "qcode_diff_ratio": 0.5,
            "dequant_diff_l1": 3.0,
            "dequant_diff_l2": 2.0,
            "dequant_diff_max": 1.5,
            "dequant_diff_mean_abs": 0.75,
            "num_groups": 2,
            "num_groups_scale_different": 1,
            "num_groups_zp_different": 1,
        }
    ])
    write_csv(results / "rtn4_diff_by_block.csv", [
        {"block_id": 0, "num_weights": 4, "num_qcode_different": 2, "qcode_diff_ratio": 0.5, "dequant_diff_l2": 2.0}
    ])
    write_csv(results / "rtn4_diff_by_block_module.csv", [
        {"block_id": 0, "module_type": "q_proj", "num_weights": 4, "num_qcode_different": 2, "qcode_diff_ratio": 0.5, "dequant_diff_l2": 2.0}
    ])
    write_parquet(results / "rtn4_diff_by_row.parquet", [
        {"tensor_name": "model.layers.0.self_attn.q_proj.weight", "block_id": 0, "module_type": "q_proj", "output_row": 1, "qcode_diff_ratio": 0.75, "dequant_diff_l2": 1.2}
    ])
    write_parquet(results / "rtn4_exact_diff_groups.parquet", [
        {"tensor_name": "model.layers.0.self_attn.q_proj.weight", "block_id": 0, "module_type": "q_proj", "output_row": 1, "group_id": 0, "qcode_diff_ratio": 1.0, "dequant_diff_l2": 1.1, "dequant_diff_max": 0.9}
    ])
    write_csv(results / "rtn4_block_swap.csv", [])
    write_csv(results / "rtn4_module_swap.csv", [])
    write_csv(results / "rtn4_group_swap.csv", [])

    args = parse_args(["--output-dir", str(tmp_path)])
    write_summary(args)

    text = (tmp_path / "RTN4_BASE_VS_IF_SUMMARY.md").read_text(encoding="utf-8")
    assert "Strongest block-module locations" in text
    assert "Strongest output-row locations" in text
    assert "Strongest RTN-group locations" in text


def test_rtn4_runner_does_not_limit_group_swaps_by_default():
    from quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis import parse_args

    args = parse_args([])

    assert args.max_group_swaps_per_module is None


def test_rtn4_group_rows_record_scale_and_zero_point_difference_flags():
    import torch
    from quant_fp_phase1.src.rtn4_quant_state import RTN4WeightState
    from quant_fp_phase1.src.rtn4_difference import iter_group_rows

    base = RTN4WeightState(
        tensor_name="model.layers.0.self_attn.q_proj.weight",
        block_id=0,
        module_type="q_proj",
        qcode=torch.tensor([[1, 2]], dtype=torch.int16),
        scale=torch.tensor([[1.0]], dtype=torch.float32),
        zero_point=torch.tensor([[0.0]], dtype=torch.float32),
        dequant=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        group_size=2,
    )
    if_state = RTN4WeightState(
        tensor_name=base.tensor_name,
        block_id=0,
        module_type="q_proj",
        qcode=base.qcode.clone(),
        scale=torch.tensor([[2.0]], dtype=torch.float32),
        zero_point=torch.tensor([[1.0]], dtype=torch.float32),
        dequant=base.dequant.clone(),
        group_size=2,
    )

    row = next(iter_group_rows(base, if_state))

    assert row["scale_different"] is True
    assert row["zero_point_different"] is True


def test_rtn4_state_matching_rejects_block_or_module_mismatch():
    import pytest
    import torch
    from quant_fp_phase1.src.rtn4_quant_state import RTN4WeightState, assert_matching_rtn4_states

    base = RTN4WeightState(
        tensor_name="model.layers.0.self_attn.q_proj.weight",
        block_id=0,
        module_type="q_proj",
        qcode=torch.zeros((1, 1), dtype=torch.int16),
        scale=torch.ones((1, 1)),
        zero_point=torch.zeros((1, 1)),
        dequant=torch.zeros((1, 1)),
    )
    other = RTN4WeightState(
        tensor_name=base.tensor_name,
        block_id=1,
        module_type="k_proj",
        qcode=base.qcode,
        scale=base.scale,
        zero_point=base.zero_point,
        dequant=base.dequant,
    )

    with pytest.raises(ValueError, match="metadata mismatch"):
        assert_matching_rtn4_states({base.tensor_name: base}, {other.tensor_name: other})


def test_select_group_swap_candidates_returns_all_when_limit_is_none():
    from quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis import select_group_swap_candidates

    candidates = [
        {"qcode_diff_ratio": 0.1, "dequant_diff_l2": 1.0, "dequant_diff_max": 1.0},
        {"qcode_diff_ratio": 0.9, "dequant_diff_l2": 0.1, "dequant_diff_max": 0.1},
        {"qcode_diff_ratio": 0.2, "dequant_diff_l2": 2.0, "dequant_diff_max": 2.0},
    ]

    selected = select_group_swap_candidates(candidates, limit=None)

    assert len(selected) == 3
    assert selected[0]["qcode_diff_ratio"] == 0.9


def test_assert_only_expected_swap_detects_unintended_tensor_change():
    import pytest
    import torch
    from types import SimpleNamespace
    from quant_fp_phase1.src.rtn4_quant_state import RTN4WeightState
    from quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis import assert_only_expected_swap, full_tensor_region

    model = torch.nn.Module()
    block = torch.nn.Module()
    block.a_proj = torch.nn.Linear(2, 1, bias=False)
    block.b_proj = torch.nn.Linear(2, 1, bias=False)
    model.model = SimpleNamespace(layers=torch.nn.ModuleList([block]))
    with torch.no_grad():
        block.a_proj.weight.copy_(torch.tensor([[10.0, 10.0]]))
        block.b_proj.weight.copy_(torch.tensor([[99.0, 99.0]]))

    def state(name, values):
        tensor = torch.tensor(values, dtype=torch.float32)
        return RTN4WeightState(
            tensor_name=name,
            block_id=0,
            module_type="unknown",
            qcode=torch.zeros_like(tensor, dtype=torch.int16),
            scale=torch.ones((1, 1)),
            zero_point=torch.zeros((1, 1)),
            dequant=tensor,
            group_size=2,
        )

    if_states = {
        "model.layers.0.a_proj.weight": state("model.layers.0.a_proj.weight", [[0.0, 0.0]]),
        "model.layers.0.b_proj.weight": state("model.layers.0.b_proj.weight", [[1.0, 1.0]]),
    }
    base_states = {
        "model.layers.0.a_proj.weight": state("model.layers.0.a_proj.weight", [[10.0, 10.0]]),
        "model.layers.0.b_proj.weight": state("model.layers.0.b_proj.weight", [[20.0, 20.0]]),
    }

    with pytest.raises(RuntimeError, match="unexpected swap state"):
        assert_only_expected_swap(model, if_states, base_states, [full_tensor_region("model.layers.0.a_proj.weight")])


def test_rtn4_summary_lists_causal_swap_regions_and_no_block_case(tmp_path):
    from quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis import write_summary, parse_args
    from quant_fp_phase1.src.rtn4_difference import write_csv

    results = tmp_path / "results"
    results.mkdir()
    base_tensor = {
        "tensor_name": "model.layers.0.self_attn.q_proj.weight",
        "block_id": 0,
        "module_type": "q_proj",
        "num_weights": 4,
        "num_qcode_different": 2,
        "qcode_diff_ratio": 0.5,
        "dequant_diff_l1": 3.0,
        "dequant_diff_l2": 2.0,
        "dequant_diff_max": 1.5,
        "dequant_diff_mean_abs": 0.75,
        "num_groups": 2,
        "num_groups_scale_different": 1,
        "num_groups_zp_different": 1,
    }
    write_csv(results / "rtn4_diff_by_tensor.csv", [base_tensor])
    write_csv(results / "rtn4_diff_by_block.csv", [{"block_id": 0, "num_weights": 4, "num_qcode_different": 2, "qcode_diff_ratio": 0.5, "dequant_diff_l2": 2.0}])
    write_csv(results / "rtn4_diff_by_block_module.csv", [{"block_id": 0, "module_type": "q_proj", "num_weights": 4, "num_qcode_different": 2, "qcode_diff_ratio": 0.5, "dequant_diff_l2": 2.0}])
    write_csv(results / "rtn4_block_swap.csv", [{"block_id": 0, "fingerprint_drop": 0, "verified_count_after_swap": 8}])
    write_csv(results / "rtn4_module_swap.csv", [])
    write_csv(results / "rtn4_group_swap.csv", [])

    args = parse_args(["--output-dir", str(tmp_path)])
    write_summary(args)
    no_drop_text = (tmp_path / "RTN4_BASE_VS_IF_SUMMARY.md").read_text(encoding="utf-8")
    assert "No single transformer block is individually necessary" in no_drop_text

    write_csv(results / "rtn4_block_swap.csv", [{"block_id": 3, "fingerprint_drop": 2, "verified_count_after_swap": 6}])
    write_csv(results / "rtn4_module_swap.csv", [{"block_id": 3, "module_type": "down_proj", "fingerprint_drop": 1, "verified_count_after_swap": 7}])
    write_csv(results / "rtn4_group_swap.csv", [{"tensor_name": "model.layers.3.mlp.down_proj.weight", "block_id": 3, "module_type": "down_proj", "output_row": 9, "group_id": 4, "fingerprint_drop": 1, "verified_count_after_swap": 7}])

    write_summary(args)
    text = (tmp_path / "RTN4_BASE_VS_IF_SUMMARY.md").read_text(encoding="utf-8")
    assert "block_id=3" in text
    assert "module_type=down_proj" in text
    assert "output_row=9" in text
    assert "group_id=4" in text


def test_rtn4_analysis_shell_does_not_apply_group_swap_limit_by_default():
    from pathlib import Path

    text = Path("run_rtn4_base_vs_if_analysis.sh").read_text(encoding="utf-8")

    assert "--max-group-swaps-per-module" not in text
    assert "MAX_GROUP_SWAPS_PER_MODULE" not in text


def test_filter_group_rows_for_modules_streams_without_materializing_all_rows(monkeypatch):
    from quant_fp_phase1.scripts import run_rtn4_base_vs_if_analysis as runner

    def forbidden_read_table_rows(_path):
        raise AssertionError("group swap candidates should not read the full group table")

    chunks = [
        [
            {"block_id": 0, "module_type": "q_proj", "qcode_diff_ratio": 0.1},
            {"block_id": 1, "module_type": "down_proj", "qcode_diff_ratio": 0.9},
        ],
        [
            {"block_id": 1, "module_type": "up_proj", "qcode_diff_ratio": 0.8},
            {"block_id": 1, "module_type": "down_proj", "qcode_diff_ratio": 0.2},
        ],
    ]

    monkeypatch.setattr(runner, "read_table_rows", forbidden_read_table_rows)
    monkeypatch.setattr(runner, "iter_table_row_batches", lambda _path, columns=None: iter(chunks))

    rows = list(runner.filter_group_rows_for_modules("groups.parquet", {(1, "down_proj")}))

    assert [row["qcode_diff_ratio"] for row in rows] == [0.9, 0.2]


def test_build_analysis_config_payload_confirms_identical_rtn4_configs():
    from quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis import build_analysis_config_payload, parse_args

    args = parse_args(["--dtype", "fp16", "--device-map", "cuda:0", "--group-size", "128"])

    payload = build_analysis_config_payload(args)

    assert payload["base_rtn4_config"] == payload["if_rtn4_config"]
    assert payload["rtn4_configs_identical"] is True
    assert payload["dtype"] == "fp16"
    assert payload["device_map"] == "cuda:0"


def test_rtn4_base_vs_if_outputs_use_dedicated_default_directory():
    from pathlib import Path
    from quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis import parse_args

    args = parse_args([])
    shell_text = Path("run_rtn4_base_vs_if_analysis.sh").read_text(encoding="utf-8")

    assert args.output_dir == "quant_fp_phase1/rtn4_base_vs_if_analysis"
    assert 'OUTPUT_DIR:-quant_fp_phase1/rtn4_base_vs_if_analysis' in shell_text

def test_assert_only_expected_swap_does_not_clone_unrelated_states():
    import torch
    from types import SimpleNamespace
    from quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis import assert_only_expected_swap, full_tensor_region

    class DequantProxy:
        def __init__(self, tensor, clone_allowed):
            self.tensor = tensor
            self.clone_allowed = clone_allowed

        def clone(self):
            if not self.clone_allowed:
                raise AssertionError("unrelated dequant state was cloned")
            return self.tensor.clone()

        def cpu(self):
            return self.tensor.cpu()

    def state(name, values, clone_allowed=False):
        return SimpleNamespace(
            tensor_name=name,
            dequant=DequantProxy(torch.tensor(values, dtype=torch.float32), clone_allowed),
        )

    model = torch.nn.Module()
    block = torch.nn.Module()
    block.a_proj = torch.nn.Linear(2, 1, bias=False)
    block.b_proj = torch.nn.Linear(2, 1, bias=False)
    model.model = SimpleNamespace(layers=torch.nn.ModuleList([block]))
    with torch.no_grad():
        block.a_proj.weight.copy_(torch.tensor([[10.0, 10.0]]))
        block.b_proj.weight.copy_(torch.tensor([[1.0, 1.0]]))

    if_states = {
        "model.layers.0.a_proj.weight": state("model.layers.0.a_proj.weight", [[0.0, 0.0]], clone_allowed=True),
        "model.layers.0.b_proj.weight": state("model.layers.0.b_proj.weight", [[1.0, 1.0]], clone_allowed=False),
    }
    base_states = {
        "model.layers.0.a_proj.weight": state("model.layers.0.a_proj.weight", [[10.0, 10.0]], clone_allowed=True),
        "model.layers.0.b_proj.weight": state("model.layers.0.b_proj.weight", [[20.0, 20.0]], clone_allowed=False),
    }

    assert_only_expected_swap(model, if_states, base_states, [full_tensor_region("model.layers.0.a_proj.weight")])

def test_coordinate_row_table_preserves_coordinate_fields_without_pylist_dicts():
    import torch
    from quant_fp_phase1.src.rtn4_quant_state import RTN4WeightState
    from quant_fp_phase1.src.rtn4_difference import coordinate_row_table

    class FakeTable:
        def __init__(self, columns):
            self.columns = columns
            self.num_rows = len(next(iter(columns.values())))

        @classmethod
        def from_pydict(cls, columns):
            return cls(columns)

        def to_pylist(self):
            return [
                {key: values[index] for key, values in self.columns.items()}
                for index in range(self.num_rows)
            ]

    class FakePA:
        Table = FakeTable

    base = RTN4WeightState(
        tensor_name="model.layers.0.self_attn.q_proj.weight",
        block_id=0,
        module_type="q_proj",
        qcode=torch.tensor([[1, 2, 3]], dtype=torch.int16),
        scale=torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        zero_point=torch.tensor([[0.0, 1.0]], dtype=torch.float32),
        dequant=torch.tensor([[0.0, 1.0, 2.0]], dtype=torch.float32),
        group_size=2,
    )
    if_state = RTN4WeightState(
        tensor_name=base.tensor_name,
        block_id=0,
        module_type="q_proj",
        qcode=torch.tensor([[1, 5, 4]], dtype=torch.int16),
        scale=torch.tensor([[2.0, 4.0]], dtype=torch.float32),
        zero_point=torch.tensor([[0.0, 3.0]], dtype=torch.float32),
        dequant=torch.tensor([[0.0, 3.0, 6.0]], dtype=torch.float32),
        group_size=2,
    )

    table = coordinate_row_table(base, if_state, output_row=0, pa=FakePA)
    rows = table.to_pylist()

    assert table.num_rows == 3
    assert rows[1]["group_id"] == 0
    assert rows[2]["group_id"] == 1
    assert rows[1]["qcode_diff"] == 3
    assert rows[1]["same_qcode"] is False
    assert rows[2]["base_scale"] == 2.0
    assert rows[2]["zero_point_diff"] == 2.0
    assert rows[2]["dequant_diff"] == 4.0

def test_default_all_stage_reloads_models_between_diff_and_swaps_to_limit_peak_ram():
    from quant_fp_phase1.scripts.run_rtn4_base_vs_if_analysis import parse_args, should_reload_models_between_diff_and_swaps

    assert should_reload_models_between_diff_and_swaps(parse_args([])) is True
    assert should_reload_models_between_diff_and_swaps(parse_args(["--stages", "diff"])) is False

def test_rtn4_state_stores_qcode_as_uint8_for_memory_efficiency(monkeypatch):
    import torch
    from types import SimpleNamespace
    from quant_fp_phase1.src import rtn4_quant_state

    class FakeRawState:
        original_dtype = torch.float32
        in_features = 2
        padded_in_features = 128
        max_int = 15
        pre_round = torch.tensor([[0.0, 15.0] + [0.0] * 126], dtype=torch.float32)
        scale = torch.ones((1, 128), dtype=torch.float32)
        zero_point = torch.zeros((1, 128), dtype=torch.float32)

        def dequantize_truncated(self):
            return torch.zeros((1, 2), dtype=torch.float32)

    monkeypatch.setattr(rtn4_quant_state, "rtn_quantize_weight_raw", lambda *args, **kwargs: FakeRawState())

    model = torch.nn.Module()
    model.model = SimpleNamespace(layers=torch.nn.ModuleList([torch.nn.Sequential(torch.nn.Linear(2, 1, bias=False))]))

    state = next(iter(rtn4_quant_state.quantize_rtn4_with_state(model, group_size=128).values()))

    assert state.qcode.dtype == torch.uint8
    assert state.qcode.tolist() == [[0, 15]]

def test_behavior_diff_stages_release_each_model_before_loading_next(monkeypatch, tmp_path):
    import torch
    from quant_fp_phase1.scripts import run_rtn4_base_vs_if_analysis as runner
    from quant_fp_phase1.src.rtn4_quant_state import RTN4WeightState

    active_models = []
    active_counts_at_load = []

    class FakeConfig:
        num_hidden_layers = 0

    class FakeModel:
        config = FakeConfig()

        def __init__(self, name):
            self.name = name
            active_counts_at_load.append(len(active_models))
            active_models.append(name)

        def __del__(self):
            if self.name in active_models:
                active_models.remove(self.name)

    def state(name):
        return RTN4WeightState(
            tensor_name="model.layers.0.self_attn.q_proj.weight",
            block_id=0,
            module_type="q_proj",
            qcode=torch.zeros((1, 1), dtype=torch.uint8),
            scale=torch.ones((1, 1)),
            zero_point=torch.zeros((1, 1)),
            dequant=torch.zeros((1, 1)),
        )

    monkeypatch.setattr(runner, "load_rtn4_model_and_state", lambda model_path, args: (FakeModel(model_path), object(), {"model.layers.0.self_attn.q_proj.weight": state(model_path)}))
    monkeypatch.setattr(runner, "selected_fingerprint_examples", lambda args: [{"dataset_index": 0}])
    monkeypatch.setattr(runner, "behavior_rows", lambda variant, model, tokenizer, examples, args: [{"sample_id": 0, "dataset_index": 0, "model_variant": variant, "verified": variant == "IF-RTN4", "generated_text": "x", "expected_text": "x"}])
    monkeypatch.setattr(runner, "write_exact_diff_and_aggregates", lambda base_states, if_states, args: active_models == [] or (_ for _ in ()).throw(AssertionError("models should be released before diff")))

    args = runner.parse_args(["--output-dir", str(tmp_path), "--stages", "behavior", "diff", "--no-strict-behavior-control"])
    runner.run_analysis(args)

    assert active_counts_at_load == [0, 0]



def test_cumulative_block_swap_plan_matches_spec():
    from quant_fp_phase1.scripts.run_rtn4_cumulative_block_swap import build_initial_configs

    configs = build_initial_configs(32)
    ids = [config.config_id for config in configs]

    assert ids[:2] == ["IF-RTN4", "BASE-RTN4"]
    assert "P04" in ids and "P28" in ids
    assert "S04" in ids and "S28" in ids
    assert "D04" in ids and "D28" in ids
    assert ids.count("ALL32") == 1
    assert "P32" not in ids and "S32" not in ids and "D32" not in ids
    assert next(config.blocks for config in configs if config.config_id == "P08") == tuple(range(8))
    assert next(config.blocks for config in configs if config.config_id == "S08") == tuple(range(24, 32))
    assert next(config.blocks for config in configs if config.config_id == "D04") == (0, 16, 8, 24)


def test_cumulative_refinement_only_adds_transition_intervals():
    from quant_fp_phase1.scripts.run_rtn4_cumulative_block_swap import refinement_configs

    rows = [
        {"family": "prefix", "num_swapped_blocks": 16, "verified_count": 8},
        {"family": "prefix", "num_swapped_blocks": 20, "verified_count": 3},
        {"family": "suffix", "num_swapped_blocks": 16, "verified_count": 8},
        {"family": "suffix", "num_swapped_blocks": 20, "verified_count": 8},
        {"family": "distributed", "num_swapped_blocks": 12, "verified_count": 8},
        {"family": "distributed", "num_swapped_blocks": 16, "verified_count": 2},
    ]

    configs = refinement_configs(rows, 32)
    ids = [config.config_id for config in configs]

    assert ids == ["P17", "P18", "P19", "D13", "D14", "D15"]


def test_apply_cumulative_block_swap_replaces_selected_blocks_and_restores_others():
    import torch
    from types import SimpleNamespace
    from quant_fp_phase1.scripts.run_rtn4_cumulative_block_swap import apply_cumulative_block_swap, assert_cumulative_swap_state
    from quant_fp_phase1.src.rtn4_quant_state import MODULE_TYPES, RTN4WeightState

    class Attn(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = torch.nn.Linear(2, 1, bias=False)
            self.k_proj = torch.nn.Linear(2, 1, bias=False)
            self.v_proj = torch.nn.Linear(2, 1, bias=False)
            self.o_proj = torch.nn.Linear(2, 1, bias=False)

    class Mlp(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = torch.nn.Linear(2, 1, bias=False)
            self.up_proj = torch.nn.Linear(2, 1, bias=False)
            self.down_proj = torch.nn.Linear(2, 1, bias=False)

    class Block(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()
            self.mlp = Mlp()

    model = torch.nn.Module()
    model.model = SimpleNamespace(layers=torch.nn.ModuleList([Block(), Block()]))
    base_states = {}
    if_states = {}
    for block_id in range(2):
        for module_type in MODULE_TYPES:
            parent = "self_attn" if module_type.endswith("_proj") and module_type in {"q_proj", "k_proj", "v_proj", "o_proj"} else "mlp"
            name = f"model.layers.{block_id}.{parent}.{module_type}.weight"
            if_value = torch.full((1, 2), float(block_id + 10))
            base_value = torch.full((1, 2), float(block_id + 100))
            if_states[name] = RTN4WeightState(name, block_id, module_type, torch.zeros((1, 2), dtype=torch.uint8), torch.ones((1, 1)), torch.zeros((1, 1)), if_value)
            base_states[name] = RTN4WeightState(name, block_id, module_type, torch.zeros((1, 2), dtype=torch.uint8), torch.ones((1, 1)), torch.zeros((1, 1)), base_value)

    replaced = apply_cumulative_block_swap(model, base_states, if_states, blocks=(1,))
    assert len(replaced) == 7
    assert_cumulative_swap_state(model, base_states, if_states, blocks=(1,))


def test_cumulative_swap_asserts_lm_head_is_unchanged():
    import torch
    from quant_fp_phase1.scripts.run_rtn4_cumulative_block_swap import assert_lm_head_unchanged, snapshot_lm_head_weight

    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.lm_head.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    snapshot = snapshot_lm_head_weight(model)
    assert_lm_head_unchanged(model, snapshot)

    with torch.no_grad():
        model.lm_head.weight[0, 0] = 99.0
    with pytest.raises(RuntimeError, match="lm_head changed"):
        assert_lm_head_unchanged(model, snapshot)

def test_cumulative_runner_only_all32_selects_single_full_swap_config():
    from quant_fp_phase1.scripts.run_rtn4_cumulative_block_swap import configs_for_run, parse_args

    args = parse_args(["--only-all32"])
    configs = configs_for_run(args, 32)

    assert args.only_all32 is True
    assert [config.config_id for config in configs] == ["ALL32"]
    assert configs[0].blocks == tuple(range(32))
