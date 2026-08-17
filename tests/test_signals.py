"""Tests for core signal abstractions."""

import pytest
import torch
import torch.nn as nn

import gradiend
from gradiend import GradiendModel
from gradiend.trainer.core.signals import (
    ActivationSignalExtractor,
    GradientSignalExtractor,
    Signal,
    SignalBatch,
    SignalScope,
    SignalSet,
    SignalSpace,
    normalize_signal_arguments,
    require_single_gradient_signal,
)
from gradiend.signal_space import default_activation_sites, resolve_activation_modules


def test_gradient_signal_has_no_scope_options():
    signal = Signal.gradient()

    assert signal.kind == "gradient"
    assert signal.id == "gradient"
    assert signal.options == {}


def test_activation_signal_keeps_token_selector_but_not_sites():
    signal = Signal.activation(name="act", token_selector="mask")

    assert signal.kind == "activation"
    assert signal.id == "act"
    assert signal.options == {"token_selector": "mask"}
    assert "sites" not in signal.options


def test_product_signal_records_source_signal_ids():
    signal = Signal.product("activation", "activation_gradient", name="act_x_grad")

    assert signal.kind == "product"
    assert signal.id == "act_x_grad"
    assert signal.options == {
        "left": "activation",
        "right": "activation_gradient",
    }


def test_signal_roundtrip_dict():
    signal = Signal.activation_gradient(name="ag", token_selector={"role": "mask"})

    restored = Signal.from_dict(signal.to_dict())

    assert restored == signal
    assert restored.options == {"token_selector": {"role": "mask"}}


def test_signal_set_preserves_order_and_rejects_duplicates():
    signals = SignalSet(
        Signal.gradient(name="gradient"),
        Signal.activation(name="activation", token_selector="mask"),
    )

    assert signals.ids == ("gradient", "activation")
    assert [s.kind for s in signals] == ["gradient", "activation"]
    assert signals["activation"].kind == "activation"

    with pytest.raises(ValueError, match="Duplicate signal id"):
        SignalSet(Signal.gradient(), Signal.gradient())


def test_signal_set_single_requires_one_signal():
    signals = SignalSet.gradient()

    assert signals.is_single
    assert signals.single.kind == "gradient"

    multi = SignalSet(Signal.gradient(name="g"), Signal.activation(name="a"))
    with pytest.raises(ValueError, match="not one"):
        _ = multi.single


def test_normalize_signal_arguments_defaults_to_gradient():
    signal, signals = normalize_signal_arguments()

    assert signal == Signal.gradient()
    assert signals.ids == ("gradient",)


def test_require_single_gradient_signal_rejects_activation():
    with pytest.raises(NotImplementedError, match="Signal.gradient"):
        require_single_gradient_signal(signal=Signal.activation(), context="test")


def test_require_single_gradient_signal_rejects_multi_signal():
    with pytest.raises(NotImplementedError, match="exactly one raw gradient signal"):
        require_single_gradient_signal(
            signals=SignalSet(Signal.gradient(), Signal.activation(name="activation")),
            context="test",
        )


def test_signal_scope_keeps_scope_separate_from_signal():
    scope = SignalScope.from_values(
        params=["bert.encoder.layer.*"],
        activation_sites=["bert.encoder.layer.*.output"],
    )

    assert scope.params == ("bert.encoder.layer.*",)
    assert scope.activation_sites == ("bert.encoder.layer.*.output",)


def test_signal_scope_roundtrip_dict():
    scope = SignalScope.from_values(params=["p*"], activation_sites=["emb", "proj"])

    restored = SignalScope.from_dict(scope.to_dict())

    assert restored == scope


def test_signal_scope_accepts_single_string_values():
    scope = SignalScope.from_values(params="p*", activation_sites="emb")

    assert scope.params == ("p*",)
    assert scope.activation_sites == ("emb",)


def test_signal_scope_default_and_full_presets_roundtrip():
    default_scope = SignalScope.default()
    full_scope = SignalScope.full()

    assert default_scope.mode == "default"
    assert full_scope.mode == "full"
    assert SignalScope.from_dict(default_scope.to_dict()) == default_scope
    assert SignalScope.from_dict(full_scope.to_dict()) == full_scope


def test_signal_scope_semantic_shortcuts_roundtrip():
    all_layers = SignalScope.layers()
    star_layers = SignalScope.layers("*")
    one_layer = SignalScope.layer(1)
    embeddings = SignalScope.embeddings()
    word_embedding = SignalScope.word_embedding()

    assert all_layers == star_layers
    assert all_layers.activation_selector == ("layers", None)
    assert one_layer.activation_selector == ("layers", (1,))
    assert embeddings.activation_selector == ("embeddings",)
    assert word_embedding.activation_selector == ("word_embedding",)
    assert SignalScope.from_dict(all_layers.to_dict()) == all_layers
    assert SignalScope.from_dict(one_layer.to_dict()) == one_layer
    assert SignalScope.from_dict(embeddings.to_dict()) == embeddings
    assert SignalScope.from_dict(word_embedding.to_dict()) == word_embedding


def test_signal_scope_from_signal_space_uses_saved_activation_sites():
    signal_space = {
        "kind": "activation",
        "mapping": [
            {"name": "transformer.wte", "shape": [768], "repr": "all"},
            {"name": "transformer.h.0", "shape": [768], "repr": "all"},
        ],
    }
    scope = SignalScope.from_signal_space(signal_space)
    assert scope.activation_sites == ("transformer.wte", "transformer.h.0")


def test_signal_scope_from_signal_space_strips_activation_prefix():
    signal_space = {
        "kind": "activation",
        "mapping": [{"name": "activation:transformer.h.1", "shape": [768], "repr": "all"}],
    }
    scope = SignalScope.from_signal_space(signal_space)
    assert scope.activation_sites == ("transformer.h.1",)


def test_signal_scope_rejects_mixed_raw_and_semantic_activation_scopes():
    with pytest.raises(ValueError, match="either activation_sites or a semantic"):
        SignalScope.from_values(
            activation_sites=["encoder.layer.0"],
            activation_selector=("layers", (0,)),
        )


def test_signal_space_validates_positive_input_dim():
    signal = Signal.gradient()
    space = SignalSpace(signal=signal, input_dim=4, mapping={"kind": "param_map"})

    assert space.signal is signal
    assert space.input_dim == 4

    with pytest.raises(ValueError, match="positive int"):
        SignalSpace(signal=signal, input_dim=0)


def test_signal_batch_computes_diff_and_selects_source_target_keywords():
    factual = torch.tensor([3.0, 5.0])
    alternative = torch.tensor([1.0, 2.0])

    batch = SignalBatch.from_factual_alternative(factual, alternative)

    assert torch.equal(batch.select("factual"), factual)
    assert torch.equal(batch.select("alternative"), alternative)
    assert torch.equal(batch.select("diff"), torch.tensor([2.0, 3.0]))
    assert batch.select(None) is None
    with pytest.raises(ValueError, match="Unknown signal selection"):
        batch.select("activation")


def test_gradient_signal_extractor_wraps_existing_gradient_creator():
    calls = []

    def gradient_creator(inputs):
        calls.append(inputs)
        return torch.tensor(inputs, dtype=torch.float32)

    extractor = GradientSignalExtractor(gradient_creator)
    batch = extractor([1.0, 2.0], [3.0, 5.0])

    assert calls == [[1.0, 2.0], [3.0, 5.0]]
    assert batch.signal_id == "gradient"
    assert torch.equal(batch.factual, torch.tensor([1.0, 2.0]))
    assert torch.equal(batch.alternative, torch.tensor([3.0, 5.0]))
    assert torch.equal(batch.diff, torch.tensor([-2.0, -3.0]))


def test_gradient_signal_extractor_can_skip_unneeded_side():
    calls = []

    def gradient_creator(inputs):
        calls.append(inputs)
        return torch.tensor([float(inputs)])

    extractor = GradientSignalExtractor(gradient_creator)
    batch = extractor(7, 9, requires_alternative=False)

    assert calls == [7]
    assert torch.equal(batch.factual, torch.tensor([7.0]))
    assert batch.alternative is None
    assert batch.diff is None


class TinyActivationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(10, 3)
        self.relu = nn.ReLU()
        self.proj = nn.Linear(3, 2, bias=False)
        with torch.no_grad():
            self.emb.weight.copy_(torch.arange(30, dtype=torch.float32).reshape(10, 3))
            self.proj.weight.copy_(torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))

    def forward(self, input_ids, attention_mask=None):
        hidden = self.relu(self.emb(input_ids))
        return {"logits": self.proj(hidden)}


class BertishEmbeddings(nn.Module):
    def __init__(self, vocab_size=12, hidden_size=4, max_positions=8):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(max_positions, hidden_size)

    def forward(self, input_ids):
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        return self.word_embeddings(input_ids) + self.position_embeddings(positions)


class BertishLayer(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.output = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        return self.output(self.dense(hidden_states))


class BertishEncoder(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.layer = nn.ModuleList([BertishLayer(hidden_size), BertishLayer(hidden_size)])

    def forward(self, hidden_states):
        for layer in self.layer:
            hidden_states = layer(hidden_states)
        return hidden_states


class BertishActivationModel(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.config = type("Config", (), {"model_type": "bert", "architectures": ["BertForMaskedLM"]})()
        self.embeddings = BertishEmbeddings(hidden_size=hidden_size)
        self.encoder = BertishEncoder(hidden_size=hidden_size)
        self.cls = nn.Linear(hidden_size, 12)

    def forward(self, input_ids, attention_mask=None):
        hidden_states = self.embeddings(input_ids)
        hidden_states = self.encoder(hidden_states)
        return {"logits": self.cls(hidden_states)}


class TinyTokenizer:
    mask_token_id = 9
    cls_token_id = 8


class TinyConceptActivationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(12, 4)
        self.proj = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.emb.weight.zero_()
            self.emb.weight[1] = torch.tensor([0.1, 0.0, 0.0, 0.0])
            self.emb.weight[2] = torch.tensor([0.0, 0.1, 0.0, 0.0])
            self.emb.weight[3] = torch.tensor([2.0, 0.1, 0.0, 0.0])
            self.emb.weight[4] = torch.tensor([2.2, -0.1, 0.0, 0.0])
            self.emb.weight[7] = torch.tensor([-2.0, -0.1, 0.0, 0.0])
            self.emb.weight[8] = torch.tensor([-2.2, 0.1, 0.0, 0.0])
            self.proj.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]))

    def forward(self, input_ids, attention_mask=None):
        hidden = self.emb(input_ids)
        return {"logits": self.proj(hidden)}


def _concept_inputs(ids):
    return {"input_ids": torch.tensor([ids], dtype=torch.long)}


class GPT2ishBlock(nn.Module):
    def __init__(self, hidden_size=5):
        super().__init__()
        self.ln_1 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        return self.ln_1(hidden_states)


class GPT2ishTransformer(nn.Module):
    def __init__(self, hidden_size=5):
        super().__init__()
        self.wte = nn.Embedding(13, hidden_size)
        self.h = nn.ModuleList([GPT2ishBlock(hidden_size), GPT2ishBlock(hidden_size)])


class GPT2ishActivationModel(nn.Module):
    def __init__(self, hidden_size=5):
        super().__init__()
        self.config = type("Config", (), {"model_type": "gpt2", "architectures": ["GPT2LMHeadModel"]})()
        self.transformer = GPT2ishTransformer(hidden_size)
        self.lm_head = nn.Linear(hidden_size, 13)


class LlamaishLayer(nn.Module):
    def __init__(self, hidden_size=6):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        return self.input_layernorm(hidden_states)


class LlamaishCore(nn.Module):
    def __init__(self, hidden_size=6):
        super().__init__()
        self.embed_tokens = nn.Embedding(17, hidden_size)
        self.layers = nn.ModuleList([LlamaishLayer(hidden_size), LlamaishLayer(hidden_size)])


class LlamaishActivationModel(nn.Module):
    def __init__(self, hidden_size=6):
        super().__init__()
        self.config = type("Config", (), {"model_type": "llama", "architectures": ["LlamaForCausalLM"]})()
        self.model = LlamaishCore(hidden_size)
        self.lm_head = nn.Linear(hidden_size, 17)


class DistilbertishBlock(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.sa_layer_norm = nn.LayerNorm(hidden_size)
        self.output_layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        return self.output_layer_norm(self.sa_layer_norm(hidden_states))


class DistilbertishTransformer(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.layer = nn.ModuleList([DistilbertishBlock(hidden_size), DistilbertishBlock(hidden_size)])


class DistilbertishCore(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.embeddings = BertishEmbeddings(hidden_size=hidden_size)
        self.transformer = DistilbertishTransformer(hidden_size=hidden_size)


class DistilbertishActivationModel(nn.Module):
    def __init__(self, hidden_size=4):
        super().__init__()
        self.config = type(
            "Config",
            (),
            {"model_type": "distilbert", "architectures": ["DistilBertForMaskedLM"]},
        )()
        self.distilbert = DistilbertishCore(hidden_size=hidden_size)
        self.vocab_transform = nn.Linear(hidden_size, hidden_size)
        self.vocab_projector = nn.Linear(hidden_size, 12)


class OPTishDecoder(nn.Module):
    def __init__(self, hidden_size=5):
        super().__init__()
        self.embed_tokens = nn.Embedding(11, hidden_size)
        self.layers = nn.ModuleList([LlamaishLayer(hidden_size), LlamaishLayer(hidden_size)])


class OPTishModel(nn.Module):
    def __init__(self, hidden_size=5):
        super().__init__()
        self.decoder = OPTishDecoder(hidden_size)


class OPTishActivationModel(nn.Module):
    def __init__(self, hidden_size=5):
        super().__init__()
        self.config = type("Config", (), {"model_type": "opt", "architectures": ["OPTForCausalLM"]})()
        self.model = OPTishModel(hidden_size)
        self.lm_head = nn.Linear(hidden_size, 11)


def test_activation_signal_extractor_captures_site_and_token_index():
    model = TinyActivationModel()
    extractor = ActivationSignalExtractor(
        model,
        signal=Signal.activation(token_selector=1),
        scope=SignalScope.from_values(activation_sites=["emb"]),
    )

    batch = extractor(
        factual_inputs={"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]])},
        alternative_inputs={"input_ids": torch.tensor([[2, 3, 4], [5, 6, 7]])},
    )

    assert batch.signal_id == "activation"
    assert torch.equal(batch.factual, torch.tensor([10.5, 11.5, 12.5]))
    assert torch.equal(batch.alternative, torch.tensor([13.5, 14.5, 15.5]))
    assert torch.equal(batch.diff, torch.tensor([-3.0, -3.0, -3.0]))


def test_activation_signal_extractor_supports_mask_selector_and_multiple_sites():
    model = TinyActivationModel()
    extractor = ActivationSignalExtractor(
        model,
        signal=Signal.activation(token_selector="mask"),
        scope=SignalScope.from_values(activation_sites=["emb", "proj"]),
        tokenizer=TinyTokenizer(),
    )

    batch = extractor(
        factual_inputs={"input_ids": torch.tensor([[1, 9, 3], [4, 9, 6]])},
        alternative_inputs=None,
        requires_alternative=False,
    )

    assert torch.equal(batch.factual, torch.tensor([27.0, 28.0, 29.0, 27.0, 28.0]))
    assert batch.alternative is None
    assert batch.diff is None


def test_activation_signal_extractor_supports_prediction_selector():
    model = TinyActivationModel()
    extractor = ActivationSignalExtractor(
        model,
        signal=Signal.activation(token_selector="prediction"),
        scope=SignalScope.from_values(activation_sites=["emb"]),
    )

    batch = extractor(
        factual_inputs={
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "prediction_mask": torch.tensor([[False, True, False], [False, True, False]]),
        },
        alternative_inputs=None,
        requires_alternative=False,
    )

    assert torch.equal(batch.factual, torch.tensor([10.5, 11.5, 12.5]))
    assert batch.factual_target is None
    assert batch.alternative is None
    assert batch.diff is None


def test_activation_signal_keeps_target_token_selector():
    signal = Signal.activation(
        token_selector="pre_prediction",
        target_token_selector="prediction",
    )
    assert signal.options == {
        "token_selector": "pre_prediction",
        "target_token_selector": "prediction",
    }
    same = Signal.activation(token_selector="prediction", target_token_selector="prediction")
    assert same.options == {"token_selector": "prediction"}


def test_activation_signal_extractor_mixed_site_one_forward_two_gathers():
    class CountingActivationModel(TinyActivationModel):
        def __init__(self):
            super().__init__()
            self.n_forward = 0

        def forward(self, input_ids, attention_mask=None):
            self.n_forward += 1
            return super().forward(input_ids, attention_mask=attention_mask)

    model = CountingActivationModel()
    extractor = ActivationSignalExtractor(
        model,
        signal=Signal.activation(
            token_selector="pre_prediction",
            target_token_selector="prediction",
        ),
        scope=SignalScope.from_values(activation_sites=["emb"]),
    )
    batch = extractor(
        factual_inputs={
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "prediction_mask": torch.tensor([[False, True, False], [False, True, False]]),
        },
        alternative_inputs=None,
        requires_alternative=False,
    )
    assert model.n_forward == 1
    # pre_prediction = index 0; prediction = index 1; batch-mean over 2 rows
    assert torch.equal(batch.factual, torch.tensor([7.5, 8.5, 9.5]))
    assert torch.equal(batch.factual_target, torch.tensor([10.5, 11.5, 12.5]))
    assert batch.alternative_target is None


def test_activation_signal_extractor_can_infer_static_input_dim_for_known_sites():
    extractor = ActivationSignalExtractor(
        TinyActivationModel(),
        signal=Signal.activation(token_selector="mask"),
        scope=SignalScope.from_values(activation_sites=["emb", "proj"]),
        tokenizer=TinyTokenizer(),
    )

    assert extractor.infer_input_dim_static() == 5


def test_activation_signal_extractor_static_input_dim_returns_none_when_unreliable():
    unknown_site = ActivationSignalExtractor(
        TinyActivationModel(),
        signal=Signal.activation(token_selector="mask"),
        scope=SignalScope.from_values(activation_sites=["relu"]),
        tokenizer=TinyTokenizer(),
    )
    callable_selector = ActivationSignalExtractor(
        TinyActivationModel(),
        signal=Signal.activation(token_selector=lambda activation, inputs: activation[:, 0, :1]),
        scope=SignalScope.from_values(activation_sites=["emb"]),
        tokenizer=TinyTokenizer(),
    )

    assert unknown_site.infer_input_dim_static() is None
    assert callable_selector.infer_input_dim_static() is None


class MisleadingStaticWidthSite(nn.Module):
    def __init__(self, *, static_width=5, actual_width=3):
        super().__init__()
        self.hidden_size = static_width
        self.actual_width = actual_width
        self.forward_count = 0

    def forward(self, input_ids):
        self.forward_count += 1
        return torch.zeros(input_ids.shape[0], input_ids.shape[1], self.actual_width)


class MisleadingStaticWidthModel(nn.Module):
    def __init__(self, *, static_width=5, actual_width=3):
        super().__init__()
        self.site = MisleadingStaticWidthSite(
            static_width=static_width,
            actual_width=actual_width,
        )

    def forward(self, input_ids, attention_mask=None):
        return self.site(input_ids.float())


def test_activation_signal_extractor_detects_wrong_static_width_on_first_real_forward():
    model = MisleadingStaticWidthModel(static_width=5, actual_width=3)
    extractor = ActivationSignalExtractor(
        model,
        signal=Signal.activation(token_selector="mean"),
        scope=SignalScope.from_values(activation_sites=["site"]),
    )

    assert extractor.infer_input_dim_static() == 5
    with pytest.raises(ValueError) as exc_info:
        extractor(
            factual_inputs={"input_ids": torch.ones(2, 4)},
            alternative_inputs=None,
            requires_alternative=False,
        )

    message = str(exc_info.value)
    assert "Activation signal width mismatch" in message
    assert "expected 5" in message
    assert "width 3" in message
    assert "site (MisleadingStaticWidthSite)" in message
    assert "SignalScope.from_values" in message
    assert model.site.forward_count == 1


def test_activation_signal_extractor_prefers_configured_gradiend_width_for_validation():
    base_model = MisleadingStaticWidthModel(static_width=3, actual_width=3)

    class WrappedModel:
        def __init__(self, base):
            self.base_model = base
            self.gradiend = type("TinyGradiend", (), {"input_dim": 7})()

    extractor = ActivationSignalExtractor(
        WrappedModel(base_model),
        signal=Signal.activation(token_selector="mean"),
        scope=SignalScope.from_values(activation_sites=["site"]),
    )

    with pytest.raises(ValueError) as exc_info:
        extractor(
            factual_inputs={"input_ids": torch.ones(2, 4)},
            alternative_inputs=None,
            requires_alternative=False,
        )

    message = str(exc_info.value)
    assert "configured encoder input_dim" in message
    assert "expected 7" in message
    assert "width 3" in message
    assert "checkpoint does not match" in message


def test_activation_signal_width_validation_runs_once_after_success():
    model = MisleadingStaticWidthModel(static_width=3, actual_width=3)
    extractor = ActivationSignalExtractor(
        model,
        signal=Signal.activation(token_selector="mean"),
        scope=SignalScope.from_values(activation_sites=["site"]),
    )

    first = extractor(
        factual_inputs={"input_ids": torch.ones(2, 4)},
        alternative_inputs=None,
        requires_alternative=False,
    )
    model.site.hidden_size = 99
    second = extractor(
        factual_inputs={"input_ids": torch.ones(2, 4)},
        alternative_inputs=None,
        requires_alternative=False,
    )

    assert first.factual.shape == (3,)
    assert second.factual.shape == (3,)
    assert model.site.forward_count == 2


def test_activation_signal_extractor_pools_all_tokens_for_all_selector():
    model = TinyActivationModel()
    extractor = ActivationSignalExtractor(
        model,
        signal=Signal.activation(token_selector="all"),
        scope=SignalScope.from_values(activation_sites=["emb"]),
    )

    batch = extractor(
        factual_inputs={
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
        },
        alternative_inputs=None,
        requires_alternative=False,
    )

    assert torch.equal(batch.factual, torch.tensor([9.75, 10.75, 11.75]))


def test_activation_signal_extractor_pools_multiple_mask_tokens():
    model = TinyActivationModel()
    extractor = ActivationSignalExtractor(
        model,
        signal=Signal.activation(token_selector="mask"),
        scope=SignalScope.from_values(activation_sites=["emb"]),
        tokenizer=TinyTokenizer(),
    )

    batch = extractor(
        factual_inputs={"input_ids": torch.tensor([[9, 1, 9], [4, 9, 6]])},
        alternative_inputs=None,
        requires_alternative=False,
    )

    assert torch.equal(batch.factual, torch.tensor([27.0, 28.0, 29.0]))


def test_activation_signal_extractor_pools_multiple_cls_tokens_when_token_ids_are_available():
    model = TinyActivationModel()
    extractor = ActivationSignalExtractor(
        model,
        signal=Signal.activation(token_selector="cls"),
        scope=SignalScope.from_values(activation_sites=["emb"]),
        tokenizer=TinyTokenizer(),
    )

    batch = extractor(
        factual_inputs={"input_ids": torch.tensor([[8, 1, 8], [4, 8, 6]])},
        alternative_inputs=None,
        requires_alternative=False,
    )

    assert torch.equal(batch.factual, torch.tensor([24.0, 25.0, 26.0]))


def test_activation_signal_extractor_without_scope_uses_default_scope():
    extractor = ActivationSignalExtractor(
        TinyActivationModel(),
        signal=Signal.activation(token_selector="mean"),
    )

    assert extractor.infer_input_dim_static() == 3


def test_default_activation_scope_uses_representation_stream_not_embedding_internals():
    model = BertishActivationModel(hidden_size=4)

    assert default_activation_sites(model) == (
        "embeddings",
        "encoder.layer.0",
        "encoder.layer.1",
    )

    extractor = ActivationSignalExtractor(
        model,
        signal=Signal.activation(token_selector="mask"),
        tokenizer=TinyTokenizer(),
    )
    batch = extractor(
        factual_inputs={"input_ids": torch.tensor([[1, 9, 2], [3, 9, 4]])},
        alternative_inputs=None,
        requires_alternative=False,
    )

    assert extractor.infer_input_dim_static() == 12
    assert batch.factual.shape == (12,)


def test_semantic_layer_scope_resolves_through_supported_topology():
    model = BertishActivationModel(hidden_size=4)

    all_layers = resolve_activation_modules(model, (), scope=SignalScope.layers())
    layer_one = resolve_activation_modules(model, (), scope=SignalScope.layer(1))

    assert tuple(name for name, _module in all_layers) == (
        "encoder.layer.0",
        "encoder.layer.1",
    )
    assert tuple(name for name, _module in layer_one) == ("encoder.layer.1",)

    extractor = ActivationSignalExtractor(
        model,
        signal=Signal.activation(token_selector="mask"),
        scope=SignalScope.layers(),
        tokenizer=TinyTokenizer(),
    )

    assert extractor.infer_input_dim_static() == 8


def test_semantic_embedding_scopes_resolve_distinct_sites():
    model = BertishActivationModel(hidden_size=4)

    embeddings = resolve_activation_modules(model, (), scope=SignalScope.embeddings())
    word_embedding = resolve_activation_modules(model, (), scope=SignalScope.word_embedding())

    assert tuple(name for name, _module in embeddings) == ("embeddings",)
    assert tuple(name for name, _module in word_embedding) == ("embeddings.word_embeddings",)


def test_semantic_scope_requires_supported_topology():
    with pytest.raises(ValueError, match="SignalScope.from_values"):
        resolve_activation_modules(TinyActivationModel(), (), scope=SignalScope.layers())


def test_semantic_scope_resolves_distilbert_like_topology():
    model = DistilbertishActivationModel(hidden_size=4)

    assert default_activation_sites(model) == (
        "distilbert.embeddings",
        "distilbert.transformer.layer.0",
        "distilbert.transformer.layer.1",
    )
    assert tuple(name for name, _module in resolve_activation_modules(model, (), scope=SignalScope.layers())) == (
        "distilbert.transformer.layer.0",
        "distilbert.transformer.layer.1",
    )
    assert tuple(
        name for name, _module in resolve_activation_modules(model, (), scope=SignalScope.word_embedding())
    ) == ("distilbert.embeddings.word_embeddings",)

    extractor = ActivationSignalExtractor(
        model,
        signal=Signal.activation(token_selector="mean"),
        scope=SignalScope.layers(),
    )

    assert extractor.infer_input_dim_static() == 8


def test_semantic_scope_resolves_opt_like_topology():
    model = OPTishActivationModel(hidden_size=5)

    assert default_activation_sites(model) == (
        "model.decoder.embed_tokens",
        "model.decoder.layers.0",
        "model.decoder.layers.1",
    )
    assert tuple(name for name, _module in resolve_activation_modules(model, (), scope=SignalScope.layers())) == (
        "model.decoder.layers.0",
        "model.decoder.layers.1",
    )


def test_semantic_scope_resolves_gpt2_like_topology():
    model = GPT2ishActivationModel(hidden_size=5)

    assert default_activation_sites(model) == (
        "transformer.wte",
        "transformer.h.0",
        "transformer.h.1",
    )
    assert tuple(name for name, _module in resolve_activation_modules(model, (), scope=SignalScope.layers())) == (
        "transformer.h.0",
        "transformer.h.1",
    )
    assert tuple(
        name for name, _module in resolve_activation_modules(model, (), scope=SignalScope.word_embedding())
    ) == ("transformer.wte",)

    extractor = ActivationSignalExtractor(
        model,
        signal=Signal.activation(token_selector="mean"),
        scope=SignalScope.layers(),
    )

    assert extractor.infer_input_dim_static() == 10


def test_semantic_scope_resolves_llama_like_topology():
    model = LlamaishActivationModel(hidden_size=6)

    assert default_activation_sites(model) == (
        "model.embed_tokens",
        "model.layers.0",
        "model.layers.1",
    )
    assert tuple(name for name, _module in resolve_activation_modules(model, (), scope=SignalScope.layer(0))) == (
        "model.layers.0",
    )
    assert tuple(name for name, _module in resolve_activation_modules(model, (), scope=SignalScope.embeddings())) == (
        "model.embed_tokens",
    )

    extractor = ActivationSignalExtractor(
        model,
        signal=Signal.activation(token_selector="mean"),
        scope=SignalScope.layers(),
    )

    assert extractor.infer_input_dim_static() == 12


def test_activation_gradiend_toy_workflow_learns_counterfactual_direction():
    torch.manual_seed(0)
    extractor = ActivationSignalExtractor(
        TinyConceptActivationModel(),
        signal=Signal.activation(token_selector=1),
        scope=SignalScope.from_values(activation_sites="emb"),
    )

    rows = [
        ([1, 3, 2], [1, 7, 2], 1.0),
        ([1, 4, 2], [1, 8, 2], 1.0),
        ([1, 7, 2], [1, 3, 2], -1.0),
        ([1, 8, 2], [1, 4, 2], -1.0),
    ]
    signals = []
    labels = []
    for factual, alternative, label in rows:
        batch = extractor(
            factual_inputs=_concept_inputs(factual),
            alternative_inputs=_concept_inputs(alternative),
        )
        signals.append(batch.diff)
        labels.append(label)

    signals = torch.stack(signals)
    labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
    gradiend = GradiendModel(
        input_dim=extractor.infer_input_dim_static(),
        latent_dim=1,
        activation_encoder="tanh",
        activation_decoder="id",
    )
    optimizer = torch.optim.Adam(gradiend.parameters(), lr=0.05)
    criterion = nn.MSELoss()

    with torch.no_grad():
        decoded, encoded = gradiend(signals, return_encoded=True)
        initial_probe_loss = criterion(encoded, labels).item()
        initial_reconstruction_loss = criterion(decoded, signals).item()

    for _step in range(250):
        decoded, encoded = gradiend(signals, return_encoded=True)
        loss = criterion(encoded, labels) + 0.1 * criterion(decoded, signals)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        decoded, encoded = gradiend(signals, return_encoded=True)
        final_probe_loss = criterion(encoded, labels).item()
        final_reconstruction_loss = criterion(decoded, signals).item()

    assert extractor.infer_input_dim_static() == signals.shape[-1] == 4
    assert final_probe_loss < 0.01
    assert final_probe_loss < initial_probe_loss / 10
    assert final_reconstruction_loss < initial_reconstruction_loss
    assert torch.all(encoded[:2] > 0.9)
    assert torch.all(encoded[2:] < -0.9)


def test_signal_exports_are_lazy_available_from_top_level():
    assert gradiend.Signal.gradient().kind == "gradient"
    assert gradiend.SignalSet.gradient().ids == ("gradient",)
    assert gradiend.ActivationSignalExtractor is ActivationSignalExtractor
