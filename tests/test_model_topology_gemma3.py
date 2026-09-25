"""Unit coverage for Gemma 3 text and multimodal model topologies."""

from fnmatch import fnmatch

import torch.nn as nn

from gradiend.model_topology import infer_model_topology
from gradiend.signal_space import gradient_params_from_selector, resolve_activation_modules
from gradiend.trainer.core.signals import SignalScope


class _Gemma3Layer(nn.Module):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(hidden_size)


class _Gemma3TextModel(nn.Module):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.config = type(
            "Gemma3TextConfig",
            (),
            {"model_type": "gemma3_text", "architectures": ["Gemma3TextModel"]},
        )()
        self.embed_tokens = nn.Embedding(11, hidden_size)
        self.layers = nn.ModuleList([_Gemma3Layer(hidden_size), _Gemma3Layer(hidden_size)])
        self.norm = nn.LayerNorm(hidden_size)


class _Gemma3ForCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.config = type(
            "Gemma3TextConfig",
            (),
            {"model_type": "gemma3_text", "architectures": ["Gemma3ForCausalLM"]},
        )()
        self.model = _Gemma3TextModel(hidden_size)
        self.lm_head = nn.Linear(hidden_size, 11, bias=False)


class _Gemma3MultimodalModel(nn.Module):
    """Newer Transformers layout: the multimodal body owns the text model."""

    def __init__(self, hidden_size: int = 4):
        super().__init__()
        self.vision_tower = nn.Sequential(nn.Linear(hidden_size, hidden_size))
        self.multi_modal_projector = nn.Linear(hidden_size, hidden_size)
        self.language_model = _Gemma3TextModel(hidden_size)


class _Gemma3ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        *,
        bare_text_child: bool = False,
        include_architectures: bool = True,
        nested_multimodal_body: bool = False,
        hidden_size: int = 4,
    ):
        super().__init__()
        config_values = {"model_type": "gemma3"}
        if include_architectures:
            config_values["architectures"] = ["Gemma3ForConditionalGeneration"]
        self.config = type(
            "Gemma3Config",
            (),
            config_values,
        )()
        if nested_multimodal_body:
            self.model = _Gemma3MultimodalModel(hidden_size)
            self.lm_head = nn.Linear(hidden_size, 11, bias=False)
        else:
            self.vision_tower = nn.Sequential(nn.Linear(hidden_size, hidden_size))
            self.multi_modal_projector = nn.Linear(hidden_size, hidden_size)
            self.language_model = (
                _Gemma3TextModel(hidden_size)
                if bare_text_child
                else _Gemma3ForCausalLM(hidden_size)
            )


def _assert_text_scopes(model: nn.Module, prefix: str) -> None:
    topology = infer_model_topology(model)
    assert topology is not None
    assert topology.layers == (f"{prefix}.layers.0", f"{prefix}.layers.1")
    assert topology.word_embedding == f"{prefix}.embed_tokens"
    assert topology.embeddings == (f"{prefix}.embed_tokens",)

    resolved = resolve_activation_modules(model, (), scope=SignalScope.layers())
    assert tuple(name for name, _module in resolved) == topology.layers
    embeddings = resolve_activation_modules(model, (), scope=SignalScope.embeddings())
    assert tuple(name for name, _module in embeddings) == topology.embeddings

    patterns = gradient_params_from_selector(model, ("layers", None))
    assert patterns == tuple(f"{name}.*" for name in topology.layers)
    matched = {
        name
        for name, _parameter in model.named_parameters()
        if any(fnmatch(name, pattern) for pattern in patterns)
    }
    assert matched
    assert all(name.startswith(f"{prefix}.layers.") for name in matched)
    assert not any(name.startswith("vision_tower.") for name in matched)
    assert not any(name.startswith("multi_modal_projector.") for name in matched)


def test_gemma3_text_only_topology_uses_llama_semantics():
    _assert_text_scopes(_Gemma3ForCausalLM(), "model")


def test_gemma3_multimodal_wrapper_composes_causal_language_model_topology():
    model = _Gemma3ForConditionalGeneration()
    _assert_text_scopes(model, "language_model.model")
    assert infer_model_topology(model).prediction_heads == ("language_model.lm_head",)


def test_gemma3_multimodal_wrapper_composes_bare_text_model_topology():
    _assert_text_scopes(
        _Gemma3ForConditionalGeneration(bare_text_child=True),
        "language_model",
    )


def test_gemma3_multimodal_body_composes_bare_text_model_and_outer_head():
    model = _Gemma3ForConditionalGeneration(nested_multimodal_body=True)
    _assert_text_scopes(model, "model.language_model")
    assert infer_model_topology(model).prediction_heads == ("lm_head",)


def test_gemma3_multimodal_wrapper_does_not_require_architectures_metadata():
    _assert_text_scopes(
        _Gemma3ForConditionalGeneration(include_architectures=False),
        "language_model.model",
    )
