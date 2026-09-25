"""Activation steering finds the decoder layers of text-only AND multimodal checkpoints.

Gemma-3 4B/27B load as conditional-generation models; their text layers are not ``model.layers.N``.
The study's registry has one template per model, and a run on gemma-3-4b-pt failed at the causal stage
with ``Activation intervention module 'model.layers.4' not found in model``.
"""

import pytest
import torch.nn as nn

from gradiend.model.modified import _module_name_candidates, _resolve_module


class _Layers(nn.Module):
    def __init__(self, n=6):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(n))


class TextOnly(nn.Module):  # Gemma3ForCausalLM / Qwen / Llama: model.layers.N
    def __init__(self):
        super().__init__()
        self.model = _Layers()


class MultimodalCurrent(nn.Module):  # transformers >= 4.52: model.language_model.layers.N (+ a vision tower)
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = _Layers()
        self.model.vision_tower = _Layers(27)


class MultimodalOld(nn.Module):  # transformers 4.50-4.51: language_model.model.layers.N
    def __init__(self):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.model = _Layers()
        self.vision_tower = _Layers(27)


def test_text_only_models_resolve_unchanged():
    m = TextOnly()
    assert _resolve_module(m, "model.layers.4") is m.model.layers[4]


def test_multimodal_current_layout_resolves_to_the_text_stack_not_the_vision_tower():
    m = MultimodalCurrent()
    assert _resolve_module(m, "model.layers.4") is m.model.language_model.layers[4]


def test_multimodal_old_layout_resolves():
    m = MultimodalOld()
    assert _resolve_module(m, "model.layers.4") is m.language_model.model.layers[4]


def test_the_name_as_given_always_wins():
    m = MultimodalCurrent()
    m.model.layers = nn.ModuleList(nn.Linear(2, 2) for _ in range(6))  # an explicit model.layers as well
    assert _resolve_module(m, "model.layers.4") is m.model.layers[4]


def test_unknown_module_still_raises_with_the_original_message_and_the_names_tried():
    with pytest.raises(KeyError, match=r"Activation intervention module 'model.layers.99' not found in model") as exc:
        _resolve_module(TextOnly(), "model.layers.99")
    assert "model.language_model.layers.99" in str(exc.value)


def test_the_error_lists_the_real_module_names_when_the_layout_is_unknown():
    class Odd(nn.Module):
        def __init__(self):
            super().__init__()
            self.thing = nn.Module()
            self.thing.decoder = _Layers()

    with pytest.raises(KeyError) as exc:
        _resolve_module(Odd(), "model.layers.4")
    assert "thing.decoder.layers.4" in str(exc.value)


def test_candidates_are_ordered_and_only_rewrite_model_prefixed_names():
    assert _module_name_candidates("model.layers.4") == [
        "model.layers.4", "model.language_model.layers.4", "language_model.model.layers.4",
    ]
    assert _module_name_candidates("transformer.h.4") == ["transformer.h.4", "language_model.transformer.h.4"]
