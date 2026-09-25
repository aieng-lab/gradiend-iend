"""Model-topology adapters resolve real (tiny) Hugging Face models, with and without config."""

import copy

import pytest
import torch

transformers = pytest.importorskip("transformers")

from gradiend.model_topology import describe_model_for_topology_error, infer_model_topology  # noqa: E402

_COMMON = dict(hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32, vocab_size=50)

_BUILDERS = {
    "bert": lambda: transformers.BertForMaskedLM(transformers.BertConfig(**_COMMON)),
    "distilbert": lambda: transformers.DistilBertForMaskedLM(
        transformers.DistilBertConfig(dim=16, n_layers=2, n_heads=2, hidden_dim=32, vocab_size=50)
    ),
    "gpt2": lambda: transformers.GPT2LMHeadModel(
        transformers.GPT2Config(n_embd=16, n_layer=2, n_head=2, vocab_size=50)
    ),
    "llama": lambda: transformers.LlamaForCausalLM(transformers.LlamaConfig(**_COMMON, num_key_value_heads=2)),
    "opt": lambda: transformers.OPTForCausalLM(
        transformers.OPTConfig(
            hidden_size=16, num_hidden_layers=2, ffn_dim=32, num_attention_heads=2,
            vocab_size=50, word_embed_proj_dim=16,
        )
    ),
    "gpt_neox": lambda: transformers.GPTNeoXForCausalLM(transformers.GPTNeoXConfig(**_COMMON)),
}

# family -> (first layer, embeddings, word embedding, prediction heads)
_EXPECTED = {
    "bert": ("bert.encoder.layer.0", ("bert.embeddings",), "bert.embeddings.word_embeddings", ("cls",)),
    "distilbert": (
        "distilbert.transformer.layer.0",
        ("distilbert.embeddings",),
        "distilbert.embeddings.word_embeddings",
        ("vocab_transform", "vocab_projector"),
    ),
    "gpt2": ("transformer.h.0", ("transformer.wte",), "transformer.wte", ("lm_head",)),
    "llama": ("model.layers.0", ("model.embed_tokens",), "model.embed_tokens", ("lm_head",)),
    "opt": (
        "model.decoder.layers.0",
        ("model.decoder.embed_tokens",),
        "model.decoder.embed_tokens",
        ("lm_head",),
    ),
    "gpt_neox": ("gpt_neox.layers.0", ("gpt_neox.embed_in",), "gpt_neox.embed_in", ("embed_out",)),
}


@pytest.mark.parametrize("family", sorted(_BUILDERS))
def test_family_topology(family):
    topology = infer_model_topology(_BUILDERS[family]())
    first_layer, embeddings, word_embedding, heads = _EXPECTED[family]
    assert topology.model_type == family
    assert len(topology.layers) == 2
    assert topology.layers[0] == first_layer
    assert topology.embeddings == embeddings
    assert topology.word_embedding == word_embedding
    assert topology.prediction_heads == heads


@pytest.mark.parametrize("family", sorted(_BUILDERS))
def test_structural_fallback_without_config_matches_typed_dispatch(family):
    typed = infer_model_topology(_BUILDERS[family]())
    model = _BUILDERS[family]()
    model.config = None
    fallback = infer_model_topology(model)
    assert fallback is not None
    assert (fallback.layers, fallback.embeddings, fallback.word_embedding, fallback.prediction_heads) == (
        typed.layers,
        typed.embeddings,
        typed.word_embedding,
        typed.prediction_heads,
    )


def test_unknown_model_has_no_topology_and_error_lists_supported_families():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    assert infer_model_topology(model) is None
    message = describe_model_for_topology_error(copy.deepcopy(model))
    assert "Supported families" in message and "bert-like" in message
    assert "SignalScope.from_values(activation_sites=" in message
