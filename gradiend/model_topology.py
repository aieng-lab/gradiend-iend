"""Explicit model-topology adapters for semantic signal scopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

import torch.nn as nn


@dataclass(frozen=True)
class ModelTopology:
    """Semantic module names for a supported model family."""

    model_type: str
    layers: Tuple[str, ...]
    embeddings: Tuple[str, ...] = ()
    word_embedding: Optional[str] = None
    prediction_heads: Tuple[str, ...] = ()


# Human-readable family labels used in error messages.
SUPPORTED_TOPOLOGY_FAMILIES: Tuple[str, ...] = (
    "bert-like (bert, roberta, deberta, electra, albert, mpnet, camembert, xlm-roberta, ...)",
    "distilbert (distilbert / DistilBert*)",
    "gpt2-like (gpt2, gpt_neo, gptj, bloom, falcon, ...)",
    "llama-like (llama, mistral, gemma, qwen2, phi, ...)",
    "opt (opt)",
    "gpt_neox (gpt_neox)",
)

_BERT_TYPES = frozenset(
    {
        "bert",
        "roberta",
        "deberta",
        "deberta-v2",
        "electra",
        "albert",
        "mpnet",
        "camembert",
        "xlm-roberta",
        "xlm_roberta",
        "layoutlm",
        "layoutlmv2",
        "layoutlmv3",
        "ernie",
        "fnet",
        "squeezebert",
        "nystromformer",
        "megatron-bert",
        "mobilebert",
    }
)
_DISTILBERT_TYPES = frozenset({"distilbert"})
_GPT2_TYPES = frozenset(
    {
        "gpt2",
        "gpt_neo",
        "gptj",
        "openai-gpt",
        "bloom",
        "falcon",
        "mpt",
        "gpt_bigcode",
        "codegen",
    }
)
_LLAMA_TYPES = frozenset(
    {
        "llama",
        "mistral",
        "gemma",
        "gemma2",
        "gemma3",
        "gemma3_text",
        "qwen2",
        "qwen3",
        "qwen3_5",
        "qwen3_moe",
        "qwen3_5_moe",
        "phi",
        "phi3",
        "stablelm",
        "cohere",
        "olmo",
        "olmo2",
        "granite",
        "internlm2",
    }
)
_OPT_TYPES = frozenset({"opt"})
_GPT_NEOX_TYPES = frozenset({"gpt_neox"})


def _named_modules(model: nn.Module) -> dict[str, nn.Module]:
    return dict(model.named_modules())


def _module_exists(model: nn.Module, name: str) -> bool:
    return name in _named_modules(model)


def _first_existing_prefix(model: nn.Module, prefixes: Sequence[str]) -> Optional[str]:
    modules = _named_modules(model)
    for prefix in prefixes:
        if prefix == "" or prefix in modules:
            return prefix
    return None


def _join(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}.{suffix}"


def _module_list_names(model: nn.Module, prefix: str, relative_path: str) -> Tuple[str, ...]:
    modules = _named_modules(model)
    name = _join(prefix, relative_path)
    module = modules.get(name)
    if not isinstance(module, (nn.ModuleList, nn.Sequential)):
        return ()
    return tuple(f"{name}.{index}" for index, child in enumerate(module) if isinstance(child, nn.Module))


def _config_values(model: nn.Module) -> set[str]:
    config = getattr(model, "config", None)
    values: set[str] = set()
    model_type = getattr(config, "model_type", None)
    if isinstance(model_type, str):
        values.add(model_type.lower())
    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, (list, tuple)):
        for item in architectures:
            text = str(item).lower()
            values.add(text)
            # DistilBertForMaskedLM / BertModel → also match family tokens in the class name.
            for token in text.replace("-", "_").split("_"):
                if token:
                    values.add(token)
    return values


def _config_model_type(model: nn.Module) -> Optional[str]:
    config = getattr(model, "config", None)
    model_type = getattr(config, "model_type", None)
    return model_type if isinstance(model_type, str) else None


def _config_architectures(model: nn.Module) -> Tuple[str, ...]:
    config = getattr(model, "config", None)
    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, (list, tuple)):
        return tuple(str(item) for item in architectures)
    return ()


def _prediction_heads(model: nn.Module, names: Sequence[str]) -> Tuple[str, ...]:
    return tuple(name for name in names if _module_exists(model, name))


def _pick_label(values: set[str], candidates: frozenset[str], fallback: str) -> str:
    matched = values & candidates
    if matched:
        return next(iter(matched))
    return fallback


def _bert_topology(model: nn.Module, model_type: str) -> Optional[ModelTopology]:
    prefix = _first_existing_prefix(
        model,
        ("bert", "roberta", "deberta", "electra", "albert", "mpnet", "camembert", "xlm_roberta", ""),
    )
    if prefix is None:
        return None
    layers = _module_list_names(model, prefix, "encoder.layer")
    if not layers:
        return None

    embeddings = (_join(prefix, "embeddings"),) if _module_exists(model, _join(prefix, "embeddings")) else ()
    word_embedding = _join(prefix, "embeddings.word_embeddings")
    if not _module_exists(model, word_embedding):
        word_embedding = None

    heads = _prediction_heads(model, ("cls", "lm_head", "classifier", "qa_outputs"))
    return ModelTopology(
        model_type=model_type,
        layers=layers,
        embeddings=embeddings,
        word_embedding=word_embedding,
        prediction_heads=heads,
    )


def _distilbert_topology(model: nn.Module, model_type: str) -> Optional[ModelTopology]:
    """DistilBERT uses ``transformer.layer`` (not BERT's ``encoder.layer``)."""
    prefix = _first_existing_prefix(model, ("distilbert", ""))
    if prefix is None:
        return None
    layers = _module_list_names(model, prefix, "transformer.layer")
    if not layers:
        return None

    embeddings = (_join(prefix, "embeddings"),) if _module_exists(model, _join(prefix, "embeddings")) else ()
    word_embedding = _join(prefix, "embeddings.word_embeddings")
    if not _module_exists(model, word_embedding):
        word_embedding = None

    heads = _prediction_heads(
        model,
        ("vocab_transform", "vocab_projector", "classifier", "pre_classifier", "qa_outputs"),
    )
    return ModelTopology(
        model_type=model_type,
        layers=layers,
        embeddings=embeddings,
        word_embedding=word_embedding,
        prediction_heads=heads,
    )


def _gpt2_topology(model: nn.Module, model_type: str) -> Optional[ModelTopology]:
    prefix = _first_existing_prefix(model, ("transformer", ""))
    if prefix is None:
        return None
    layers = _module_list_names(model, prefix, "h")
    if not layers:
        # MPT-style blocks under the same transformer root.
        layers = _module_list_names(model, prefix, "blocks")
    if not layers:
        return None

    word_embedding = _join(prefix, "wte")
    if not _module_exists(model, word_embedding):
        word_embedding = _join(prefix, "word_embeddings")
    if not _module_exists(model, word_embedding):
        word_embedding = None

    heads = _prediction_heads(model, ("lm_head", "score"))
    embeddings = (word_embedding,) if word_embedding is not None else ()
    return ModelTopology(
        model_type=model_type,
        layers=layers,
        embeddings=embeddings,
        word_embedding=word_embedding,
        prediction_heads=heads,
    )


def _llama_topology(model: nn.Module, model_type: str) -> Optional[ModelTopology]:
    prefix = _first_existing_prefix(model, ("model", "language_model.model", ""))
    if prefix is None:
        return None
    layers = _module_list_names(model, prefix, "layers")
    if not layers:
        return None

    word_embedding = _join(prefix, "embed_tokens")
    if not _module_exists(model, word_embedding):
        word_embedding = None

    heads = _prediction_heads(model, ("lm_head", "score"))
    embeddings = (word_embedding,) if word_embedding is not None else ()
    return ModelTopology(
        model_type=model_type,
        layers=layers,
        embeddings=embeddings,
        word_embedding=word_embedding,
        prediction_heads=heads,
    )


def _opt_topology(model: nn.Module, model_type: str) -> Optional[ModelTopology]:
    prefix = _first_existing_prefix(model, ("model.decoder", "decoder", "model", ""))
    if prefix is None:
        return None
    layers = _module_list_names(model, prefix, "layers")
    if not layers:
        return None

    word_embedding = _join(prefix, "embed_tokens")
    if not _module_exists(model, word_embedding):
        # OPTForCausalLM often keeps embeddings on the top-level model.
        for candidate in ("model.decoder.embed_tokens", "decoder.embed_tokens", "embed_tokens"):
            if _module_exists(model, candidate):
                word_embedding = candidate
                break
        else:
            word_embedding = None

    heads = _prediction_heads(model, ("lm_head", "score"))
    embeddings = (word_embedding,) if word_embedding is not None else ()
    return ModelTopology(
        model_type=model_type,
        layers=layers,
        embeddings=embeddings,
        word_embedding=word_embedding,
        prediction_heads=heads,
    )


def _gpt_neox_topology(model: nn.Module, model_type: str) -> Optional[ModelTopology]:
    prefix = _first_existing_prefix(model, ("gpt_neox", ""))
    if prefix is None:
        return None
    layers = _module_list_names(model, prefix, "layers")
    if not layers:
        return None

    word_embedding = _join(prefix, "embed_in")
    if not _module_exists(model, word_embedding):
        word_embedding = None

    heads = _prediction_heads(model, ("embed_out", "lm_head", "score"))
    embeddings = (word_embedding,) if word_embedding is not None else ()
    return ModelTopology(
        model_type=model_type,
        layers=layers,
        embeddings=embeddings,
        word_embedding=word_embedding,
        prediction_heads=heads,
    )


_TopologyBuilder = Callable[[nn.Module, str], Optional[ModelTopology]]

# Ordered structural probes used when model_type is missing/unknown.
# DistilBERT must be tried before GPT-2: both use a ``transformer`` root, but
# DistilBERT stores blocks at ``transformer.layer`` while GPT-2 uses ``transformer.h``.
_STRUCTURAL_ADAPTERS: Tuple[Tuple[str, _TopologyBuilder], ...] = (
    ("bert", _bert_topology),
    ("distilbert", _distilbert_topology),
    ("gpt2", _gpt2_topology),
    ("llama", _llama_topology),
    ("opt", _opt_topology),
    ("gpt_neox", _gpt_neox_topology),
)


def _typed_adapter(values: set[str]) -> Optional[Tuple[str, _TopologyBuilder, frozenset[str]]]:
    if values & _BERT_TYPES:
        return ("bert", _bert_topology, _BERT_TYPES)
    if values & _DISTILBERT_TYPES or any("distilbert" in value for value in values):
        return ("distilbert", _distilbert_topology, _DISTILBERT_TYPES)
    if values & _GPT2_TYPES:
        return ("gpt2", _gpt2_topology, _GPT2_TYPES)
    if values & _LLAMA_TYPES:
        return ("llama", _llama_topology, _LLAMA_TYPES)
    if values & _OPT_TYPES:
        return ("opt", _opt_topology, _OPT_TYPES)
    if values & _GPT_NEOX_TYPES:
        return ("gpt_neox", _gpt_neox_topology, _GPT_NEOX_TYPES)
    return None


def infer_model_topology(model: nn.Module) -> Optional[ModelTopology]:
    """Return semantic topology for supported Hugging Face families.

    Dispatch prefers ``config.model_type`` / ``architectures``, then falls back to
    structural probing so common layouts still resolve when the type string is
    missing or custom.
    """

    values = _config_values(model)
    typed = _typed_adapter(values)
    if typed is not None:
        label, builder, candidates = typed
        topology = builder(model, _pick_label(values, candidates, label))
        if topology is not None:
            return topology

    # Structural fallback (and second chance if the typed adapter's paths missed).
    preferred = typed[0] if typed is not None else None
    for label, builder in _STRUCTURAL_ADAPTERS:
        if label == preferred:
            continue
        topology = builder(model, _config_model_type(model) or label)
        if topology is not None:
            return topology
    return None


def describe_model_for_topology_error(model: nn.Module) -> str:
    """Build an actionable error snippet when semantic scopes cannot be resolved."""
    model_type = _config_model_type(model)
    architectures = _config_architectures(model)
    top_level = tuple(name for name, _module in model.named_children())
    module_lists = tuple(
        name
        for name, module in model.named_modules()
        if name and isinstance(module, (nn.ModuleList, nn.Sequential)) and name.count(".") <= 3
    )

    lines = [
        f"model_type={model_type!r}, architectures={list(architectures)!r}.",
        "Semantic scopes such as SignalScope.layers() / .embeddings() need a known "
        "model topology adapter.",
        "Supported families:",
    ]
    lines.extend(f"  - {family}" for family in SUPPORTED_TOPOLOGY_FAMILIES)
    if top_level:
        lines.append(f"Top-level modules on this model: {list(top_level)!r}.")
    if module_lists:
        example = module_lists[0]
        lines.append(
            "Pass an explicit site list instead, for example:\n"
            f'  signal_scope=SignalScope.from_values(activation_sites=["{example}.*"])\n'
            "or add/extend an adapter in gradiend/model_topology.py for this architecture."
        )
    else:
        lines.append(
            "Pass an explicit site list, for example:\n"
            '  signal_scope=SignalScope.from_values(activation_sites=["path.to.module"])\n'
            "or add/extend an adapter in gradiend/model_topology.py for this architecture."
        )
    return "\n".join(lines)
