"""Explicit model-topology adapters for semantic signal scopes.

A *topology* names the modules of a supported Hugging Face model family that
semantic scopes such as ``SignalScope.layers()`` / ``.embeddings()`` refer to.
Every family is described declaratively by a :class:`_FamilySpec`; adding an
architecture means adding one spec to :data:`_FAMILIES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, FrozenSet, Optional, Sequence, Tuple

import torch.nn as nn


@dataclass(frozen=True)
class ModelTopology:
    """Semantic module names for a supported model family."""

    model_type: str
    layers: Tuple[str, ...]
    embeddings: Tuple[str, ...] = ()
    word_embedding: Optional[str] = None
    prediction_heads: Tuple[str, ...] = ()


@dataclass(frozen=True)
class _FamilySpec:
    """Where a family keeps its modules, relative to its backbone prefix."""

    label: str
    #: ``config.model_type`` / architecture tokens dispatched to this family.
    types: FrozenSet[str]
    #: Candidate backbone prefixes, tried in order (``""`` = the model itself).
    prefixes: Tuple[str, ...]
    #: Candidate module-list paths holding the transformer blocks, tried in order.
    layer_paths: Tuple[str, ...]
    #: Candidate word-embedding paths relative to the prefix, tried in order.
    word_embeddings: Tuple[str, ...]
    #: Absolute word-embedding paths tried when no relative path exists.
    word_embedding_fallbacks: Tuple[str, ...] = ()
    #: Relative path of a combined embedding stream (BERT-style); when ``None`` the
    #: word embedding is the only embedding site.
    combined_embeddings: Optional[str] = None
    #: Top-level prediction-head module names present on the model, in order.
    heads: Tuple[str, ...] = ()


# Human-readable family labels used in error messages.
SUPPORTED_TOPOLOGY_FAMILIES: Tuple[str, ...] = (
    "gemma3 multimodal wrapper (Gemma3ForConditionalGeneration)",
    "bert-like (bert, roberta, deberta, electra, albert, mpnet, camembert, xlm-roberta, ...)",
    "distilbert (distilbert / DistilBert*)",
    "gpt2-like (gpt2, gpt_neo, gptj, bloom, falcon, ...)",
    "llama-like (llama, mistral, gemma, qwen2, phi, ...)",
    "opt (opt)",
    "gpt_neox (gpt_neox)",
)

# Order matters for the structural fallback: DistilBERT must be tried before GPT-2.
# Both use a ``transformer`` root, but DistilBERT stores blocks at ``transformer.layer``
# while GPT-2 uses ``transformer.h``.
_FAMILIES: Tuple[_FamilySpec, ...] = (
    _FamilySpec(
        label="bert",
        types=frozenset(
            {
                "bert", "roberta", "deberta", "deberta-v2", "electra", "albert", "mpnet",
                "camembert", "xlm-roberta", "xlm_roberta", "layoutlm", "layoutlmv2",
                "layoutlmv3", "ernie", "fnet", "squeezebert", "nystromformer",
                "megatron-bert", "mobilebert",
            }
        ),
        prefixes=("bert", "roberta", "deberta", "electra", "albert", "mpnet", "camembert", "xlm_roberta", ""),
        layer_paths=("encoder.layer",),
        word_embeddings=("embeddings.word_embeddings",),
        combined_embeddings="embeddings",
        heads=("cls", "lm_head", "classifier", "qa_outputs"),
    ),
    _FamilySpec(
        label="distilbert",
        types=frozenset({"distilbert"}),
        prefixes=("distilbert", ""),
        layer_paths=("transformer.layer",),
        word_embeddings=("embeddings.word_embeddings",),
        combined_embeddings="embeddings",
        heads=("vocab_transform", "vocab_projector", "classifier", "pre_classifier", "qa_outputs"),
    ),
    _FamilySpec(
        label="gpt2",
        types=frozenset(
            {"gpt2", "gpt_neo", "gptj", "openai-gpt", "bloom", "falcon", "mpt", "gpt_bigcode", "codegen"}
        ),
        prefixes=("transformer", ""),
        layer_paths=("h", "blocks"),  # ``blocks``: MPT-style
        word_embeddings=("wte", "word_embeddings"),
        heads=("lm_head", "score"),
    ),
    _FamilySpec(
        label="llama",
        types=frozenset(
            {
                "llama", "mistral", "gemma", "gemma2", "gemma3", "gemma3_text", "qwen2", "qwen3",
                "qwen3_5", "qwen3_moe", "qwen3_5_moe", "phi", "phi3", "stablelm", "cohere", "olmo",
                "olmo2", "granite", "internlm2",
            }
        ),
        prefixes=("model", ""),
        layer_paths=("layers",),
        word_embeddings=("embed_tokens",),
        heads=("lm_head", "score"),
    ),
    _FamilySpec(
        label="opt",
        types=frozenset({"opt"}),
        prefixes=("model.decoder", "decoder", "model", ""),
        layer_paths=("layers",),
        word_embeddings=("embed_tokens",),
        # OPTForCausalLM often keeps embeddings on the top-level model.
        word_embedding_fallbacks=("model.decoder.embed_tokens", "decoder.embed_tokens", "embed_tokens"),
        heads=("lm_head", "score"),
    ),
    _FamilySpec(
        label="gpt_neox",
        types=frozenset({"gpt_neox"}),
        prefixes=("gpt_neox", ""),
        layer_paths=("layers",),
        word_embeddings=("embed_in",),
        heads=("embed_out", "lm_head", "score"),
    ),
)

_GEMMA3_MULTIMODAL_CONFIG_VALUES = frozenset({"gemma3", "gemma3forconditionalgeneration"})


def _join(prefix: str, suffix: str) -> str:
    if not prefix:
        return suffix
    if not suffix:
        return prefix
    return f"{prefix}.{suffix}"


def _module_list_names(modules: Dict[str, nn.Module], name: str) -> Tuple[str, ...]:
    module = modules.get(name)
    if not isinstance(module, (nn.ModuleList, nn.Sequential)):
        return ()
    return tuple(f"{name}.{index}" for index, child in enumerate(module) if isinstance(child, nn.Module))


def _config_values(model: nn.Module) -> set:
    config = getattr(model, "config", None)
    values: set = set()
    model_type = getattr(config, "model_type", None)
    if isinstance(model_type, str):
        values.add(model_type.lower())
    architectures = getattr(config, "architectures", None)
    if isinstance(architectures, (list, tuple)):
        for item in architectures:
            text = str(item).lower()
            values.add(text)
            # DistilBertForMaskedLM / BertModel -> also match family tokens in the class name.
            values.update(token for token in text.replace("-", "_").split("_") if token)
    return values


def _config_model_type(model: nn.Module) -> Optional[str]:
    model_type = getattr(getattr(model, "config", None), "model_type", None)
    return model_type if isinstance(model_type, str) else None


def _config_architectures(model: nn.Module) -> Tuple[str, ...]:
    architectures = getattr(getattr(model, "config", None), "architectures", None)
    if isinstance(architectures, (list, tuple)):
        return tuple(str(item) for item in architectures)
    return ()


def _existing(modules: Dict[str, nn.Module], names: Sequence[str]) -> Tuple[str, ...]:
    return tuple(name for name in names if name in modules)


def _pick_label(values: set, candidates: FrozenSet[str], fallback: str) -> str:
    matched = values & candidates
    return sorted(matched)[0] if matched else fallback


def _prefix_topology(topology: ModelTopology, prefix: str, *, model_type: str) -> ModelTopology:
    """Namespace a child model's semantic topology inside its parent wrapper."""
    return ModelTopology(
        model_type=model_type,
        layers=tuple(_join(prefix, name) for name in topology.layers),
        embeddings=tuple(_join(prefix, name) for name in topology.embeddings),
        word_embedding=(
            _join(prefix, topology.word_embedding) if topology.word_embedding is not None else None
        ),
        prediction_heads=tuple(_join(prefix, name) for name in topology.prediction_heads),
    )


def _build_from_spec(
    model: nn.Module, modules: Dict[str, nn.Module], spec: _FamilySpec, model_type: str
) -> Optional[ModelTopology]:
    """Resolve one family's topology on ``model`` (``None`` when its layout is absent)."""
    prefix = next((p for p in spec.prefixes if p == "" or p in modules), None)
    if prefix is None:
        return None
    layers: Tuple[str, ...] = ()
    for path in spec.layer_paths:
        layers = _module_list_names(modules, _join(prefix, path))
        if layers:
            break
    if not layers:
        return None

    word_embedding = next(
        (name for name in (_join(prefix, rel) for rel in spec.word_embeddings) if name in modules),
        None,
    )
    if word_embedding is None:
        word_embedding = next((name for name in spec.word_embedding_fallbacks if name in modules), None)

    if spec.combined_embeddings is not None:
        combined = _join(prefix, spec.combined_embeddings)
        embeddings = (combined,) if combined in modules else ()
    else:
        embeddings = (word_embedding,) if word_embedding is not None else ()

    return ModelTopology(
        model_type=model_type,
        layers=layers,
        embeddings=embeddings,
        word_embedding=word_embedding,
        prediction_heads=_existing(modules, spec.heads),
    )


def _gemma3_multimodal_topology(model: nn.Module, model_type: str) -> Optional[ModelTopology]:
    """Compose Gemma 3's multimodal wrapper with its text model topology.

    ``Gemma3ForConditionalGeneration`` contains a causal language model under
    ``language_model`` (older Transformers) or under its multimodal ``model``
    body (newer Transformers). Semantic text scopes must describe that
    contained language model, not the wrapper's unrelated vision modules.
    Infer the child normally so both a ``Gemma3ForCausalLM`` child
    (``model.layers``) and a bare text-model child (``layers``) are supported.
    """
    if not (_config_values(model) & _GEMMA3_MULTIMODAL_CONFIG_VALUES):
        return None

    language_model = getattr(model, "language_model", None)
    language_model_prefix = "language_model"
    if not isinstance(language_model, nn.Module):
        language_model = getattr(getattr(model, "model", None), "language_model", None)
        language_model_prefix = "model.language_model"
    if not isinstance(language_model, nn.Module):
        return None
    topology = infer_model_topology(language_model)
    if topology is None:
        return None
    nested = _prefix_topology(topology, language_model_prefix, model_type=model_type)
    # Depending on the Transformers version, the LM head belongs either to the
    # contained causal LM or to the outer conditional-generation wrapper.
    outer_heads = _existing(dict(model.named_modules()), ("lm_head", "score"))
    return ModelTopology(
        model_type=nested.model_type,
        layers=nested.layers,
        embeddings=nested.embeddings,
        word_embedding=nested.word_embedding,
        prediction_heads=tuple(dict.fromkeys((*nested.prediction_heads, *outer_heads))),
    )


def _typed_family(values: set) -> Optional[_FamilySpec]:
    for spec in _FAMILIES:
        if values & spec.types:
            return spec
        # ``DistilBertForMaskedLM`` etc.: match the family name inside architecture strings.
        if spec.label == "distilbert" and any("distilbert" in value for value in values):
            return spec
    return None


def infer_model_topology(model: nn.Module) -> Optional[ModelTopology]:
    """Return semantic topology for supported Hugging Face families.

    Dispatch prefers ``config.model_type`` / ``architectures``, then falls back to
    structural probing so common layouts still resolve when the type string is
    missing or custom.
    """
    values = _config_values(model)

    # Wrapper composition precedes family dispatch: the wrapper's config says
    # ``gemma3``, while the semantic text modules belong to its contained LM.
    wrapped = _gemma3_multimodal_topology(model, _config_model_type(model) or "gemma3")
    if wrapped is not None:
        return wrapped

    modules = dict(model.named_modules())
    typed = _typed_family(values)
    if typed is not None:
        topology = _build_from_spec(model, modules, typed, _pick_label(values, typed.types, typed.label))
        if topology is not None:
            return topology

    # Structural fallback (and second chance if the typed adapter's paths missed).
    for spec in _FAMILIES:
        if spec is typed:
            continue
        topology = _build_from_spec(model, modules, spec, _config_model_type(model) or spec.label)
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
            "or add a family to _FAMILIES in gradiend/model_topology.py for this architecture."
        )
    else:
        lines.append(
            "Pass an explicit site list, for example:\n"
            '  signal_scope=SignalScope.from_values(activation_sites=["path.to.module"])\n'
            "or add a family to _FAMILIES in gradiend/model_topology.py for this architecture."
        )
    return "\n".join(lines)
