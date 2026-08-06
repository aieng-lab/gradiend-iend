"""Source/target keywords and decoder-eval contracts.

Training vs decoder evaluation
------------------------------

* **Training** uses ``TrainingArguments.source`` for which gradients feed the encoder

  (``factual``, ``alternative``, ``diff``, or ``both``).

* ``source="both"`` alternates the encoder pole **per training batch** (factual on even
  batch indices, alternative on odd). Internally each batch is compiled to the
  existing ``factual`` + ``diff`` path by optionally swapping factual/alternative
  (and inverting the label) so target is always ``input − opposite``.

* **Encoder evaluation** (``target=None``) expands each base example to **both poles**
  for every training ``source``. That way one-pole training data (only ``A→B`` rows)
  still yields labels ``+1`` and ``-1`` for correlation, whether training used
  ``factual``, ``alternative``, ``diff``, or ``both``.

* **Decoder rewrite/intervention** uses ``model.source`` and ``model.target``
  (persisted in ``gradiend_context.json``) to

  pick the default ``feature_factor`` sign per class. It is set once before training
  via :func:`sync_model_source_target_from_training_args` and must not be overwritten
  when loading a finished checkpoint for analysis.

Feature-factor sign (strengthen class ``C``)
--------------------------------------------
The intervention is ``base + learning_rate * decoder(feature_factor)`` for
gradient-space rewrites and ``activation + learning_rate * decoder(feature_factor)``
for ACTIEND hooks. Learning rate is never negated by source.

Gradient-space GRADIEND rewrites move model weights with respect to loss
gradients. To strengthen a class, the rewrite follows the opposite of the
factual/diff loss-gradient direction:

+------------------+-------------------------------+
| ``model.source`` | gradient ``ff`` for class C |
+==================+===============================+
| factual, diff, both | ``-encoding_direction[C]`` |
| alternative      | ``+encoding_direction[C]``    |
+------------------+-------------------------------+

``both`` matches ``factual`` because training compiles each batch to the
factual/diff geometry via optional fac↔alt swap.

Activation-space ACTIEND hooks add decoded activation-space displacements
directly. For the meaningful steering target ``target="diff"``, this is the
activation analogue of moving opposite a loss gradient. Therefore ACTIEND uses
the negated gradient-rewrite feature factor:

``activation_ff = -gradient_ff``

Equivalently:

+------------------+-------------------------------+
| ``model.source`` | activation ``ff`` for class C |
+==================+===============================+
| factual, diff, both | ``+encoding_direction[C]`` |
| alternative      | ``-encoding_direction[C]``    |
+------------------+-------------------------------+

ACTIEND intervention feature-factor derivation intentionally rejects
``target!="diff"`` for now. Raw factual or alternative activations are not
well-defined additive steering directions in this API, so accepting them would
make the sign look more certain than the method actually is.

``encoding_direction`` is ``model.feature_class_encoding_direction`` (from training pair:
``pair[0] -> +1``, ``pair[1] -> -1``).

Decoder grid / plots
--------------------

* Strengthen ``target_class="3SG"`` evaluates only ``ff = derive(3SG)``.
* Summary and probability-shift plots use that same ``ff`` (not another class's sign).

Cross-encoding heatmaps (anchor-aligned)
--------------------------------------
Column alignment (``factual`` vs ``counterfactual``) must match how the model was
trained. When ``model.source="alternative"`` but columns are factual-aligned (or
the reverse), encoded values are multiplied by ``-1`` — same XOR rule as decoder
``feature_factor`` (see :func:`encoding_view_sign_for_source`).
"""

from __future__ import annotations

from typing import Any

SOURCE_TARGET_KEYWORDS: frozenset[str] = frozenset({"factual", "alternative", "diff", "both"})


def validate_source_target(name: str, value: object) -> str:
    """Validate a source or target keyword."""
    if not isinstance(value, str):
        raise TypeError(f"{name} must be str, got {type(value).__name__}")
    if value not in SOURCE_TARGET_KEYWORDS:
        raise ValueError(
            f"{name} must be one of {sorted(SOURCE_TARGET_KEYWORDS)!r}, got {value!r}"
        )
    return value


def validate_source_target_combination(source: str, target: object) -> None:
    """Reject unsupported source/target pairs.

    ``source="both"`` requires ``target="diff"`` during training. Encoder-only
    evaluation may pass ``target=None`` (no reconstruction target).
    """
    source = validate_source_target("source", source)
    if source != "both":
        return
    if target is None:
        return
    target = validate_source_target("target", target)
    if target != "diff":
        raise ValueError(
            "source='both' requires target='diff' "
            f"(or target=None for encoder-only evaluation), got target={target!r}"
        )


def gradient_feature_factor_from_encoding_direction(direction: float, source: str) -> float:
    """Map a class encoding direction to a gradient-space rewrite feature factor."""
    validate_source_target("source", source)
    if source == "alternative":
        return float(direction)
    # factual, diff, and both (compiled to factual/diff) share the same sign.
    return float(-direction)


def feature_factor_from_encoding_direction(direction: float, source: str) -> float:
    """Backward-compatible gradient-space feature-factor resolver.

    Historically this public helper described GRADIEND weight rewrites. Keep
    that behavior unchanged; use
    :func:`intervention_feature_factor_from_encoding_direction` when the model's
    signal space may be activation-space ACTIEND.
    """
    return gradient_feature_factor_from_encoding_direction(direction, source)


def activation_feature_factor_from_encoding_direction(
    direction: float,
    source: str,
    target: str = "diff",
) -> float:
    """Map a class encoding direction to an ACTIEND activation intervention factor.

    ACTIEND decoders emit activation-space targets, not loss gradients. For the
    steering target that matters in practice, ``target="diff"``, the decoded
    vector is an activation displacement that should be added directly. This is
    the opposite of gradient-space weight rewrites, where strengthening a class
    means moving against the loss-gradient direction. Keep this as the explicit
    invariant: ACTIEND feature factor = ``-gradient_feature_factor``.

    ``target`` must be ``"diff"``. Raw factual or alternative activation targets
    are not yet supported as additive ACTIEND steering directions.
    """
    validate_source_target("source", source)
    target = validate_source_target("target", target)
    if target != "diff":
        raise ValueError(
            "ACTIEND activation interventions currently require target='diff'. "
            "Raw factual/alternative activation targets are not well-defined additive steering directions."
        )
    return -gradient_feature_factor_from_encoding_direction(direction, source)


def intervention_feature_factor_from_encoding_direction(
    direction: float,
    source: str,
    target: str = "diff",
    *,
    signal_kind: str = "gradient",
) -> float:
    """Map a class direction to the decoder feature factor for an intervention.

    ``signal_kind`` is the measured signal space, not the location where it was
    measured. ``SignalScope`` decides locations such as layers; it must not
    influence this sign. Gradient-space GRADIEND uses the historical rewrite
    convention. Activation-space ACTIEND uses direct activation-displacement
    semantics.
    """
    kind = str(signal_kind or "gradient").strip().lower()
    if kind == "activation":
        return activation_feature_factor_from_encoding_direction(direction, source, target)
    return gradient_feature_factor_from_encoding_direction(direction, source)


def resolve_model_signal_kind(model: Any, trainer: Any = None, *, default: str = "gradient") -> str:
    """Return the model's measured signal kind for intervention sign semantics."""
    gradiend = getattr(model, "gradiend", None) if model is not None else None
    mapping_kind = getattr(gradiend, "mapping_kind", None)
    if mapping_kind is not None:
        return "activation" if str(mapping_kind).strip().lower() == "activation" else "gradient"

    if trainer is not None:
        args = getattr(trainer, "_training_args", None) or getattr(trainer, "training_args", None)
        signal = training_arg_value(args, "signal", None)
        signal_kind = getattr(signal, "kind", None)
        if signal_kind is not None:
            return "activation" if str(signal_kind).strip().lower() == "activation" else "gradient"
    return default


def resolve_model_target(model: Any, trainer: Any = None, *, default: str = "diff") -> str:
    """Return ``model.target``, else ``TrainingArguments.target``, else *default*."""
    target = getattr(model, "target", None) if model is not None else None
    if target is not None:
        return validate_source_target("target", target)
    if trainer is not None:
        args = getattr(trainer, "_training_args", None) or getattr(trainer, "training_args", None)
        args_target = training_arg_value(args, "target", None)
        if args_target is not None:
            return validate_source_target("target", args_target)
    return default


def encoding_view_sign_for_source(source: str, alignment: str) -> float:
    """Sign to map encoder outputs into a factual- or counterfactual-aligned plot.

    Returns ``+1`` when ``model.source`` matches the column view, ``-1`` when they
    differ (e.g. ``source=alternative`` with ``alignment=factual``). Transition
    alignment is unchanged (``+1``).
    """
    validate_source_target("source", source)
    key = str(alignment).strip().lower()
    if key in {"transition", "transitions"}:
        return 1.0
    counterfactual_view = key in {
        "counterfactual", "cf", "alternative", "alternatives",
    }
    # both compiles to factual/diff geometry, so treat it as factual-aligned.
    alternative_source = source == "alternative"
    return -1.0 if counterfactual_view != alternative_source else 1.0


def training_arg_value(training_args: Any, key: str, default: Any = None) -> Any:
    """Read one field from TrainingArguments or a plain dict."""
    if training_args is None:
        return default
    if isinstance(training_args, dict):
        return training_args.get(key, default)
    return getattr(training_args, key, default)


def resolve_model_source(model: Any, trainer: Any = None, *, default: str = "factual") -> str:
    """Return ``model.source``, else ``TrainingArguments.source``, else *default*."""
    source = getattr(model, "source", None) if model is not None else None
    if source is not None:
        return validate_source_target("source", source)
    if trainer is not None:
        args = getattr(trainer, "_training_args", None) or getattr(trainer, "training_args", None)
        args_source = training_arg_value(args, "source", None)
        if args_source is not None:
            return validate_source_target("source", args_source)
    return default


def resolve_source_from_checkpoint_dir(
    checkpoint_dir: str,
    *,
    default: str = "factual",
) -> str:
    """Return the source a finished GRADIEND was trained with.

    Reads ``gradiend_context.json`` and the checkpoint's own ``training.json``.
    When they disagree, the persisted training source wins because older
    checkpoints sometimes saved the ``ModelWithGradiend`` constructor default
    ``factual`` instead of the source used to create their gradients.

    This deliberately does not inspect a caller's current TrainingArguments:
    evaluation settings must never reinterpret an already-trained checkpoint.
    """
    from gradiend.model.utils import read_gradiend_context
    from gradiend.trainer.core.stats import load_training_stats
    from gradiend.util.logging import get_logger

    log = get_logger(__name__)
    try:
        context_source, _, _ = read_gradiend_context(checkpoint_dir)
        context_source = validate_source_target("source", context_source)
    except (FileNotFoundError, KeyError, TypeError, ValueError):
        return default

    stats = load_training_stats(checkpoint_dir) or {}
    training_args = stats.get("training_args") or {}
    trained_source = (
        training_args.get("source")
        if isinstance(training_args, dict)
        else None
    )
    if trained_source is None:
        return context_source
    try:
        trained_source = validate_source_target("source", trained_source)
    except (TypeError, ValueError):
        log.warning(
            "Checkpoint %s: training.json has invalid source=%r; using "
            "gradiend_context.json source=%r.",
            checkpoint_dir,
            trained_source,
            context_source,
        )
        return context_source
    if trained_source != context_source:
        log.warning(
            "Checkpoint %s: gradiend_context.json has source=%r but training.json has %r; "
            "using training.json (the next save will repair gradiend_context.json).",
            checkpoint_dir,
            context_source,
            trained_source,
        )
    return trained_source


def sync_model_source_target_from_training_args(
    model: Any,
    training_args: Any,
    *,
    log_mismatch: bool = True,
    allow_overwrite: bool = True,
) -> None:
    """Align in-memory ``model._source`` / ``model._target`` before a new training run.

    Only call this when constructing a model for training from a base checkpoint,
    not when loading a finished GRADIEND for evaluation (``allow_overwrite=False``).
    """
    if model is None or training_args is None:
        return
    from gradiend.util.logging import get_logger

    log = get_logger(__name__)
    for key, private_attr in (("source", "_source"), ("target", "_target")):
        expected = training_arg_value(training_args, key, None)
        if expected is None:
            continue
        validate_source_target(key, expected)
        current = getattr(model, key, None)
        if current == expected:
            continue
        if not allow_overwrite:
            if log_mismatch:
                log.warning(
                    "Checkpoint model.%s is %r but TrainingArguments has %r; "
                    "keeping checkpoint value (source/target are fixed at training time).",
                    key,
                    current,
                    expected,
                )
            continue
        if log_mismatch:
            log.debug(
                "Setting model.%s to %r for training (TrainingArguments)",
                key,
                expected,
            )
        setattr(model, private_attr, expected)
