"""
Compare ACTIEND intervention application strategies across supported text tasks.

The script reuses the training setup from ``train_english_pronouns_activation.py``
(and ``train_gender_en.build_gender_trainer`` for ``gender_en``), derives the
target-class feature factor from model metadata, then sweeps hook application
parameters and writes a CSV, markdown summaries, heatmaps, and one decoder
probability-shift plot per full LR-swept intervention technique.

Run from the repository root:

    python -m gradiend.examples.experimental.sweep_english_pronouns_activation_interventions
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from gradiend import Signal, TextPredictionConfig, TextPredictionTrainer, TrainingArguments
from gradiend.evaluator.decoder import derive_default_feature_factor
from gradiend.examples.english_pronoun_datasets import (
    EN_PRONOUN_HF_SPLITS,
    EN_PRONOUNS_HF_DATASET,
    load_english_pronoun_neutral_data,
)
from gradiend.examples.train_gender_en import build_gender_trainer
from gradiend.gradiend_split import GradiendSplit
from gradiend.model.utils import prediction_eval_kind
from gradiend.trainer.core import SignalScope

SUPPORTED_TASKS = ("gender_en", "english_pronouns_activation")
DEFAULT_TASK = "gender_en"

TASK_CONFIGS: Dict[str, Dict[str, Any]] = {
    "gender_en": {
        #"model_name": "distilbert-base-cased",
        #"model_name": "bert-base-cased",
        "model_name": "gpt2",
        "experiment_dir": "runs/examples/gender_en_activation_sweep_gpt2_new",
        "target_class": "M",
        "target_classes": ("M", "F"),
    },
    "english_pronouns_activation": {
        "model_name": "gpt2",
        "experiment_dir": "runs/english_pronouns_activation_factual",
        "target_class": "3SG",
        "target_classes": ("3SG", "3PL"),
        "run_id": "pronoun_3sg_3pl_activation",
    },
}
OUTPUT_DIR = Path(TASK_CONFIGS[DEFAULT_TASK]["experiment_dir"]) / "intervention_sweep"

MAX_STEPS = 50
#MAX_STEPS = 200
#MAX_STEPS = 100
EVAL_MAX_SIZE = 200
EVAL_MAX_SIZE = 64
LMS_PREFILTER_RATIO = 0.99
# Dense around the empirically useful ACTIEND range while still retaining a
# couple of larger sanity-check values for collapse/fluency diagnostics.
LR_GRID = (1e-2, 1e-1, 1e0, 3e0, 5e0, 1e1, 2e1, 3e1, 5e1, 1e2, 1e3, 1e4, 1e5, 1e6)
COVERAGE_BATCH_SIZE = 8

USE_CACHE = False  # Decoder eval is wrong with caching!!
WRITE_DECODER_RAW_CSV = False  # per-sample decoder CSVs are large; grids/plots are enough


def _is_static_application_metric(metric: str) -> bool:
    """Return whether a plotted metric belongs to the application config, not LR.

    Selector coverage is measured once on baseline activations before an
    intervention strength is applied. Repeating it across the LR grid makes the
    heatmap look like an intervention sweep even though the value cannot depend
    on LR.
    """
    return metric.startswith("selector_")


def _model_device(model: Any) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _frame_records(frame: Any, *, max_rows: int) -> List[Dict[str, Any]]:
    if frame is None or len(frame) == 0:
        return []
    subset = frame.head(max_rows) if hasattr(frame, "head") else frame[:max_rows]
    if hasattr(subset, "to_dict"):
        return list(subset.to_dict("records"))
    return [dict(row) for row in subset]


def _coverage_text_from_row(row: Mapping[str, Any], *, tokenizer: Any, eval_kind: str, key_text: str) -> Optional[str]:
    text = row.get(key_text)
    if text is None:
        text = row.get("masked")
    if text is None:
        text = row.get("text")
    if text is None:
        return None
    text = str(text)
    if "[MASK]" in text and eval_kind in {"clm_next_token", "clm_sequence_cloze"}:
        text = text.split("[MASK]", 1)[0]
    elif "[MASK]" in text:
        mask_token = getattr(tokenizer, "mask_token", None)
        if mask_token:
            text = text.replace("[MASK]", mask_token)
    return text if text.strip() else None


def _merge_coverage_summaries(summaries: List[Mapping[str, Any]]) -> Dict[str, Any]:
    selected = sum(int(summary.get("selected_positions") or 0) for summary in summaries)
    candidate = sum(int(summary.get("candidate_positions") or 0) for summary in summaries)
    total = sum(int(summary.get("total_positions") or 0) for summary in summaries)
    return {
        "selected_positions": selected,
        "candidate_positions": candidate,
        "total_positions": total,
        "coverage": (selected / total) if total else None,
        "scope_coverage": (selected / candidate) if candidate else None,
    }


def _coverage_for_rows(
    model_with_gradiend: Any,
    tokenizer: Any,
    frame: Any,
    *,
    feature_factor: float,
    selector_config: Mapping[str, Any],
    activation_modules: Optional[Any] = None,
    max_rows: int,
    key_text: str = "masked",
) -> Dict[str, Any]:
    base_model = model_with_gradiend.base_model
    device = _model_device(base_model)
    eval_kind = prediction_eval_kind(base_model)
    records = _frame_records(frame, max_rows=max_rows)
    texts = [
        text
        for row in records
        if (text := _coverage_text_from_row(row, tokenizer=tokenizer, eval_kind=eval_kind, key_text=key_text)) is not None
    ]
    summaries: List[Mapping[str, Any]] = []
    if len(texts) > 1 and getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    intervention_kwargs = {
        key: value
        for key, value in selector_config.items()
        if key in {"threshold", "direction", "target_encoding", "tolerance"}
    }
    for start in range(0, len(texts), COVERAGE_BATCH_SIZE):
        batch_texts = texts[start:start + COVERAGE_BATCH_SIZE]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=len(batch_texts) > 1,
            truncation=True,
            max_length=512,
        )
        inputs = {
            key: value.to(device) if torch.is_tensor(value) else value
            for key, value in dict(encoded).items()
        }
        summaries.append(
            model_with_gradiend.activation_selector_coverage(
                feature_factor=feature_factor,
                token_selector=selector_config["token_selector"],
                activation_gate=selector_config.get("activation_gate"),
                activation_modules=activation_modules,
                **intervention_kwargs,
                **inputs,
            )
        )
    out = _merge_coverage_summaries(summaries)
    out["n_texts"] = len(texts)
    return out


def _selector_coverage_columns(
    model_with_gradiend: Any,
    tokenizer: Any,
    training_like_df: Any,
    neutral_df: Any,
    *,
    target_class: str,
    feature_factor: float,
    selector_config: Mapping[str, Any],
    activation_modules: Optional[Any] = None,
    max_rows: int,
) -> Dict[str, Any]:
    dataset_col = None
    for candidate in ("label_class", "factual_id"):
        if candidate in getattr(training_like_df, "columns", []):
            dataset_col = candidate
            break
    if dataset_col is not None:
        target_df = training_like_df[training_like_df[dataset_col].astype(str) == str(target_class)]
        other_df = training_like_df[training_like_df[dataset_col].astype(str) != str(target_class)]
    else:
        target_df = training_like_df
        other_df = None

    target = _coverage_for_rows(
        model_with_gradiend,
        tokenizer,
        target_df,
        feature_factor=feature_factor,
        selector_config=selector_config,
        activation_modules=activation_modules,
        max_rows=max_rows,
    )
    other = _coverage_for_rows(
        model_with_gradiend,
        tokenizer,
        other_df,
        feature_factor=feature_factor,
        selector_config=selector_config,
        activation_modules=activation_modules,
        max_rows=max_rows,
    )
    neutral = _coverage_for_rows(
        model_with_gradiend,
        tokenizer,
        neutral_df,
        feature_factor=feature_factor,
        selector_config=selector_config,
        activation_modules=activation_modules,
        max_rows=max_rows,
        key_text="text",
    )
    target_cov = _as_float(target.get("coverage"))
    other_cov = _as_float(other.get("coverage"))
    neutral_cov = _as_float(neutral.get("coverage"))
    target_scope_cov = _as_float(target.get("scope_coverage"))
    other_scope_cov = _as_float(other.get("scope_coverage"))
    neutral_scope_cov = _as_float(neutral.get("scope_coverage"))
    return {
        "selector_target_coverage": target_cov,
        "selector_other_coverage": other_cov,
        "selector_neutral_coverage": neutral_cov,
        "selector_target_scope_coverage": target_scope_cov,
        "selector_other_scope_coverage": other_scope_cov,
        "selector_neutral_scope_coverage": neutral_scope_cov,
        "selector_specificity": (1.0 - neutral_cov) if neutral_cov is not None else None,
        "selector_scope_specificity": (
            1.0 - neutral_scope_cov
            if neutral_scope_cov is not None
            else None
        ),
        "selector_target_minus_neutral": (
            target_cov - neutral_cov
            if target_cov is not None and neutral_cov is not None
            else None
        ),
        "selector_scope_target_minus_neutral": (
            target_scope_cov - neutral_scope_cov
            if target_scope_cov is not None and neutral_scope_cov is not None
            else None
        ),
        "selector_other_minus_neutral": (
            other_cov - neutral_cov
            if other_cov is not None and neutral_cov is not None
            else None
        ),
        "selector_scope_other_minus_neutral": (
            other_scope_cov - neutral_scope_cov
            if other_scope_cov is not None and neutral_scope_cov is not None
            else None
        ),
        "selector_target_selected_positions": target.get("selected_positions"),
        "selector_other_selected_positions": other.get("selected_positions"),
        "selector_neutral_selected_positions": neutral.get("selected_positions"),
        "selector_target_candidate_positions": target.get("candidate_positions"),
        "selector_other_candidate_positions": other.get("candidate_positions"),
        "selector_neutral_candidate_positions": neutral.get("candidate_positions"),
        "selector_target_positions": target.get("total_positions"),
        "selector_other_positions": other.get("total_positions"),
        "selector_neutral_positions": neutral.get("total_positions"),
        "selector_target_texts": target.get("n_texts"),
        "selector_other_texts": other.get("n_texts"),
        "selector_neutral_texts": neutral.get("n_texts"),
        "selector_coverage_note": "baseline activations; same selector mask resolver as ACTIEND hooks",
    }


def _empty_selector_coverage_columns() -> Dict[str, Any]:
    return {
        "selector_target_coverage": None,
        "selector_other_coverage": None,
        "selector_neutral_coverage": None,
        "selector_target_scope_coverage": None,
        "selector_other_scope_coverage": None,
        "selector_neutral_scope_coverage": None,
        "selector_specificity": None,
        "selector_scope_specificity": None,
        "selector_target_minus_neutral": None,
        "selector_scope_target_minus_neutral": None,
        "selector_other_minus_neutral": None,
        "selector_scope_other_minus_neutral": None,
        "selector_target_selected_positions": None,
        "selector_other_selected_positions": None,
        "selector_neutral_selected_positions": None,
        "selector_target_candidate_positions": None,
        "selector_other_candidate_positions": None,
        "selector_neutral_candidate_positions": None,
        "selector_target_positions": None,
        "selector_other_positions": None,
        "selector_neutral_positions": None,
        "selector_target_texts": None,
        "selector_other_texts": None,
        "selector_neutral_texts": None,
        "selector_coverage_note": "not available: model is not activation-space ACTIEND",
    }


def _as_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _nested_get(mapping: Mapping[str, Any], keys: Iterable[str]) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _mapping_get_by_name(mapping: Mapping[str, Any], name: str) -> Any:
    if name in mapping:
        return mapping[name]
    for key, value in mapping.items():
        if str(key) == str(name):
            return value
    return None


def _probability_classes(result: Mapping[str, Any], *, target_class: str) -> List[str]:
    classes: List[str] = []

    def add(name: Any) -> None:
        if name is None:
            return
        as_text = str(name)
        if as_text not in classes:
            classes.append(as_text)

    add(target_class)
    probs_by_dataset = result.get("probs_by_dataset")
    if isinstance(probs_by_dataset, Mapping):
        for dataset_class, class_probs in probs_by_dataset.items():
            add(dataset_class)
            if isinstance(class_probs, Mapping):
                for class_name in class_probs:
                    add(class_name)
    for key in ("probs", "probs_factual"):
        probs = result.get(key)
        if isinstance(probs, Mapping):
            for class_name in probs:
                add(class_name)
    return classes


def _panel_probability(
    probs_by_dataset: Mapping[str, Any],
    *,
    dataset_class: Optional[str],
    class_name: Optional[str],
) -> Optional[float]:
    if dataset_class is None or class_name is None:
        return None
    panel = _mapping_get_by_name(probs_by_dataset, dataset_class)
    if not isinstance(panel, Mapping):
        return None
    return _as_float(_mapping_get_by_name(panel, class_name))


def _flatten_eval(result: Mapping[str, Any], *, target_class: str) -> Dict[str, Any]:
    probs = result.get("probs") if isinstance(result.get("probs"), Mapping) else {}
    probs_factual = result.get("probs_factual") if isinstance(result.get("probs_factual"), Mapping) else {}
    probs_by_dataset = (
        result.get("probs_by_dataset")
        if isinstance(result.get("probs_by_dataset"), Mapping)
        else {}
    )
    classes = _probability_classes(result, target_class=target_class)
    other_classes = [class_name for class_name in classes if class_name != str(target_class)]
    other_class = other_classes[0] if len(other_classes) == 1 else None

    target_on_other = _as_float(_mapping_get_by_name(probs, target_class))
    target_on_target = _as_float(_mapping_get_by_name(probs_factual, target_class))
    if target_on_other is None:
        target_on_other = _panel_probability(
            probs_by_dataset,
            dataset_class=other_class,
            class_name=target_class,
        )
    if target_on_target is None:
        target_on_target = _panel_probability(
            probs_by_dataset,
            dataset_class=target_class,
            class_name=target_class,
        )

    other_on_other = _panel_probability(
        probs_by_dataset,
        dataset_class=other_class,
        class_name=other_class,
    )
    other_on_target = _panel_probability(
        probs_by_dataset,
        dataset_class=target_class,
        class_name=other_class,
    )

    out: Dict[str, Any] = {
        # target_probability is the decoder selection metric:
        # P(target class) on the other class's factual dataset.
        "target_probability": target_on_other,
        "target_factual_probability": target_on_target,
        "target_probability_on_other_dataset": target_on_other,
        "target_probability_on_target_dataset": target_on_target,
        "other_probability_on_other_dataset": other_on_other,
        "other_probability_on_target_dataset": other_on_target,
        "lms": _as_float(_nested_get(result, ("lms", "lms"))),
        "perplexity": _as_float(_nested_get(result, ("lms", "perplexity"))),
        "total_log_likelihood": _as_float(_nested_get(result, ("lms", "total_log_likelihood"))),
        "total_tokens": _as_float(_nested_get(result, ("lms", "total_tokens"))),
    }
    if other_class is not None:
        out["other_class"] = other_class
    if target_on_other is not None and other_on_other is not None:
        out["target_margin_on_other_dataset"] = target_on_other - other_on_other
    if target_on_target is not None and other_on_target is not None:
        out["target_margin_on_target_dataset"] = target_on_target - other_on_target

    for key in ("feature_score", "accuracy", "mean_probability", "loss"):
        value = _as_float(result.get(key))
        if value is not None:
            out[key] = value
    return out


def _add_baseline_deltas(row: Dict[str, Any], *, baseline: Mapping[str, Any]) -> None:
    delta_metrics = [
        "target_probability",
        "target_factual_probability",
        "target_probability_on_other_dataset",
        "target_probability_on_target_dataset",
        "other_probability_on_other_dataset",
        "other_probability_on_target_dataset",
        "target_margin_on_other_dataset",
        "target_margin_on_target_dataset",
    ]
    for metric in delta_metrics:
        baseline_value = _as_float(baseline.get(metric))
        value = _as_float(row.get(metric))
        row[f"{metric}_delta"] = (
            value - baseline_value
            if value is not None and baseline_value is not None
            else None
        )

    baseline_lms = _as_float(baseline.get("lms"))
    lms = _as_float(row.get("lms"))
    row["lms_ratio"] = lms / baseline_lms if lms is not None and baseline_lms else None
    row["lms_prefilter_pass"] = (
        row["lms_ratio"] >= LMS_PREFILTER_RATIO
        if row["lms_ratio"] is not None
        else None
    )

    target_delta = _as_float(row.get("target_probability_delta"))
    factual_target_delta = _as_float(row.get("target_factual_probability_delta"))
    if factual_target_delta is not None:
        row["target_factual_probability_side_effect_abs"] = abs(factual_target_delta)
    if target_delta is not None and factual_target_delta is not None:
        row["probability_specificity_score"] = target_delta - abs(factual_target_delta)

    margin_delta = _as_float(row.get("target_margin_on_other_dataset_delta"))
    factual_margin_delta = _as_float(row.get("target_margin_on_target_dataset_delta"))
    if factual_margin_delta is not None:
        row["target_margin_side_effect_abs"] = abs(factual_margin_delta)
    if margin_delta is not None and factual_margin_delta is not None:
        row["margin_specificity_score"] = margin_delta - abs(factual_margin_delta)


def _rank_probability_shift(row: Mapping[str, Any]) -> tuple:
    """Rank by probability shift after the LMS gate; tie-break with causal and selector specificity."""
    passed_lms = row.get("lms_prefilter_pass") is True
    target_delta = _as_float(row.get("target_probability_delta"))
    probability_specificity = _as_float(row.get("probability_specificity_score"))
    target_probability = _as_float(row.get("target_probability"))
    neutral_coverage = _as_float(row.get("selector_neutral_scope_coverage"))
    if neutral_coverage is None:
        neutral_coverage = _as_float(row.get("selector_neutral_coverage"))
    target_minus_neutral = _as_float(row.get("selector_scope_target_minus_neutral"))
    if target_minus_neutral is None:
        target_minus_neutral = _as_float(row.get("selector_target_minus_neutral"))
    lms = _as_float(row.get("lms"))
    perplexity = _as_float(row.get("perplexity"))
    return (
        0 if passed_lms else 1,
        -(target_delta if target_delta is not None else float("-inf")),
        -(probability_specificity if probability_specificity is not None else float("-inf")),
        neutral_coverage if neutral_coverage is not None else float("inf"),
        -(target_minus_neutral if target_minus_neutral is not None else float("-inf")),
        -(target_probability if target_probability is not None else float("-inf")),
        -(lms if lms is not None else float("-inf")),
        perplexity if perplexity is not None else float("inf"),
    )


_APPLICATION_MODE_ORDER = {
    "steering": 0,
    "encoder_gated": 1,
}
_TOKEN_SCOPE_ORDER = {
    "all_tokens": 0,
    "prediction": 1,
}
_GATE_ORDER = {
    "always": 0,
    "encoder_direction": 1,
    "encoder_range": 2,
    "encoder_abs": 3,
}


def _configuration_sort_key(row: Mapping[str, Any]) -> tuple:
    """Heatmap/table order: target → mode → token scope → gate → full sites before single-site children."""
    target = str(row.get("target_class") or "")
    mode = str(row.get("application_mode") or "")
    scope = str(row.get("token_scope") or "")
    gate = str(row.get("gate") or "")
    threshold = row.get("threshold")
    threshold_rank = float(threshold) if isinstance(threshold, (int, float)) else 0.0
    site_scope = str(row.get("site_scope") or "trained_sites")
    is_single = 0 if site_scope == "trained_sites" else 1
    module = str(row.get("activation_modules") or "")
    base_label = str(row.get("selector_label") or row.get("label") or "")
    if is_single and module and base_label.endswith(f" | {module}"):
        base_label = base_label[: -(len(module) + 3)]
    return (
        target,
        _APPLICATION_MODE_ORDER.get(mode, 9),
        _TOKEN_SCOPE_ORDER.get(scope, 9),
        _GATE_ORDER.get(gate, 9),
        threshold_rank,
        base_label,
        is_single,
        module,
    )


def _strategy_grid(direction: float) -> List[Dict[str, Any]]:
    """Build an explicit application-mode × token-scope × gate ACTIEND sweep grid."""
    # Broader token scope first; steering before gated (heatmap uses the same key).
    token_scopes = [
        {"token_selector": "all", "token_scope": "all_tokens", "token_scope_label": "all tokens"},
        {"token_selector": "prediction", "token_scope": "prediction", "token_scope_label": "prediction slot"},
    ]
    gates = [
        {
            "activation_gate": None,
            "gate": "always",
            "gate_label": "always",
            "application_mode": "steering",
            "application_mode_label": "steering",
        },
        {
            "activation_gate": "encoder_direction",
            "gate": "encoder_direction",
            "threshold": 0.5,
            "direction": direction,
            "gate_label": "direction gt 0.5",
            "application_mode": "encoder_gated",
            "application_mode_label": "encoder-gated",
        },
        {
            "activation_gate": "encoder_direction",
            "gate": "encoder_direction",
            "threshold": 0.8,
            "direction": direction,
            "gate_label": "direction gt 0.8",
            "application_mode": "encoder_gated",
            "application_mode_label": "encoder-gated",
        },
        # Note: encoder_range near-target ±0.2 is omitted — for |feature_factor|=1
        # it is identical to direction gt 0.8 on encodings in [-1, 1].
        {
            "activation_gate": "encoder_abs",
            "gate": "encoder_abs",
            "threshold": 0.8,
            "gate_label": "abs gt 0.8",
            "application_mode": "encoder_gated",
            "application_mode_label": "encoder-gated",
        },
    ]
    strategies: List[Dict[str, Any]] = []
    for gate in gates:
        for token_scope in token_scopes:
            strategy = {**token_scope, **gate}
            strategy["selector"] = strategy["token_selector"]
            strategy["label"] = (
                f"{strategy['application_mode_label']} | "
                f"{strategy['token_scope_label']} | "
                f"{strategy['gate_label']}"
            )
            strategy["site_scope"] = "trained_sites"
            strategies.append(strategy)
    strategies.sort(key=_configuration_sort_key)
    return strategies


def _make_trainer() -> TextPredictionTrainer:
    task_cfg = TASK_CONFIGS["english_pronouns_activation"]
    neutral_data = load_english_pronoun_neutral_data()
    config = TextPredictionConfig(
        run_id=task_cfg["run_id"],
        hf_dataset=EN_PRONOUNS_HF_DATASET,
        hf_splits=EN_PRONOUN_HF_SPLITS,
        target_classes=["3SG", "3PL"],
        neutral_data=neutral_data,
        decoder_eval_export_row_wise_csv=True,
    )
    args = TrainingArguments(
        experiment_dir=task_cfg["experiment_dir"],
        train_batch_size=1,
        eval_batch_size=1,
        gradiend_batch_size=1,
        num_train_epochs=5,
        max_steps=MAX_STEPS,
        eval_steps=100,
        bias_encoder=False,
        learning_rate=1e-5,
        target="diff",
        source="factual",
        signal=Signal.activation(),
        signal_scope=SignalScope.layers(),
        gradiend_split=GradiendSplit.by_tensor(),
        add_neutral_identity_transitions=True,
        fail_on_non_convergence=False,
        use_cache=USE_CACHE,
    )
    return TextPredictionTrainer(model=task_cfg["model_name"], config=config, args=args)


def _make_gender_en_trainer() -> TextPredictionTrainer:
    task_cfg = TASK_CONFIGS["gender_en"]
    args = TrainingArguments(
        experiment_dir=task_cfg["experiment_dir"],
        train_batch_size=8,
        eval_batch_size=8,
        gradiend_batch_size=8,
        num_train_epochs=1,
        encoder_eval_max_size=100,
        max_steps=MAX_STEPS,
        eval_steps=25,
        source="factual",
        target="diff",
        signal=Signal.activation(),
        #signal_scope=SignalScope.layers(),
        #gradiend_split=GradiendSplit.by_tensor(),
        add_neutral_identity_transitions=True,
        #learning_rate=1e-5,
        learning_rate=1e-4,
        fail_on_non_convergence=False,
        use_cache=USE_CACHE,
    )
    trainer = build_gender_trainer(
        model=task_cfg["model_name"],
        names_per_template=4,
        args=args,
    )
    # Keep sweep behavior on the standard evaluate_decoder/evaluate_base_model path.
    trainer.evaluate_base_model = TextPredictionTrainer.evaluate_base_model.__get__(trainer, TextPredictionTrainer)
    # Export per-row scores only when scoring is already row-wise (overlap auto-fallback).
    # Do not force decoder_eval_targets to "label" just for CSV — that changes scoring mode.
    trainer.config.decoder_eval_export_row_wise_csv = True
    return trainer


def _make_trainer_for_task(task: str) -> TextPredictionTrainer:
    if task == "gender_en":
        return _make_gender_en_trainer()
    if task == "english_pronouns_activation":
        return _make_trainer()
    raise ValueError(f"Unsupported task: {task}")


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key.startswith("_"):
                continue
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: value for key, value in row.items() if not key.startswith("_")})


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def _format_markdown_cell(value: Any) -> str:
    """Format a table cell without letting literal pipes split Markdown columns."""
    return _format_cell(value).replace("\n", "<br>").replace("|", "\\|")


def _has_report_value(rows: List[Dict[str, Any]], column: str) -> bool:
    return any(row.get(column) is not None and row.get(column) != "" for row in rows)


def _write_markdown_table(path: Path, rows: List[Dict[str, Any]]) -> None:
    always_columns = {
        "rank",
        "target_rank",
        "selector_label",
        "application_mode",
        "token_scope",
        "gate",
        "site_scope",
        "learning_rate",
        "feature_factor",
        "target_probability",
        "target_probability_delta",
        "target_factual_probability",
        "target_factual_probability_delta",
        "lms",
        "lms_ratio",
        "lms_prefilter_pass",
    }
    preferred_columns = [
        "rank",
        "target_rank",
        "selector_label",
        "application_mode",
        "application_mode_label",
        "token_scope",
        "gate",
        "site_scope",
        "activation_modules",
        "learning_rate",
        "feature_factor",
        "target_class",
        "other_class",
        "target_probability",
        "target_probability_delta",
        "target_factual_probability",
        "target_factual_probability_delta",
        "probability_specificity_score",
        "target_margin_on_other_dataset",
        "target_margin_on_other_dataset_delta",
        "target_margin_on_target_dataset",
        "target_margin_on_target_dataset_delta",
        "margin_specificity_score",
        "lms",
        "lms_ratio",
        "lms_prefilter_pass",
        "selector_target_coverage",
        "selector_other_coverage",
        "selector_neutral_coverage",
        "selector_target_scope_coverage",
        "selector_other_scope_coverage",
        "selector_neutral_scope_coverage",
        "selector_target_minus_neutral",
        "selector_scope_target_minus_neutral",
        "selector_specificity",
        "selector_scope_specificity",
        "perplexity",
        "feature_score",
        "accuracy",
        "mean_probability",
        "loss",
    ]
    columns = [
        column
        for column in preferred_columns
        if column in always_columns or _has_report_value(rows, column)
    ]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_markdown_cell(row.get(column)) for column in columns) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _append_markdown_rows(lines: List[str], columns: List[str], rows: List[Mapping[str, Any]]) -> None:
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join("---" for _ in columns) + " |")
    for row in rows:
        lines.append("| " + " | ".join(_format_markdown_cell(row.get(column)) for column in columns) + " |")


def _specificity_rank_key(row: Mapping[str, Any], metric: str) -> tuple:
    passed_lms = row.get("lms_prefilter_pass") is True
    score = _as_float(row.get(metric))
    target_delta = _as_float(row.get("target_probability_delta"))
    lms_ratio = _as_float(row.get("lms_ratio"))
    neutral_coverage = _as_float(row.get("selector_neutral_scope_coverage"))
    if neutral_coverage is None:
        neutral_coverage = _as_float(row.get("selector_neutral_coverage"))
    return (
        0 if passed_lms else 1,
        -(score if score is not None else float("-inf")),
        -(target_delta if target_delta is not None else float("-inf")),
        neutral_coverage if neutral_coverage is not None else float("inf"),
        -(lms_ratio if lms_ratio is not None else float("-inf")),
    )


def _unique_selector_coverage_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for row in rows:
        if row.get("kind") != "intervention":
            continue
        key = (row.get("selector_label"), row.get("site_scope"), row.get("activation_modules"))
        key = (row.get("target_class"),) + key
        if key in seen:
            continue
        seen.add(key)
        if any(row.get(column) is not None for column in (
            "selector_target_coverage",
            "selector_other_coverage",
            "selector_neutral_coverage",
            "selector_neutral_scope_coverage",
            "selector_target_minus_neutral",
            "selector_scope_target_minus_neutral",
            "selector_specificity",
            "selector_scope_specificity",
        )):
            out.append(row)
    def sort_key(row: Mapping[str, Any]) -> tuple:
        target_minus_neutral = _as_float(row.get("selector_scope_target_minus_neutral"))
        if target_minus_neutral is None:
            target_minus_neutral = _as_float(row.get("selector_target_minus_neutral"))
        neutral_coverage = _as_float(row.get("selector_neutral_scope_coverage"))
        if neutral_coverage is None:
            neutral_coverage = _as_float(row.get("selector_neutral_coverage"))
        return (
            -(target_minus_neutral if target_minus_neutral is not None else float("-inf")),
            neutral_coverage if neutral_coverage is not None else float("inf"),
        )

    return sorted(out, key=sort_key)


def _write_specificity_summary(path: Path, rows: List[Dict[str, Any]], *, target_class: str) -> None:
    """Write a compact human-readable view of causal and selector specificity."""
    intervention_rows = [row for row in rows if row.get("kind") == "intervention"]
    lms_rows = [row for row in intervention_rows if row.get("lms_prefilter_pass") is True]
    ranked_pool = lms_rows or intervention_rows
    top_probability_specific = sorted(
        ranked_pool,
        key=lambda row: _specificity_rank_key(row, "probability_specificity_score"),
    )[:8]
    top_margin_specific = sorted(
        ranked_pool,
        key=lambda row: _specificity_rank_key(row, "margin_specificity_score"),
    )[:8]
    selector_rows = _unique_selector_coverage_rows(intervention_rows)[:12]

    lines = [
        "# ACTIEND Intervention Specificity Summary",
        "",
        f"Target class: `{target_class}`.",
        "",
        "`probability_specificity_score = target_probability_delta - abs(target_factual_probability_delta)`.",
        "`margin_specificity_score = target_margin_on_other_dataset_delta - abs(target_margin_on_target_dataset_delta)`.",
        "`selector_specificity = 1 - selector_neutral_coverage`; coverage is measured on baseline activations.",
        "`selector_*_coverage` uses all token positions as denominator; `selector_*_scope_coverage` uses only positions allowed by the token scope.",
        "",
        "Higher causal specificity means the intervention moves the counterfactual target probability more than it moves the factual target-class panel.",
        "",
        "## Top Probability-Specific Rows",
        "",
    ]
    _append_markdown_rows(
        lines,
        [
            "rank",
            "selector_label",
            "application_mode",
            "site_scope",
            "activation_modules",
            "learning_rate",
            "target_probability_delta",
            "target_factual_probability_delta",
            "probability_specificity_score",
            "lms_ratio",
            "selector_neutral_coverage",
            "selector_neutral_scope_coverage",
        ],
        top_probability_specific,
    )
    lines.extend(["", "## Top Margin-Specific Rows", ""])
    _append_markdown_rows(
        lines,
        [
            "rank",
            "selector_label",
            "application_mode",
            "site_scope",
            "activation_modules",
            "learning_rate",
            "target_margin_on_other_dataset_delta",
            "target_margin_on_target_dataset_delta",
            "margin_specificity_score",
            "lms_ratio",
            "selector_neutral_coverage",
            "selector_neutral_scope_coverage",
        ],
        top_margin_specific,
    )
    lines.extend(["", "## Selector Coverage Specificity", ""])
    _append_markdown_rows(
        lines,
        [
            "selector_label",
            "target_class",
            "application_mode",
            "site_scope",
            "activation_modules",
            "selector_target_coverage",
            "selector_other_coverage",
            "selector_neutral_coverage",
            "selector_target_scope_coverage",
            "selector_other_scope_coverage",
            "selector_neutral_scope_coverage",
            "selector_target_minus_neutral",
            "selector_scope_target_minus_neutral",
            "selector_specificity",
            "selector_scope_specificity",
        ],
        selector_rows,
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _safe_filename_stem(value: str, *, max_len: int = 120) -> str:
    stem = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    stem = "_".join(part for part in stem.split("_") if part)
    return (stem or "technique")[:max_len]


def _strategy_intervention_kwargs(
    selector_config: Mapping[str, Any],
    *,
    activation_modules: Optional[Any] = None,
) -> Dict[str, Any]:
    kwargs = {
        "token_selector": selector_config.get("token_selector"),
        "activation_gate": selector_config.get("activation_gate"),
        "activation_modules": activation_modules,
    }
    for key in ("threshold", "direction", "target_encoding", "tolerance"):
        if key in selector_config:
            kwargs[key] = selector_config[key]
    return {key: value for key, value in kwargs.items() if value is not None}


def _decoder_grid_cache_path(selector_label: str, *, target_class: str) -> Path:
    stem = _safe_filename_stem(f"{target_class}_{selector_label}")
    return OUTPUT_DIR / "decoder_grids" / f"decoder_grid_{stem}.json"


def _decoder_plot_output_path(selector_label: str, *, target_class: str) -> Path:
    stem = _safe_filename_stem(f"{target_class}_{selector_label}")
    return OUTPUT_DIR / "decoder_by_technique" / f"decoder_probability_shifts_{stem}.png"


def _decoder_raw_output_path(selector_label: str, *, target_class: str) -> Optional[str]:
    if not WRITE_DECODER_RAW_CSV:
        return None
    stem = _safe_filename_stem(f"{target_class}_{selector_label}")
    return str(OUTPUT_DIR / "decoder_raw" / f"decoder_raw_samples_{stem}.csv")


def _decoder_grid_feature_lr(id_key: Any, entry: Mapping[str, Any]) -> Optional[tuple[float, float]]:
    if isinstance(id_key, tuple) and len(id_key) == 2:
        ff, lr = id_key
        return float(ff), float(lr)
    raw_id = entry.get("id")
    if isinstance(raw_id, Mapping):
        ff = raw_id.get("feature_factor")
        lr = raw_id.get("learning_rate")
        if ff is not None and lr is not None:
            return float(ff), float(lr)
    if isinstance(raw_id, tuple) and len(raw_id) == 2:
        ff, lr = raw_id
        return float(ff), float(lr)
    return None


def _decoder_kwargs_for_task(
    *,
    target_class: str,
    lrs: List[float],
    feature_factors: Optional[List[float]] = None,
    output_path: Optional[str] = None,
    plot: bool = True,
    show: bool = False,
    plot_kwargs: Optional[Mapping[str, Any]] = None,
    max_size_training_like: Optional[int] = None,
    max_size_neutral: Optional[int] = None,
    training_like_df: Any = None,
    neutral_df: Any = None,
    **kwargs: Any,
) -> Mapping[str, Any]:
    call_kwargs: Dict[str, Any] = {
        "lrs": lrs,
        "use_cache": USE_CACHE,
        "plot": plot,
        "show": show,
        **kwargs,
    }
    if feature_factors is not None:
        call_kwargs["feature_factors"] = feature_factors
    if output_path is not None:
        call_kwargs["output_path"] = output_path
    if plot_kwargs is not None:
        call_kwargs["plot_kwargs"] = dict(plot_kwargs)
    if max_size_training_like is not None:
        call_kwargs["max_size_training_like"] = max_size_training_like
    if max_size_neutral is not None:
        call_kwargs["max_size_neutral"] = max_size_neutral
    if training_like_df is not None:
        call_kwargs["training_like_df"] = training_like_df
    if neutral_df is not None:
        call_kwargs["neutral_df"] = neutral_df

    call_kwargs["target_class"] = target_class
    return call_kwargs


def _baseline_row_from_decoder_result(
    decoder_result: Mapping[str, Any],
    *,
    target_class: str,
    feature_factor: float,
) -> Dict[str, Any]:
    grid = decoder_result.get("grid") if isinstance(decoder_result.get("grid"), Mapping) else {}
    base_entry = grid.get("base") if isinstance(grid, Mapping) else None
    if not isinstance(base_entry, Mapping):
        raise ValueError("evaluate_decoder result is missing grid['base']; cannot build sweep baseline row")
    row: Dict[str, Any] = {
        "rank": None,
        "kind": "baseline",
        "selector": "none",
        "selector_label": "baseline",
        "learning_rate": 0.0,
        "feature_factor": feature_factor,
        "target_class": target_class,
        **_flatten_eval(base_entry, target_class=target_class),
    }
    _add_baseline_deltas(row, baseline=row)
    row["target_probability_delta"] = 0.0
    row["lms_ratio"] = 1.0 if row.get("lms") is not None else None
    row["lms_prefilter_pass"] = row.get("lms") is not None
    return row


def _rows_from_decoder_result(
    decoder_result: Mapping[str, Any],
    *,
    selector_config: Mapping[str, Any],
    target_class: str,
    baseline_row: Mapping[str, Any],
    coverage_columns: Mapping[str, Any],
    activation_modules: Optional[Any] = None,
    site_scope: str = "trained_sites",
    selector_label: Optional[str] = None,
) -> List[Dict[str, Any]]:
    grid = decoder_result.get("grid") if isinstance(decoder_result.get("grid"), Mapping) else {}
    label = selector_label or str(selector_config["label"])
    rows: List[Dict[str, Any]] = []
    for id_key, entry in grid.items():
        if id_key == "base" or not isinstance(entry, Mapping):
            continue
        parsed = _decoder_grid_feature_lr(id_key, entry)
        if parsed is None:
            continue
        feature_factor, learning_rate = parsed
        row = {
            "rank": None,
            "kind": "intervention",
            "selector": selector_config.get("token_selector"),
            "selector_label": label,
            "application_mode": selector_config.get("application_mode"),
            "application_mode_label": selector_config.get("application_mode_label"),
            "token_scope": selector_config.get("token_scope"),
            "gate": selector_config.get("gate"),
            "gate_label": selector_config.get("gate_label"),
            "activation_gate": selector_config.get("activation_gate"),
            "site_scope": site_scope,
            "activation_modules": (
                ",".join(activation_modules)
                if isinstance(activation_modules, list)
                else activation_modules
            ),
            "threshold": selector_config.get("threshold"),
            "direction": selector_config.get("direction"),
            "target_encoding": selector_config.get("target_encoding"),
            "tolerance": selector_config.get("tolerance"),
            "learning_rate": learning_rate,
            "feature_factor": feature_factor,
            "target_class": target_class,
            **dict(coverage_columns),
            **_flatten_eval(entry, target_class=target_class),
        }
        _add_baseline_deltas(row, baseline=baseline_row)
        rows.append(row)
    return rows


def _plot_selector_metrics_heatmap(
    path: Path,
    rows: List[Dict[str, Any]],
    metrics: Sequence[str],
    *,
    trained_sites_only: bool = True,
    title: str = "baseline selector metrics",
) -> bool:
    """One heatmap: rows = application configs, columns = static selector metrics.

    Coverage / specificity do not depend on LR, so merging them beats four
    single-column heatmaps.
    """
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    from gradiend.visualizer.plot_style import disable_usetex_for_axis_text

    intervention_rows = [row for row in rows if row.get("kind") == "intervention"]
    if trained_sites_only:
        intervention_rows = [
            row
            for row in intervention_rows
            if str(row.get("site_scope") or "trained_sites") == "trained_sites"
        ]
    if not intervention_rows or not metrics:
        return False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target_classes = {
        str(row.get("target_class"))
        for row in intervention_rows
        if row.get("target_class") is not None
    }

    def row_label(row: Mapping[str, Any]) -> str:
        label = str(row["selector_label"])
        if len(target_classes) > 1:
            return f"{row.get('target_class')} | {label}"
        return label

    label_rows: Dict[str, Mapping[str, Any]] = {}
    for row in intervention_rows:
        label = row_label(row)
        if label not in label_rows:
            label_rows[label] = row
    labels = sorted(label_rows, key=lambda label: _configuration_sort_key(label_rows[label]))
    columns = list(metrics)
    column_labels = [m.replace("selector_", "").replace("_", " ") for m in columns]

    matrix: List[List[float]] = []
    for label in labels:
        row = label_rows[label]
        values: List[float] = []
        for metric in columns:
            value = _as_float(row.get(metric))
            values.append(float("nan") if value is None else value)
        matrix.append(values)

    fig_width = max(7.0, len(columns) * 1.6)
    fig_height = max(4.5, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    flat = [value for row in matrix for value in row if not math.isnan(value)]
    # Mix of coverage [0,1] and signed diffs → diverging around 0 when any signed col present.
    signed = any("_minus_" in m or m.endswith("_specificity_score") for m in columns)
    if signed and flat:
        limit = max(abs(value) for value in flat) or 1.0
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xlabel("baseline selector metric (independent of LR)")
    ax.set_ylabel("application configuration")
    ax.set_xticks(range(len(column_labels)), column_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(image, ax=ax)
    disable_usetex_for_axis_text(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_heatmap(
    path: Path,
    rows: List[Dict[str, Any]],
    metric: str,
    *,
    trained_sites_only: bool = True,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return False

    from gradiend.visualizer.plot_style import disable_usetex_for_axis_text

    intervention_rows = [row for row in rows if row.get("kind") == "intervention"]
    if trained_sites_only:
        intervention_rows = [
            row
            for row in intervention_rows
            if str(row.get("site_scope") or "trained_sites") == "trained_sites"
        ]
    if not intervention_rows:
        return False

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    target_classes = {
        str(row.get("target_class"))
        for row in intervention_rows
        if row.get("target_class") is not None
    }

    def row_label(row: Mapping[str, Any]) -> str:
        label = str(row["selector_label"])
        if len(target_classes) > 1:
            return f"{row.get('target_class')} | {label}"
        return label

    label_rows: Dict[str, Mapping[str, Any]] = {}
    for row in intervention_rows:
        label = row_label(row)
        if label not in label_rows:
            label_rows[label] = row
    labels = sorted(label_rows, key=lambda label: _configuration_sort_key(label_rows[label]))
    static_application_metric = _is_static_application_metric(metric)
    if static_application_metric:
        columns = [metric.replace("_", " ")]
    else:
        columns = list(dict.fromkeys(row["learning_rate"] for row in intervention_rows))
    matrix: List[List[float]] = []
    for label in labels:
        values: List[float] = []
        if static_application_metric:
            metric_values = [
                value
                for value in (
                    _as_float(row.get(metric))
                    for row in intervention_rows
                    if row_label(row) == label
                )
                if value is not None
            ]
            if metric_values:
                value = metric_values[0]
                if any(not math.isclose(other, value, rel_tol=1e-12, abs_tol=1e-12) for other in metric_values[1:]):
                    raise ValueError(
                        f"Static application metric {metric!r} varied across intervention values for "
                        f"{label!r}; selector coverage should be computed from baseline activations."
                    )
            else:
                value = None
            values.append(float("nan") if value is None else value)
        else:
            for learning_rate in columns:
                matching = [
                    row
                    for row in intervention_rows
                    if row_label(row) == label
                    and row["learning_rate"] == learning_rate
                ]
                value = _as_float(matching[0].get(metric)) if matching else None
                values.append(float("nan") if value is None else value)
        matrix.append(values)

    fig_width = max(7.0, len(columns) * 1.2)
    fig_height = max(4.5, len(labels) * 0.45)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    values = [value for row in matrix for value in row if not math.isnan(value)]
    if (metric.endswith("_delta") or "_minus_" in metric or metric.endswith("_specificity_score")) and values:
        limit = max(abs(value) for value in values) or 1.0
        image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit)
    else:
        image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_title(metric.replace("_", " "))
    if static_application_metric:
        ax.set_xlabel("baseline selector metric (independent of intervention value)")
    else:
        ax.set_xlabel("intervention value (evaluate_decoder learning_rate)")
    ax.set_ylabel("application configuration")
    ax.set_xticks(range(len(columns)), [_format_cell(value) for value in columns])
    ax.set_yticks(range(len(labels)), labels)
    fig.colorbar(image, ax=ax)
    # Keep labels literal; some local matplotlib configs enable TeX rendering.
    disable_usetex_for_axis_text(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def main(task: str = DEFAULT_TASK) -> None:
    task_cfg = TASK_CONFIGS[task]
    target_classes = [
        str(item)
        for item in task_cfg.get("target_classes", (task_cfg["target_class"],))
    ]
    global OUTPUT_DIR
    OUTPUT_DIR = Path(task_cfg["experiment_dir"]) / "intervention_sweep"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for sub in ("decoder_grids", "decoder_by_technique"):
        (OUTPUT_DIR / sub).mkdir(parents=True, exist_ok=True)
    if WRITE_DECODER_RAW_CSV:
        (OUTPUT_DIR / "decoder_raw").mkdir(parents=True, exist_ok=True)
    trainer = _make_trainer_for_task(task)

    print("\n=== ACTIEND intervention sweep ===")
    print(f"  task: {task}")
    print(f"  target classes: {', '.join(target_classes)}")
    print("  training or loading activation GRADIEND")
    trainer.train()

    model_with_gradiend = trainer.get_model()

    tokenizer = model_with_gradiend.tokenizer
    supports_selector_coverage = bool(getattr(model_with_gradiend.capabilities, "activation_selector_coverage", False))
    if not supports_selector_coverage:
        print("  selector coverage: skipped (model is not activation-space ACTIEND)")
    training_like_df, neutral_df = trainer._get_decoder_eval_dataframe(
        tokenizer,
        max_size_training_like=EVAL_MAX_SIZE,
        max_size_neutral=EVAL_MAX_SIZE,
    )
    rows: List[Dict[str, Any]] = []
    baseline_rows: Dict[str, Dict[str, Any]] = {}
    decoder_plot_paths: List[Path] = []
    decoder_raw_paths: List[Path] = []
    strategy_by_target_label: Dict[tuple[str, str], Dict[str, Any]] = {}

    for target_class in target_classes:
        feature_factor = float(
            derive_default_feature_factor(
                trainer,
                model_with_gradiend,
                class_name=target_class,
            )
        )
        print(f"  feature_factor for target {target_class}: {feature_factor:g}")

        strategies = _strategy_grid(feature_factor)
        for strategy in strategies:
            strategy_by_target_label[(target_class, strategy["label"])] = strategy
        for selector_config in strategies:
            selector_label = selector_config["label"]
            if supports_selector_coverage:
                coverage_columns = _selector_coverage_columns(
                    model_with_gradiend,
                    tokenizer,
                    training_like_df,
                    neutral_df,
                    target_class=target_class,
                    feature_factor=feature_factor,
                    selector_config=selector_config,
                    max_rows=EVAL_MAX_SIZE,
                )
                print(
                    f"  selector coverage {target_class} | {selector_label}: "
                    f"target={_format_cell(coverage_columns.get('selector_target_coverage'))} "
                    f"(scope={_format_cell(coverage_columns.get('selector_target_scope_coverage'))}), "
                    f"other={_format_cell(coverage_columns.get('selector_other_coverage'))} "
                    f"(scope={_format_cell(coverage_columns.get('selector_other_scope_coverage'))}), "
                    f"neutral={_format_cell(coverage_columns.get('selector_neutral_coverage'))} "
                    f"(scope={_format_cell(coverage_columns.get('selector_neutral_scope_coverage'))})"
                )
            else:
                coverage_columns = _empty_selector_coverage_columns()
            print(f"  evaluating decoder grid for {target_class} | {selector_label}")
            technique_result = trainer.evaluate_decoder(
                target_class=target_class,
                feature_factors=[feature_factor],
                lrs=list(LR_GRID),
                use_cache=USE_CACHE,
                plot=True,
                show=True,
                output_path=str(_decoder_grid_cache_path(selector_label, target_class=target_class)),
                raw_output_path=_decoder_raw_output_path(selector_label, target_class=target_class),
                plot_kwargs={
                    "output": str(_decoder_plot_output_path(selector_label, target_class=target_class)),
                    "show": True,
                    "title": f"{task} | {selector_label} | target={target_class}",
                },
                max_size_training_like=EVAL_MAX_SIZE,
                max_size_neutral=EVAL_MAX_SIZE,
                training_like_df=training_like_df,
                neutral_df=neutral_df,
                **_strategy_intervention_kwargs(selector_config),
            )
            decoder_plot_paths.extend(Path(path) for path in technique_result.get("plot_paths", []) if path)
            if technique_result.get("raw_output_path"):
                decoder_raw_paths.append(Path(str(technique_result["raw_output_path"])))
            if target_class not in baseline_rows:
                baseline_rows[target_class] = _baseline_row_from_decoder_result(
                    technique_result,
                    target_class=target_class,
                    feature_factor=feature_factor,
                )
                rows.append(baseline_rows[target_class])
            rows.extend(
                _rows_from_decoder_result(
                    technique_result,
                    selector_config=selector_config,
                    target_class=target_class,
                    baseline_row=baseline_rows[target_class],
                    coverage_columns=coverage_columns,
                )
            )

    supports_single_site_ablation = bool(getattr(model_with_gradiend.capabilities, "activation_module_ablation", False))
    if supports_single_site_ablation:
        activation_sites = list(getattr(model_with_gradiend, "activation_site_modules", []))
        for target_class in target_classes:
            ranked_all_sites = sorted(
                (
                    row for row in rows
                    if row["kind"] == "intervention"
                    and row.get("target_class") == target_class
                    and row.get("site_scope") == "trained_sites"
                ),
                key=_rank_probability_shift,
            )
            if not ranked_all_sites:
                continue
            site_reference = ranked_all_sites[0]
            site_strategy = strategy_by_target_label.get((target_class, str(site_reference["selector_label"])))
            if site_strategy is None or len(activation_sites) <= 1:
                continue
            print(
                f"  single-site ablation for target {target_class} best all-site setting: "
                f"{site_reference['selector_label']} "
                f"(full LR grid; reference value was {site_reference['learning_rate']:g})"
            )
            for module_name in activation_sites:
                site_label = f"{site_reference['selector_label']} | {module_name}"
                print(f"  evaluating single site {module_name}")
                site_coverage_columns = _selector_coverage_columns(
                    model_with_gradiend,
                    tokenizer,
                    training_like_df,
                    neutral_df,
                    target_class=target_class,
                    feature_factor=float(site_reference["feature_factor"]),
                    selector_config=site_strategy,
                    activation_modules=module_name,
                    max_rows=EVAL_MAX_SIZE,
                )
                site_result = trainer.evaluate_decoder(
                    target_class=target_class,
                    feature_factors=[float(site_reference["feature_factor"])],
                    lrs=list(LR_GRID),
                    use_cache=USE_CACHE,
                    plot=True,
                    show=True,
                    output_path=str(_decoder_grid_cache_path(site_label, target_class=target_class)),
                    raw_output_path=_decoder_raw_output_path(site_label, target_class=target_class),
                    plot_kwargs={
                        "output": str(_decoder_plot_output_path(site_label, target_class=target_class)),
                        "show": True,
                        "title": f"{task} | {site_label} | target={target_class}",
                    },
                    max_size_training_like=EVAL_MAX_SIZE,
                    max_size_neutral=EVAL_MAX_SIZE,
                    training_like_df=training_like_df,
                    neutral_df=neutral_df,
                    **_strategy_intervention_kwargs(site_strategy, activation_modules=module_name),
                )
                decoder_plot_paths.extend(Path(path) for path in site_result.get("plot_paths", []) if path)
                if site_result.get("raw_output_path"):
                    decoder_raw_paths.append(Path(str(site_result["raw_output_path"])))
                rows.extend(
                    _rows_from_decoder_result(
                        site_result,
                        selector_config=site_strategy,
                        target_class=target_class,
                        baseline_row=baseline_rows[target_class],
                        coverage_columns=site_coverage_columns,
                        activation_modules=module_name,
                        site_scope=f"single:{module_name}",
                        selector_label=site_label,
                    )
                )

    ranked = sorted(
        (row for row in rows if row["kind"] == "intervention"),
        key=_rank_probability_shift,
    )
    for index, row in enumerate(ranked, start=1):
        row["rank"] = index
    for target_class in target_classes:
        target_ranked = [
            row for row in ranked
            if row.get("target_class") == target_class
        ]
        for index, row in enumerate(target_ranked, start=1):
            row["target_rank"] = index

    csv_path = OUTPUT_DIR / "actiend_intervention_sweep.csv"
    markdown_path = OUTPUT_DIR / "actiend_intervention_sweep.md"
    specificity_summary_path = OUTPUT_DIR / "actiend_specificity_summary.md"
    lms_plot_path = OUTPUT_DIR / "actiend_lms_heatmap.png"
    probability_plot_path = OUTPUT_DIR / "actiend_target_probability_delta_heatmap.png"
    probability_specificity_plot_path = OUTPUT_DIR / "actiend_probability_specificity_heatmap.png"
    selector_metrics_plot_path = OUTPUT_DIR / "actiend_selector_metrics_heatmap.png"

    _write_csv(csv_path, rows)
    _write_markdown_table(markdown_path, ranked)
    try:
        _write_specificity_summary(
            specificity_summary_path,
            ranked,
            target_class=", ".join(target_classes),
        )
    except Exception as exc:
        print(f"  WARNING: failed to write specificity summary: {exc}")

    def _safe_heatmap(path: Path, metric: str) -> bool:
        try:
            return _plot_heatmap(path, rows, metric, trained_sites_only=True)
        except Exception as exc:
            print(f"  WARNING: heatmap {path.name} failed: {exc}")
            return False

    wrote_lms = _safe_heatmap(lms_plot_path, "lms")
    wrote_probability = _safe_heatmap(probability_plot_path, "target_probability_delta")
    wrote_probability_specificity = _safe_heatmap(
        probability_specificity_plot_path,
        "probability_specificity_score",
    )
    wrote_selector_metrics = False
    try:
        wrote_selector_metrics = _plot_selector_metrics_heatmap(
            selector_metrics_plot_path,
            rows,
            (
                "selector_neutral_coverage",
                "selector_neutral_scope_coverage",
                "selector_target_minus_neutral",
                "selector_scope_target_minus_neutral",
            ),
            trained_sites_only=True,
            title="baseline selector metrics (coverage / target−neutral)",
        )
    except Exception as exc:
        print(f"  WARNING: heatmap {selector_metrics_plot_path.name} failed: {exc}")

    best = ranked[0] if ranked else rows[0]
    print("\n=== Sweep complete ===")
    print(
        f"  best by P({best.get('target_class')}) shift after LMS prefilter "
        f"(selector specificity tie-break): "
        f"{best['selector_label']} at value {best['learning_rate']:g} "
        f"(delta={_format_cell(best.get('target_probability_delta'))}, "
        f"lms_ratio={_format_cell(best.get('lms_ratio'))}, "
        f"neutral_scope_coverage={_format_cell(best.get('selector_neutral_scope_coverage'))})"
    )
    print(f"  csv: {csv_path}")
    print(f"  table: {markdown_path}")
    print(f"  specificity summary: {specificity_summary_path}")
    if decoder_plot_paths:
        print(f"  decoder plots by technique: {decoder_plot_paths[0].parent} ({len(decoder_plot_paths)} plots)")
    if decoder_raw_paths:
        print(f"  decoder raw per-sample CSVs: {decoder_raw_paths[0].parent} ({len(decoder_raw_paths)} files)")
    if wrote_lms:
        print(f"  LMS heatmap: {lms_plot_path}")
    if wrote_probability:
        print(f"  target probability delta heatmap: {probability_plot_path}")
    if wrote_probability_specificity:
        print(f"  probability specificity heatmap: {probability_specificity_plot_path}")
    if wrote_selector_metrics:
        print(f"  selector metrics heatmap: {selector_metrics_plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=list(SUPPORTED_TASKS),
        default=DEFAULT_TASK,
        help=f"Task to sweep (default: {DEFAULT_TASK}).",
    )
    cli_args = parser.parse_args()
    main(task=cli_args.task)
