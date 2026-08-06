"""
Test ``source="both"`` by learning 3SG↔3PL from **3SG-only** factual rows.

Classic GRADIEND wants both 3SG→3PL and 3PL→3SG rows so the encoder sees +1 and −1.
With ``source="both"``, odd batches compile to the alternative pole (fac↔alt swap),
so 3SG→3PL rows alone are enough:

- even batches: encode g(3SG), label +1, target = g(3SG) − g(3PL)
- odd batches:  encode g(3PL), label −1, target = g(3PL) − g(3SG)

By default this uses the Wikipedia English pronoun CSVs from
:mod:`gradiend.examples.create_english_pronoun_data` (~1000 rows/class).
Set ``USE_TOY_CORPUS=True`` only for a tiny smoke on the 76 artificial sentences
from :mod:`gradiend.examples.start_workflow` (that path yields ~14 one-pole rows).

Run from the repo root:

    python -m gradiend.examples.train_source_both_one_pole

Requires (default path): ``pip install gradiend[data]`` and a one-time Wikipedia
filter via ``ensure_english_pronoun_data`` (cached under ``data/english_pronouns/``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Union

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

from gradiend import (
    TextFilterConfig,
    TextPredictionDataCreator,
    TextPredictionTrainer,
    TrainingArguments,
)
from gradiend.examples.create_english_pronoun_data import (
    DEFAULT_OUTPUT_DIR,
    NEUTRAL_EXCLUDE_ENGLISH_PRONOUNS,
    ensure_english_pronoun_data,
)
from gradiend.examples.start_workflow import ARTIFICIAL_TEXTS, NEUTRAL_EXCLUDE
from gradiend.trainer.core.dataset import (
    _both_side_for_batch,
    _swap_factual_alternative_batch,
)
from gradiend.trainer.core.unified_schema import (
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_SPLIT,
    UNIFIED_TRANSITION,
)


# --- knobs -----------------------------------------------------------------
# False (default): Wikipedia pronoun CSVs (~1000/class). True: 76-sentence toy list.
USE_TOY_CORPUS = False
COMPARE_FACTUAL_SOURCE = True
MODEL = "bert-base-uncased"
MAX_STEPS = 500
MAX_SEEDS = 1
TRAIN_BATCH_SIZE = 4
POS_CLASS = "3SG"
NEG_CLASS = "3PL"
DATA_DIR = DEFAULT_OUTPUT_DIR
# ---------------------------------------------------------------------------

DataLike = Union[str, Path, pd.DataFrame, Mapping[str, pd.DataFrame]]


def _keep_one_pole_factual(trainer: TextPredictionTrainer) -> None:
    """Deprecated helper — use ``target_classes=[A]`` + ``counterfactual_classes``."""
    trainer.config.target_classes = [POS_CLASS]
    trainer.config.counterfactual_classes = "all"
    if trainer._data_loaded:
        trainer._data_loaded = False
        trainer.data = None
        trainer.class_datasets = None
        trainer._combined_data = None
        trainer._combined_data_template = None
    trainer._ensure_data()
    combined = trainer.combined_data
    if combined is None or combined.empty:
        raise RuntimeError(f"No {POS_CLASS} factual rows after one-pole filter.")
    n = len(combined)
    trans = {
        str(k).replace("\u2192", "->"): int(v)
        for k, v in combined[UNIFIED_TRANSITION].value_counts().items()
    }
    print(f"  one-pole filter (target={POS_CLASS!r}): {n} rows ({trans})")


def _probe_batch_polarity(trainer: TextPredictionTrainer, model, *, n_batches: int = 16) -> None:
    """Show that one-pole feature rows still yield ±1 labels under source='both'."""
    raw = trainer.create_training_data(model, split="train", batch_size=1)
    n_groups = int(getattr(raw, "n_balance_groups", 1) or 1)
    n = min(n_batches, len(raw))
    print(f"\n--- Polarity probe (n_balance_groups={n_groups}, first {n} batches) ---")

    feature_raw = [
        int(raw[i]["label"])
        for i in range(n)
        if not bool(raw[i].get("is_identity_transition"))
    ]
    print(f"  feature-row raw labels: {feature_raw}")
    if not feature_raw or set(feature_raw) != {1}:
        raise RuntimeError(
            f"Expected feature-row raw labels +1 only, got {set(feature_raw) or 'none'}"
        )

    print("  after source='both' compile (pole orthogonal to balance groups):")
    feature_compiled: list[int] = []
    for i in range(n):
        item = dict(raw[i])
        side = _both_side_for_batch(i, n_balance_groups=n_groups)
        if side == "alternative":
            item = _swap_factual_alternative_batch(item)
        is_identity = bool(item.get("is_identity_transition"))
        lab = int(item["label"])
        print(
            f"    batch {i}: side={side:11s} identity={is_identity} "
            f"label={lab:+d}  factual_id={item['factual_id']}  "
            f"token={item['factual_token']!r}"
        )
        if not is_identity:
            feature_compiled.append(lab)
    if set(feature_compiled) != {1, -1}:
        raise RuntimeError(
            f"Expected feature compiled labels {{+1, -1}}, got {set(feature_compiled) or 'none'}"
        )


def _run(
    *,
    data: DataLike,
    neutral: DataLike,
    neutral_exclude: list[str],
    source: str,
    experiment_dir: str,
    fail_on_non_convergence: bool,
    probe: bool,
) -> dict[str, Any]:
    args = TrainingArguments(
        source=source,
        target="diff",
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_steps=25,
        max_steps=MAX_STEPS,
        max_seeds=MAX_SEEDS,
        learning_rate=1e-3,
        experiment_dir=experiment_dir,
        use_cache=False,
        fail_on_non_convergence=fail_on_non_convergence,
        add_neutral_identity_transitions=True,
    )
    trainer = TextPredictionTrainer(
        model=MODEL,
        data=data,
        # Clean one-pole: target_classes=[A] + counterfactual_classes (no factual_classes).
        target_classes=[POS_CLASS],
        all_classes=[POS_CLASS, NEG_CLASS],
        counterfactual_classes="all",
        neutral_data=neutral,
        eval_neutral_additional_excluded_words=neutral_exclude,
        max_counterfactuals_per_sentence=2,
        img_format="png",
        args=args,
    )
    print(f"\n=== Training source={source!r} on {POS_CLASS}-only factual data ===")
    trainer._ensure_data()
    print(f"  one-pole unified rows: {len(trainer.combined_data)}")
    print(
        f"  factual={sorted(trainer.combined_data[UNIFIED_FACTUAL_CLASS].astype(str).unique())} "
        f"alt={sorted(trainer.combined_data[UNIFIED_ALTERNATIVE_CLASS].astype(str).unique())}"
    )

    model = trainer.get_model()
    if probe:
        _probe_batch_polarity(trainer, model)

    trainer.train()
    trainer.plot_training_convergence()

    enc = trainer.evaluate_encoder(plot=True, split="test")
    means = enc.get("mean_by_class") or {}
    corr = enc.get("correlation")
    print(f"  encoder correlation: {corr}")
    print(f"  mean encoded by label (+1/-1): {means}")
    return {"correlation": corr, "means": means}


def _load_toy_data() -> tuple[DataLike, DataLike, list[str]]:
    """Tiny smoke corpus: 76 sentences → typically ~14 one-pole training rows."""
    creator = TextPredictionDataCreator(
        base_data=ARTIFICIAL_TEXTS,
        feature_targets=[
            TextFilterConfig(targets=["he", "she", "it"], id=POS_CLASS),
            TextFilterConfig(targets=["they"], id=NEG_CLASS),
        ],
    )
    training_by_class = creator.generate_training_data(max_size_per_class=50)
    neutral = creator.generate_neutral_data(
        additional_excluded_words=NEUTRAL_EXCLUDE,
        max_size=50,
    )
    print(
        "WARNING: USE_TOY_CORPUS=True — only "
        f"{sum(len(v) for v in training_by_class.values())} filtered matches from "
        f"{len(ARTIFICIAL_TEXTS)} artificial sentences. "
        "Set USE_TOY_CORPUS=False for Wikipedia-scale data."
    )
    print("Per-class sizes:", {k: len(v) for k, v in training_by_class.items()})
    return training_by_class, neutral, list(NEUTRAL_EXCLUDE)


def _load_wikipedia_pronoun_data() -> tuple[DataLike, DataLike, list[str]]:
    training_path, neutral_path = ensure_english_pronoun_data(output_dir=DATA_DIR)
    training = pd.read_csv(training_path)
    neutral = pd.read_csv(neutral_path)
    by_class = (
        training["label_class"].value_counts().astype(int).to_dict()
        if "label_class" in training.columns
        else {"rows": len(training)}
    )
    print(f"Loaded Wikipedia pronoun data from {training_path}")
    print(f"  training rows={len(training)} by label_class={by_class}")
    print(f"  neutral rows={len(neutral)} from {neutral_path}")
    return training_path, neutral_path, list(NEUTRAL_EXCLUDE_ENGLISH_PRONOUNS)


if __name__ == "__main__":
    if USE_TOY_CORPUS:
        data, neutral, exclude = _load_toy_data()
    else:
        data, neutral, exclude = _load_wikipedia_pronoun_data()

    both = _run(
        data=data,
        neutral=neutral,
        neutral_exclude=exclude,
        source="both",
        experiment_dir="runs/examples/train_source_both_one_pole",
        fail_on_non_convergence=True,
        probe=True,
    )

    if COMPARE_FACTUAL_SOURCE:
        factual = _run(
            data=data,
            neutral=neutral,
            neutral_exclude=exclude,
            source="factual",
            experiment_dir="runs/examples/train_source_both_one_pole_factual_baseline",
            fail_on_non_convergence=False,
            probe=False,
        )
        print("\n=== Comparison ===")
        print(f"  source='both'     correlation={both['correlation']}, means={both['means']}")
        print(f"  source='factual'  correlation={factual['correlation']}, means={factual['means']}")
        print(
            "  Expectation: 'both' gets +/-1 labels via batch swap and should correlate; "
            f"'factual' on {POS_CLASS}-only has only +1 labels => correlation ~ 0."
        )
