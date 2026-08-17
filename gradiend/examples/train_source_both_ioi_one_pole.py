"""
``source="both"`` one-pole demo for an IOI-style **copy** contrast.

Circuit framing (not a demographic peer pair):

- same masked prompt ``x``
- factual token = **IO** (correct copy target)
- alternative token = **SUBJECT** (incorrect repeated name)
- contrast: ``∇L(IO|x) ↔ ∇L(SUBJECT|x)``

Only **IO→SUBJECT** rows are built (one pole via ``target_classes=["IO"]`` +
``counterfactual_classes="all"``). With ``source="both"``, alternate batches
swap poles so the encoder also sees SUBJECT at label ``-1``. Neutral identity
rows (label ``0``) are interleaved via balance groups; poles stay orthogonal
to that cycling.

After training, a **causal** check runs ``evaluate_decoder`` / ``rewrite_base_model``
to strengthen IO and reports per-row P(IO) vs P(SUBJECT) before/after the rewrite.

Names come from ``aieng-lab/namexact`` (train/val/test splits preserved).
Neutral eval uses ``aieng-lab/biasneutral``. Templates vary place / object / verb.

Run from the repo root:

    python -m gradiend.examples.train_source_both_ioi_one_pole
"""
from __future__ import annotations

import os
from typing import Any, Iterable, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from gradiend import TextPredictionTrainer, TrainingArguments
from gradiend.examples.train_gender_en import read_geneutral, read_namexact
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
MODEL = "bert-base-uncased"
NUM_TRAIN_EPOCHS = 5
MAX_STEPS = 2000  # -1 = use NUM_TRAIN_EPOCHS; set >0 to override epochs
MAX_SEEDS = 1
TRAIN_BATCH_SIZE = 4
EVAL_STEPS = 50
# Rows per GRADIEND split (sampled name pairs × template slots).
ROWS_PER_SPLIT = {"train": 8000, "validation": 800, "test": 800}
NEUTRAL_MAX_ROWS = 2000
DATA_SEED = 0
POS_CLASS = "IO"
NEG_CLASS = "SUBJECT"
# Cap for decoder grid + before/after probability scoring.
DECODER_EVAL_MAX_SIZE = 64
# ---------------------------------------------------------------------------

# namexact uses "val"; GRADIEND expects "validation".
_NAMEXACT_TO_GRADIEND_SPLIT = {
    "train": "train",
    "val": "validation",
    "validation": "validation",
    "test": "test",
}

_PLACES = [
    "the shop", "the park", "the office", "the museum", "the cafe", "the library",
    "the station", "the market", "the school", "the hospital", "the bakery", "the gallery",
    "the theatre", "the stadium", "the beach", "the airport", "the hotel", "the restaurant",
    "the garden", "the studio", "the warehouse", "the courtroom", "the laboratory", "the campus",
    "the bookstore", "the pharmacy", "the gym", "the cinema", "the harbor", "the plaza",
    "the workshop", "the clinic", "the farm", "the factory", "the cathedral", "the zoo",
]

_OBJECTS = [
    "a book", "a gift", "a letter", "a key", "a drink", "a ticket",
    "a map", "a note", "a parcel", "a flower", "a photo", "a pen",
    "a bag", "a box", "a card", "a message", "a present", "a bottle",
    "a sandwich", "a coffee", "a report", "a folder", "a laptop", "a phone",
    "a coat", "a hat", "an umbrella", "a scarf", "a camera", "a sketch",
    "a receipt", "a passport", "a brochure", "a magazine", "a basket", "a tray",
]

_VERBS = [
    "gave", "handed", "sent", "brought", "passed", "offered",
    "delivered", "showed", "lent", "returned", "tossed", "sold",
    "mailed", "carried", "presented", "forwarded", "shipped", "granted",
    "awarded", "donated", "shared", "supplied", "provided", "assigned",
]


def _capitalize(name: str) -> str:
    return name[:1].upper() + name[1:]


def _is_single_token_name(tokenizer, name: str) -> bool:
    ids = tokenizer.encode(str(name).lower(), add_special_tokens=False)
    return len(ids) == 1


def load_namexact_names_by_split(
    *,
    model_name: str = MODEL,
) -> dict[str, list[str]]:
    """Load single-token namexact names, keyed by GRADIEND split."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    names_df = read_namexact(split="train")
    if "split" not in names_df.columns or "name" not in names_df.columns:
        raise RuntimeError(f"Unexpected namexact columns: {list(names_df.columns)}")

    by_split: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    for raw_split, group in names_df.groupby(names_df["split"].astype(str), sort=False):
        gradiend_split = _NAMEXACT_TO_GRADIEND_SPLIT.get(str(raw_split))
        if gradiend_split is None:
            continue
        names = []
        for name in group["name"].astype(str).tolist():
            lowered = name.lower().strip()
            if lowered and _is_single_token_name(tokenizer, lowered):
                names.append(lowered)
        # Preserve frequency order from namexact; drop duplicates.
        seen = set()
        ordered = []
        for name in names:
            if name not in seen:
                seen.add(name)
                ordered.append(name)
        by_split[gradiend_split] = ordered

    for split, names in by_split.items():
        if len(names) < 2:
            raise RuntimeError(
                f"Need at least 2 single-token namexact names for split={split!r}, got {len(names)}"
            )
    return by_split


def _iter_ioi_rows_for_split(
    names: Sequence[str],
    *,
    split: str,
    n_rows: int,
    places: Sequence[str],
    objects: Sequence[str],
    verbs: Sequence[str],
    rng: np.random.Generator,
) -> Iterable[dict[str, str]]:
    """Sample one-pole IO→SUBJECT rows for one split."""
    name_arr = np.asarray(list(names), dtype=object)
    place_arr = np.asarray(list(places), dtype=object)
    object_arr = np.asarray(list(objects), dtype=object)
    verb_arr = np.asarray(list(verbs), dtype=object)

    for _ in range(int(n_rows)):
        a, b = rng.choice(name_arr, size=2, replace=False)
        place = str(rng.choice(place_arr))
        obj = str(rng.choice(object_arr))
        verb = str(rng.choice(verb_arr))
        a_cap, b_cap = _capitalize(str(a)), _capitalize(str(b))
        if bool(rng.integers(0, 2)):
            # ABBA: subject A repeats; IO is B.
            pattern = "ABBA"
            masked = f"When {a_cap} and {b_cap} went to {place}, {a_cap} {verb} {obj} to [MASK]"
            io_name, subject_name = str(b), str(a)
        else:
            # BABA: subject B repeats; IO is A.
            pattern = "BABA"
            masked = f"When {a_cap} and {b_cap} went to {place}, {b_cap} {verb} {obj} to [MASK]"
            io_name, subject_name = str(a), str(b)
        yield {
            "masked": masked,
            "label": io_name,
            "label_class": POS_CLASS,
            "alternative": subject_name,
            "alternative_class": NEG_CLASS,
            "pattern": pattern,
            "split": split,
        }


def build_ioi_one_pole_data(
    *,
    rows_per_split: dict[str, int] | None = None,
    neutral_max_rows: int = NEUTRAL_MAX_ROWS,
    seed: int = DATA_SEED,
    model_name: str = MODEL,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Build one-pole IOI training rows from namexact + biasneutral eval texts."""
    rows_per_split = dict(rows_per_split or ROWS_PER_SPLIT)
    names_by_split = load_namexact_names_by_split(model_name=model_name)
    rng = np.random.default_rng(seed)

    rows: list[dict[str, str]] = []
    for split, n_rows in rows_per_split.items():
        if n_rows <= 0:
            continue
        if split not in names_by_split:
            raise KeyError(f"Unknown split {split!r}; expected one of {sorted(names_by_split)}")
        rows.extend(
            _iter_ioi_rows_for_split(
                names_by_split[split],
                split=split,
                n_rows=n_rows,
                places=_PLACES,
                objects=_OBJECTS,
                verbs=_VERBS,
                rng=rng,
            )
        )

    training = pd.DataFrame(rows)
    if training.empty:
        raise RuntimeError("IOI builder produced no rows.")
    if not (training["label_class"] == POS_CLASS).all():
        raise RuntimeError("IOI builder must emit IO-only factual rows.")
    if not (training["alternative_class"] == NEG_CLASS).all():
        raise RuntimeError("IOI builder must emit SUBJECT-only alternatives.")

    all_names = sorted({n for names in names_by_split.values() for n in names})
    neutral = read_geneutral(max_size=neutral_max_rows)
    if "text" not in neutral.columns:
        raise RuntimeError(f"biasneutral missing text column; got {list(neutral.columns)}")
    # Drop obvious name hits so neutral stays feature-independent.
    name_set = set(all_names)
    keep = []
    for text in neutral["text"].astype(str).tolist():
        tokens = set(text.lower().replace("'", " ").split())
        if tokens.isdisjoint(name_set):
            keep.append(text)
    neutral = pd.DataFrame({"text": keep})
    if len(neutral) < 32:
        raise RuntimeError(
            f"Too few biasneutral sentences after name exclusion ({len(neutral)}); "
            "increase NEUTRAL_MAX_ROWS or relax filtering."
        )
    return training, neutral, all_names


def _print_data_summary(
    training: pd.DataFrame,
    *,
    names_by_split: dict[str, list[str]] | None = None,
) -> None:
    print(f"IOI one-pole rows: {len(training)}")
    print(f"  patterns: {training['pattern'].value_counts().to_dict()}")
    print(f"  splits: {training['split'].value_counts().to_dict()}")
    print(f"  transitions: {(training['label_class'] + '->' + training['alternative_class']).unique().tolist()}")
    if names_by_split is not None:
        print(
            "  namexact single-token names by split: "
            + str({k: len(v) for k, v in names_by_split.items()})
        )
    print(f"  template slots: places={len(_PLACES)} objects={len(_OBJECTS)} verbs={len(_VERBS)}")
    print("  examples:")
    for _, row in training.head(3).iterrows():
        print(f"    [{row['pattern']}/{row['split']}] {row['masked']}")
        print(f"      IO={row['label']!r}  SUBJECT={row['alternative']!r}")


def _probe_batch_polarity(trainer: TextPredictionTrainer, model, *, n_batches: int = 16) -> None:
    """Check one-pole feature rows and source=both poles (with neutrals allowed)."""
    raw = trainer.create_training_data(model, split="train", batch_size=1)
    n_groups = int(getattr(raw, "n_balance_groups", 1) or 1)
    n = min(n_batches, len(raw))
    print(f"\n--- Polarity probe (n_balance_groups={n_groups}, first {n} batches) ---")

    feature_raw_labels: list[int] = []
    for i in range(n):
        item = raw[i]
        if bool(item.get("is_identity_transition")):
            continue
        feature_raw_labels.append(int(item["label"]))
    print(f"  feature-row raw labels: {feature_raw_labels}")
    if not feature_raw_labels or set(feature_raw_labels) != {1}:
        raise RuntimeError(
            f"Expected feature-row raw labels +1 only ({POS_CLASS} factual), "
            f"got {set(feature_raw_labels) or 'none'} "
            f"(neutral identity rows may be interleaved and are ignored here)"
        )

    print(
        "  after source='both' compile "
        "(pole = visit // n_groups; orthogonal to feature/neutral balance):"
    )
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
            f"Expected feature compiled labels {{+1, -1}} under source='both', "
            f"got {set(feature_compiled) or 'none'} "
            f"(n_balance_groups={n_groups}; neutrals must not lock the feature pole)"
        )


def _assert_still_one_pole(trainer: TextPredictionTrainer) -> None:
    trainer._ensure_data()
    combined = trainer.combined_data
    if combined is None:
        raise RuntimeError("expected combined_data")
    fac = set(combined[UNIFIED_FACTUAL_CLASS].astype(str).unique())
    alt = set(combined[UNIFIED_ALTERNATIVE_CLASS].astype(str).unique())
    if fac != {POS_CLASS} or alt != {NEG_CLASS}:
        raise RuntimeError(f"Expected only {POS_CLASS}->{NEG_CLASS}, got factual={fac} alt={alt}")
    print(
        f"  confirmed one-pole unified data: {len(combined)} rows, "
        f"splits={combined[UNIFIED_SPLIT].value_counts().astype(int).to_dict()}, "
        f"transitions={combined[UNIFIED_TRANSITION].value_counts().astype(int).to_dict()}"
    )


def _io_panel_probs(stats: dict[str, Any]) -> dict[str, float]:
    """Row-wise probs on the IO factual panel: P(IO) and P(SUBJECT) per masked prompt."""
    pbd = stats.get("probs_by_dataset") or {}
    panel = pbd.get(POS_CLASS) or (next(iter(pbd.values())) if pbd else {})
    return {str(k): float(v) for k, v in panel.items()}


def _run_causal_decoder_test(trainer: TextPredictionTrainer) -> dict[str, Any]:
    """Strengthen IO via decoder rewrite; report P(IO)/P(SUBJECT) before and after."""
    print(f"\n=== Causal decoder test (strengthen {POS_CLASS}) ===")
    dec = trainer.evaluate_decoder(
        plot=True,
        target_class=POS_CLASS,
        split="test",
        max_size=DECODER_EVAL_MAX_SIZE,
        use_cache=False
    )
    summary = dec.get(POS_CLASS) or {}
    print(
        f"  selected: value={summary.get('value')} "
        f"feature_factor={summary.get('feature_factor')} "
        f"lr={summary.get('learning_rate')}"
    )
    print(f"  lms={summary.get('lms')}  base_lms={summary.get('base_lms')}")

    mwg = trainer.get_model()
    base = trainer.evaluate_base_model(
        mwg.base_model,
        mwg.tokenizer,
        use_cache=False,
        max_size_training_like=DECODER_EVAL_MAX_SIZE,
        max_size_neutral=DECODER_EVAL_MAX_SIZE,
    )
    changed = trainer.rewrite_base_model(decoder_results=dec, target_class=POS_CLASS)
    after = trainer.evaluate_base_model(
        changed,
        mwg.tokenizer,
        use_cache=False,
        max_size_training_like=DECODER_EVAL_MAX_SIZE,
        max_size_neutral=DECODER_EVAL_MAX_SIZE,
    )
    before_p = _io_panel_probs(base)
    after_p = _io_panel_probs(after)
    print(
        f"  before rewrite: P({POS_CLASS})={before_p.get(POS_CLASS)}  "
        f"P({NEG_CLASS})={before_p.get(NEG_CLASS)}"
    )
    print(
        f"  after  rewrite: P({POS_CLASS})={after_p.get(POS_CLASS)}  "
        f"P({NEG_CLASS})={after_p.get(NEG_CLASS)}"
    )
    print(
        f"  Expectation: strengthen {POS_CLASS} raises P({POS_CLASS}) and lowers "
        f"P({NEG_CLASS}) on the same masked IOI prompts."
    )
    return {
        "decoder": summary,
        "before": before_p,
        "after": after_p,
    }


def _run(
    *,
    training: pd.DataFrame,
    neutral: pd.DataFrame,
    excluded_names: list[str],
    experiment_dir: str,
    fail_on_non_convergence: bool,
    probe: bool,
) -> dict[str, Any]:
    args = TrainingArguments(
        source="both",
        target="diff",
        train_batch_size=TRAIN_BATCH_SIZE,
        eval_steps=EVAL_STEPS,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        max_steps=MAX_STEPS,
        max_seeds=MAX_SEEDS,
        encoder_eval_max_size=100,
        learning_rate=1e-5,
        experiment_dir=experiment_dir,
        use_cache=False,
        fail_on_non_convergence=fail_on_non_convergence,
        # Neutrals stay on: source=both poles are orthogonal to balance groups.
        add_neutral_identity_transitions=True,
    )
    trainer = TextPredictionTrainer(
        model=MODEL,
        data=training,
        # Clean one-pole: single + class + CFs (no factual_classes).
        target_classes=[POS_CLASS],
        all_classes=[POS_CLASS, NEG_CLASS],
        counterfactual_classes="all",  # -> [SUBJECT]
        neutral_data=neutral,
        eval_neutral_additional_excluded_words=excluded_names,
        # Per-row IO vs SUBJECT names (not a fixed pronoun list).
        decoder_eval_targets="label",
        # One-pole data: only an IO factual panel; score P(IO)/P(SUBJECT) there.
        decoder_eval_prob_on_other_class=False,
        img_format="png",
        args=args,
    )
    print(f"\n=== Training source='both' on {POS_CLASS}-only factual IOI data ===")
    _assert_still_one_pole(trainer)

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

    causal = _run_causal_decoder_test(trainer)
    return {"correlation": corr, "means": means, "causal": causal}


if __name__ == "__main__":
    names_by_split = load_namexact_names_by_split()
    training, neutral, excluded_names = build_ioi_one_pole_data()
    _print_data_summary(training, names_by_split=names_by_split)
    print(f"Neutral rows (biasneutral, name-filtered): {len(neutral)}")

    _run(
        training=training,
        neutral=neutral,
        excluded_names=excluded_names,
        experiment_dir="runs/examples/train_source_both_ioi_one_pole",
        fail_on_non_convergence=False,
        probe=True,
    )
