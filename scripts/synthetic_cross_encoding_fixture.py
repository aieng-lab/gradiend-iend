"""Doc-only synthetic cross-encoding fixture (not part of the gradiend package).

Used by ``scripts/generate_cross_encoding_matrix_doc_figures.py`` and
``tests/test_synthetic_cross_encoding_fixture.py``.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from gradiend.trainer.core.unified_schema import transition_id

RACE_CLASSES: Tuple[str, ...] = ("white", "black", "asian")
GENDER_CLASSES: Tuple[str, ...] = ("he", "she")

FEATURE_ORDER: Tuple[str, ...] = RACE_CLASSES + GENDER_CLASSES
FEATURE_CLASSES: Tuple[str, ...] = FEATURE_ORDER

FEATURE_PLOT_GROUPS: Dict[str, List[str]] = {
    "Race": list(RACE_CLASSES),
    "English Gender": list(GENDER_CLASSES),
}

FEATURE_LABELS: Dict[str, str] = {
    "white": "White",
    "black": "Black",
    "asian": "Asian",
    "he": "he",
    "she": "she",
}

PAIR_BY_ID: Dict[str, Tuple[str, str]] = {
    "race_white_asian": ("white", "asian"),
    "race_black_asian": ("black", "asian"),
    "race_white_black": ("white", "black"),
    "gender_he_she": ("he", "she"),
}

TRAINER_ORDER: Tuple[str, ...] = tuple(PAIR_BY_ID.keys())

TRAINER_PLOT_GROUPS: Dict[str, List[str]] = {
    "Race": ["race_white_asian", "race_black_asian", "race_white_black"],
    "English Gender": ["gender_he_she"],
}

TRAINER_LABELS: Dict[str, str] = {
    "race_white_asian": "White↔Asian",
    "race_black_asian": "Black↔Asian",
    "race_white_black": "White↔Black",
    "gender_he_she": "he↔she",
}

SOURCE_BY_ID: Dict[str, str] = {trainer_id: "alternative" for trainer_id in TRAINER_ORDER}

_EXPLICIT_ENCODED: Dict[Tuple[str, str, str], float] = {
    ("race_white_asian", "white", "asian"): -0.60,
    ("race_white_asian", "black", "asian"): -0.40,
    ("race_white_asian", "white", "black"): 0.80,
    ("race_white_asian", "black", "white"): -0.70,
    ("race_white_asian", "asian", "white"): -0.55,
    ("race_white_asian", "asian", "black"): 0.50,
    ("race_black_asian", "white", "asian"): -0.50,
    ("race_black_asian", "black", "asian"): -0.70,
    ("race_black_asian", "black", "white"): 0.75,
    ("race_black_asian", "white", "black"): -0.65,
    ("race_black_asian", "asian", "white"): 0.45,
    ("race_black_asian", "asian", "black"): -0.60,
    ("race_white_black", "white", "black"): 0.75,
    ("race_white_black", "black", "white"): -0.65,
    ("race_white_black", "white", "asian"): 0.15,
    ("race_white_black", "black", "asian"): -0.10,
    ("race_white_black", "asian", "white"): -0.08,
    ("race_white_black", "asian", "black"): 0.12,
    ("gender_he_she", "he", "she"): 0.88,
    ("gender_he_she", "she", "he"): 0.82,
}

WORKED_DIAGONAL: Dict[str, str] = {
    "anchor": "asian",
    "column": "asian",
    "alignment": "counterfactual",
}

WORKED_OFF_DIAGONAL: Dict[str, str] = {
    "anchor": "white",
    "column": "asian",
    "alignment": "counterfactual",
}

WORKED_NEGATIVE_DIAGONAL: Dict[str, str] = {
    "anchor": "white",
    "column": "white",
    "alignment": "counterfactual",
}


def parse_transition_id(transition: str) -> Tuple[str, str]:
    for sep in ("→", "->"):
        if sep in transition:
            left, right = transition.split(sep, 1)
            return left.strip(), right.strip()
    raise ValueError(f"Not a transition id: {transition!r}")


def feature_family(feature: str) -> str:
    if feature in RACE_CLASSES:
        return "race"
    if feature in GENDER_CLASSES:
        return "gender"
    raise ValueError(feature)


def synthetic_transition_order() -> List[str]:
    race = list(RACE_CLASSES)
    gender = list(GENDER_CLASSES)
    ordered: List[str] = []
    for src in race:
        for tgt in race:
            if src != tgt:
                ordered.append(transition_id(src, tgt))
    ordered.extend([transition_id("he", "she"), transition_id("she", "he")])
    for src in race:
        for tgt in gender:
            ordered.append(transition_id(src, tgt))
    for src in gender:
        for tgt in race:
            ordered.append(transition_id(src, tgt))
    return ordered


def _family_of(feature: str) -> str:
    return feature_family(feature)


def _parse_transition(transition: str) -> Tuple[str, str]:
    return parse_transition_id(transition)


def _default_encoded(trainer_id: str, factual: str, counterfactual: str) -> float:
    if _family_of(factual) != _family_of(counterfactual):
        return 0.0
    left, right = PAIR_BY_ID[trainer_id]
    pair = {left, right}
    if factual in pair and counterfactual in pair:
        return 0.65 if counterfactual == right else -0.60
    return 0.04


def build_synthetic_observations(*, full_pool: bool = True) -> Tuple[Tuple[str, str, str, float], ...]:
    rows: List[Tuple[str, str, str, float]] = []
    for trainer_id in TRAINER_ORDER:
        left, _right = PAIR_BY_ID[trainer_id]
        trainer_family = feature_family(left)
        for transition in synthetic_transition_order():
            factual, counterfactual = parse_transition_id(transition)
            if not full_pool:
                if feature_family(factual) != trainer_family or feature_family(counterfactual) != trainer_family:
                    continue
            key = (trainer_id, factual, counterfactual)
            encoded = _EXPLICIT_ENCODED.get(key, _default_encoded(trainer_id, factual, counterfactual))
            rows.append((trainer_id, factual, counterfactual, encoded))
    return tuple(rows)


def transition_label_mapping(transition_ids: Sequence[str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for tid in transition_ids:
        factual, counterfactual = _parse_transition(tid)
        src = FEATURE_LABELS.get(factual, factual)
        tgt = FEATURE_LABELS.get(counterfactual, counterfactual)
        mapping[str(tid)] = f"{src}→{tgt}"
    return mapping


def _encoder_row(factual: str, counterfactual: str, encoded: float) -> Dict[str, Any]:
    return {
        "factual_id": factual,
        "counterfactual_id": counterfactual,
        "transition_id": transition_id(factual, counterfactual),
        "encoded": encoded,
        "type": "training",
    }


def build_synthetic_encoder_summary(
    observations: Iterable[Tuple[str, str, str, float]] | None = None,
    *,
    full_pool: bool = True,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    obs = build_synthetic_observations(full_pool=full_pool) if observations is None else observations
    rows_by_trainer: Dict[str, List[Dict[str, Any]]] = {tid: [] for tid in TRAINER_ORDER}
    for trainer_id, factual, counterfactual, encoded in obs:
        rows_by_trainer[str(trainer_id)].append(_encoder_row(factual, counterfactual, encoded))
    return {
        trainer_id: {"encoder_df": pd.DataFrame(rows)}
        for trainer_id, rows in rows_by_trainer.items()
        if rows
    }


def dummy_trainers() -> Dict[str, object]:
    from types import SimpleNamespace

    from gradiend.trainer.core.arguments import TrainingArguments

    trainers: Dict[str, object] = {}
    for trainer_id, pair in PAIR_BY_ID.items():
        trainers[trainer_id] = SimpleNamespace(
            target_classes=list(pair),
            _training_args=TrainingArguments(source=SOURCE_BY_ID[trainer_id]),
        )
    return trainers


def cell_contribution_rows(
    payload: Dict[str, Any],
    *,
    anchor: str,
    column: str,
) -> pd.DataFrame:
    aligned = payload.get("aligned_rows")
    if aligned is None or aligned.empty:
        return pd.DataFrame()
    mask = (
        aligned["anchor_class"].astype(str) == str(anchor)
    ) & (aligned["eval_class"].astype(str) == str(column))
    return aligned.loc[mask].copy()


def preanchor_highlight_sets(
    contrib: pd.DataFrame,
    *,
    same_family_only: bool = False,
) -> Tuple[List[str], List[str]]:
    if contrib.empty:
        return [], []
    work = contrib
    if same_family_only:
        anchor = str(work["anchor_class"].iloc[0])
        family = _family_of(anchor)
        mask = work["transition_id"].astype(str).map(
            lambda tid: all(_family_of(part) == family for part in _parse_transition(tid))
        )
        work = work.loc[mask]
    trainers = sorted(work["trainer_id"].astype(str).unique().tolist())
    transitions = sorted(work["transition_id"].astype(str).unique().tolist())
    return trainers, transitions
