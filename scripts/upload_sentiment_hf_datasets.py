#!/usr/bin/env python3
"""Stage and upload the NRC adjective sentiment datasets to Hugging Face.

Publishes:

- ``aieng-lab/en-sentiment-nrc`` — labeled cloze with **two configs**:
  - ``default``: split column as written by ``generate_data``
  - ``split``: vocabulary-held-out by adjective (60/20/20, seed=0;
    ``apply_vocabulary_held_out_split`` — package-paper training split)
- ``aieng-lab/en-sentiment-nrc-neutral`` — neutrals (single config)

Mirrors ``scripts/upload_english_pronoun_hf_datasets.py``.

Package-paper recipe (``gradiend.examples.train_sentiment``):

- top-10 NRC adjectives per valence (``require_adjectives=True``)
- tweet_eval sentiment texts only
- ``max_size_per_class=500``, ``min_count_per_word=50``, ``balance=strict``
  → **50 rows per adjective** (500/class, 1000 labeled total); tweet_eval
  unique yield for the rarest top-10 ADJs is ~50–60, so 50 is the intentional
  even floor rather than a soft 3k aspiration

**Licensing:** CC BY 3.0 (same as TweetEval sentiment). See
``scripts/README_sentiment_hf_upload.md``.

Examples::

    python scripts/upload_sentiment_hf_datasets.py --dry-run
    python scripts/upload_sentiment_hf_datasets.py --regenerate --dry-run
    python scripts/upload_sentiment_hf_datasets.py --regenerate
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

DEFAULT_SOURCE_DIR = Path("data/sentiment_tweets")
DEFAULT_STAGE_DIR = Path("build/hf_datasets/sentiment_nrc")
DEFAULT_TRAINING_REPO_ID = "aieng-lab/en-sentiment-nrc"
DEFAULT_NEUTRAL_REPO_ID = "aieng-lab/en-sentiment-nrc-neutral"
TRAINING_CARD = Path("hf_datasets/en-sentiment-nrc/README.md")
NEUTRAL_CARD = Path("hf_datasets/en-sentiment-nrc-neutral/README.md")

TRAINING_REQUIRED = {"masked", "label_class", "label"}
EXPECTED_CLASSES = ("positive", "negative")
VOCAB_HELDOUT_RATIOS = (0.6, 0.2, 0.2)
CONFIG_DEFAULT = "default"
CONFIG_SPLIT = "split"


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy_generation_meta(source_dir: Path, target_dir: Path) -> None:
    for name in (
        "generation_config.json",
        "generation_meta.json",
        "lexicon_stats.json",
    ):
        src = source_dir / name
        if src.is_file():
            shutil.copy2(src, target_dir / name)


def apply_paper_split(df: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    """Vocabulary-held-out splits (``train_sentiment``, seed=0, 60/20/20)."""
    from gradiend.examples.train_sentiment import apply_vocabulary_held_out_split

    out = apply_vocabulary_held_out_split(
        df,
        seed=int(seed),
        split_ratios=VOCAB_HELDOUT_RATIOS,
    )
    if "feature_class_id" not in out.columns:
        out = out.copy()
        out["feature_class_id"] = out["label_class"]
    return out


def _validate_training_frame(df: pd.DataFrame) -> dict:
    missing = sorted(TRAINING_REQUIRED.difference(df.columns))
    if missing:
        raise ValueError(f"training frame is missing required columns: {missing}")
    if "split" not in df.columns:
        raise ValueError("training frame has no 'split' column")
    if "feature_class_id" not in df.columns:
        df = df.copy()
        df["feature_class_id"] = df["label_class"]

    classes = sorted(df["label_class"].astype(str).unique().tolist())
    if set(classes) != set(EXPECTED_CLASSES):
        raise ValueError(f"Expected label_class in {EXPECTED_CLASSES}; got {classes}")

    labels_by_class = {
        str(c): sorted(
            df.loc[df["label_class"] == c, "label"].astype(str).str.casefold().unique().tolist()
        )
        for c in classes
    }
    n_labels = sum(len(v) for v in labels_by_class.values())
    if n_labels > 30:
        raise ValueError(
            f"Found {n_labels} distinct labels across classes; package paper uses "
            "top-10 ADJ/valence (~20). Pass --regenerate or fix source CSVs."
        )
    return {"df": df, "labels_by_class": labels_by_class, "n_labels": n_labels}


def _label_to_splits(df: pd.DataFrame) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for lab, g in df.groupby(df["label"].astype(str).str.casefold()):
        out[str(lab)] = sorted(g["split"].astype(str).unique().tolist())
    return out


def _write_config_dir(
    config_dir: Path,
    df: pd.DataFrame,
    *,
    config_name: str,
    split_scheme: str,
    seed: Optional[int],
    labels_by_class: dict,
    n_labels: int,
) -> dict:
    config_dir.mkdir(parents=True, exist_ok=True)
    label_to_splits = _label_to_splits(df)
    if split_scheme == "vocabulary_held_out_by_adjective":
        multi = {k: v for k, v in label_to_splits.items() if len(v) > 1}
        if multi:
            raise ValueError(
                f"Config {config_name!r}: held-out split failed; "
                f"labels in multiple splits: {multi}"
            )

    summary = {
        "config": config_name,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "rows_by_class": {
            str(key): int(value)
            for key, value in df["label_class"].value_counts().sort_index().items()
        },
        "rows_by_split": {
            str(key): int(value)
            for key, value in df["split"].value_counts().sort_index().items()
        },
        "labels_by_class": labels_by_class,
        "label_to_split": {k: v[0] for k, v in label_to_splits.items() if len(v) == 1},
        "label_to_splits_if_multi": {k: v for k, v in label_to_splits.items() if len(v) > 1},
        "n_distinct_labels": int(n_labels),
        "split_scheme": split_scheme,
        "split_ratios": list(VOCAB_HELDOUT_RATIOS)
        if split_scheme == "vocabulary_held_out_by_adjective"
        else None,
        "split_seed": int(seed) if seed is not None else None,
        "recipe": "NRC ADJ top-10/valence + tweet_eval (train_sentiment defaults)",
    }
    _write_json(config_dir / "dataset_summary.json", summary)
    df.to_csv(config_dir / "training.csv", index=False)
    for split_name, split_df in df.groupby("split", sort=False):
        split_df = split_df.reset_index(drop=True)
        split_df.to_parquet(config_dir / f"{split_name}.parquet", index=False)
    return summary


def regenerate_source_csvs(
    source_dir: Path,
    *,
    max_size_per_class: int,
    neutral_max_size: int,
    lexicon_words_per_class: int,
    min_occurrences_per_target: int,
    seed: int,
) -> tuple[Path, Path]:
    """Rebuild training/neutral CSVs via package ``generate_data`` defaults."""
    from gradiend.examples.train_sentiment import generate_data

    source_dir = source_dir.resolve()
    source_dir.mkdir(parents=True, exist_ok=True)
    train_path, neutral_path = generate_data(
        output_dir=source_dir,
        max_size_per_class=max_size_per_class,
        neutral_max_size=neutral_max_size,
        lexicon_words_per_class=lexicon_words_per_class,
        min_occurrences_per_target=min_occurrences_per_target,
        seed=seed,
    )
    df = pd.read_csv(train_path)
    labels_by_class = {
        str(c): sorted(
            df.loc[df["label_class"] == c, "label"].astype(str).str.casefold().unique().tolist()
        )
        for c in sorted(df["label_class"].astype(str).unique())
    }
    _write_json(
        source_dir / "generation_meta.json",
        {
            "generator": "gradiend.examples.train_sentiment.generate_data",
            "require_adjectives": True,
            "lexicon_words_per_class": lexicon_words_per_class,
            "min_occurrences_per_target": min_occurrences_per_target,
            "max_size_per_class": max_size_per_class,
            "neutral_max_size": neutral_max_size,
            "seed": seed,
            "corpus": "cardiffnlp/tweet_eval:sentiment",
            "nrc": "vladinc/nrc",
            "balance": "strict",
            "n_rows": int(len(df)),
            "labels_by_class": labels_by_class,
            "hf_configs": {
                CONFIG_DEFAULT: "split column as written by generate_data",
                CONFIG_SPLIT: (
                    "apply_vocabulary_held_out_split "
                    f"(ratios={VOCAB_HELDOUT_RATIOS}, seed={seed})"
                ),
            },
        },
    )
    return Path(train_path), Path(neutral_path)


def stage_training_dataset(source_dir: Path, stage_dir: Path, *, seed: int = 0) -> Path:
    """Stage one HF dataset folder with configs ``default`` and ``split``."""
    training_csv = source_dir / "training.csv"
    if not training_csv.is_file():
        raise FileNotFoundError(
            f"Missing generated training data: {training_csv}\n"
            "Run with --regenerate or: python -m gradiend.examples.train_sentiment"
        )
    if not TRAINING_CARD.is_file():
        raise FileNotFoundError(f"Missing dataset card template: {TRAINING_CARD}")

    repo_dir = stage_dir / "en-sentiment-nrc"
    _clean_dir(repo_dir)
    shutil.copy2(TRAINING_CARD, repo_dir / "README.md")
    _copy_generation_meta(source_dir, repo_dir)

    raw = pd.read_csv(training_csv)
    validated = _validate_training_frame(raw)
    df_default = validated["df"]
    if "feature_class_id" not in df_default.columns:
        df_default = df_default.copy()
        df_default["feature_class_id"] = df_default["label_class"]

    print(f"Staging config {CONFIG_DEFAULT!r} (generate_data split as-is) …")
    summary_default = _write_config_dir(
        repo_dir / CONFIG_DEFAULT,
        df_default,
        config_name=CONFIG_DEFAULT,
        split_scheme="as_in_generate_data",
        seed=None,
        labels_by_class=validated["labels_by_class"],
        n_labels=validated["n_labels"],
    )

    print(
        f"Staging config {CONFIG_SPLIT!r} "
        f"(vocabulary-held-out, ratios={VOCAB_HELDOUT_RATIOS}, seed={seed}) …"
    )
    df_split = apply_paper_split(raw, seed=seed)
    validated_split = _validate_training_frame(df_split)
    summary_split = _write_config_dir(
        repo_dir / CONFIG_SPLIT,
        validated_split["df"],
        config_name=CONFIG_SPLIT,
        split_scheme="vocabulary_held_out_by_adjective",
        seed=seed,
        labels_by_class=validated_split["labels_by_class"],
        n_labels=validated_split["n_labels"],
    )

    _write_json(
        repo_dir / "dataset_summary.json",
        {
            "configs": [CONFIG_DEFAULT, CONFIG_SPLIT],
            CONFIG_DEFAULT: summary_default,
            CONFIG_SPLIT: summary_split,
        },
    )
    return repo_dir


def stage_neutral_dataset(source_dir: Path, stage_dir: Path) -> Path:
    neutral_csv = source_dir / "neutral.csv"
    if not neutral_csv.is_file():
        raise FileNotFoundError(f"Missing generated neutral data: {neutral_csv}")
    if not NEUTRAL_CARD.is_file():
        raise FileNotFoundError(f"Missing dataset card template: {NEUTRAL_CARD}")

    repo_dir = stage_dir / "en-sentiment-nrc-neutral"
    _clean_dir(repo_dir)
    shutil.copy2(NEUTRAL_CARD, repo_dir / "README.md")
    _copy_generation_meta(source_dir, repo_dir)

    df = pd.read_csv(neutral_csv)
    if "text" not in df.columns:
        if "masked" in df.columns and len(df.columns) == 1:
            df = df.rename(columns={"masked": "text"})
        else:
            raise ValueError(
                f"neutral.csv must include a 'text' column; got {list(df.columns)}"
            )
    df = df[["text"]].copy()

    _write_json(
        repo_dir / "dataset_summary.json",
        {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "unique_text": int(df["text"].nunique()),
        },
    )
    df.to_csv(repo_dir / "neutral.csv", index=False)
    df.to_parquet(repo_dir / "train.parquet", index=False)
    return repo_dir


def upload_dataset_folder(
    *,
    repo_id: str,
    folder: Path,
    private: bool,
    token: Optional[str],
    commit_message: str,
) -> None:
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise ImportError(
            "Install huggingface_hub to upload datasets: pip install huggingface_hub"
        ) from exc

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
    api.upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=str(folder),
        commit_message=commit_message,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    parser.add_argument("--training-repo-id", default=DEFAULT_TRAINING_REPO_ID)
    parser.add_argument("--neutral-repo-id", default=DEFAULT_NEUTRAL_REPO_ID)
    parser.add_argument(
        "--token",
        default=None,
        help="HF token. Defaults to the logged-in huggingface_hub token.",
    )
    parser.add_argument("--private", action="store_true", help="Create repos as private.")
    parser.add_argument("--dry-run", action="store_true", help="Only stage files; do not upload.")
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Rebuild data/sentiment_tweets via train_sentiment.generate_data before staging.",
    )
    parser.add_argument(
        "--max-size-per-class",
        type=int,
        default=500,
        help="Only with --regenerate (package default: 500 → 50 rows/adjective).",
    )
    parser.add_argument(
        "--neutral-max-size",
        type=int,
        default=1000,
        help="Only with --regenerate (package default: 1000).",
    )
    parser.add_argument(
        "--lexicon-words-per-class",
        type=int,
        default=10,
        help="Only with --regenerate (package default: top-10 ADJ/valence).",
    )
    parser.add_argument(
        "--min-occurrences-per-target",
        type=int,
        default=50,
        help="Only with --regenerate (package default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the ``split`` config (vocabulary-held-out).",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload NRC adjective sentiment datasets (default + split configs)",
        help="Commit message for dataset uploads.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    source_dir = args.source_dir
    stage_root = args.stage_dir.resolve()
    stage_root.mkdir(parents=True, exist_ok=True)

    if args.regenerate:
        print(
            f"Regenerating sentiment CSVs in {source_dir} "
            f"(top-{args.lexicon_words_per_class} ADJ/valence, "
            f"max_size_per_class={args.max_size_per_class}) …"
        )
        regenerate_source_csvs(
            source_dir,
            max_size_per_class=args.max_size_per_class,
            neutral_max_size=args.neutral_max_size,
            lexicon_words_per_class=args.lexicon_words_per_class,
            min_occurrences_per_target=args.min_occurrences_per_target,
            seed=args.seed,
        )

    training_dir = stage_training_dataset(source_dir, stage_root, seed=args.seed)
    neutral_dir = stage_neutral_dataset(source_dir, stage_root)

    print(f"Staged training dataset: {training_dir}")
    print(f"  configs: {CONFIG_DEFAULT!r} (as generated), {CONFIG_SPLIT!r} (vocab held-out)")
    print(f"Staged neutral dataset:  {neutral_dir}")
    print(f"Training repo id: {args.training_repo_id}")
    print(f"Neutral repo id:  {args.neutral_repo_id}")

    summary_path = training_dir / "dataset_summary.json"
    if summary_path.is_file():
        print("Training summary:", summary_path.read_text(encoding="utf-8").strip())

    if args.dry_run:
        print("Dry run complete; no upload performed.")
        return 0

    upload_dataset_folder(
        repo_id=args.training_repo_id,
        folder=training_dir,
        private=args.private,
        token=args.token,
        commit_message=args.commit_message,
    )
    upload_dataset_folder(
        repo_id=args.neutral_repo_id,
        folder=neutral_dir,
        private=args.private,
        token=args.token,
        commit_message=args.commit_message,
    )
    print("Upload complete.")
    print(f"  https://huggingface.co/datasets/{args.training_repo_id}")
    print(f"  https://huggingface.co/datasets/{args.neutral_repo_id}")
    print(
        f"  load: load_dataset({args.training_repo_id!r})  "
        f"or load_dataset({args.training_repo_id!r}, {CONFIG_SPLIT!r})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
