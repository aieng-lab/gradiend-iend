#!/usr/bin/env python3
"""Stage and upload the generated English pronoun datasets to Hugging Face.

This publishes two dataset repositories because the neutral data has a
different schema from the labeled pronoun data:

- aieng-lab/en-pronouns
- aieng-lab/en-pronoun-neutral

The script converts the generated CSV files in data/english_pronouns to parquet
files in a temporary staging directory, copies the dataset cards from
hf_datasets/, and uploads each folder with huggingface_hub.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

DEFAULT_SOURCE_DIR = Path("data/english_pronouns")
DEFAULT_STAGE_DIR = Path("build/hf_datasets/english_pronouns")
DEFAULT_TRAINING_REPO_ID = "aieng-lab/en-pronouns"
DEFAULT_NEUTRAL_REPO_ID = "aieng-lab/en-pronoun-neutral"
TRAINING_CARD = Path("hf_datasets/en-pronouns/README.md")
NEUTRAL_CARD = Path("hf_datasets/en-pronoun-neutral/README.md")


def validate_training_frame(df: pd.DataFrame) -> None:
    """Fail closed when generated prediction examples are not publication-ready."""
    required = {"masked", "split", "label_class", "label", "feature_class_id"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"training.csv is missing required columns: {missing}")
    if df[list(required)].isna().any().any():
        raise ValueError("training.csv contains missing values in required columns")
    duplicate = df.duplicated(["masked", "label_class", "label"], keep=False)
    if duplicate.any():
        raise ValueError(
            "training.csv contains duplicate prediction examples: "
            f"{int(duplicate.sum())} rows; regenerate with the current gradiend API"
        )
    prompt_split_counts = df.groupby("masked", sort=False)["split"].nunique()
    leaking = prompt_split_counts[prompt_split_counts > 1]
    if not leaking.empty:
        raise ValueError(
            "training.csv has masked prompts shared across data splits: "
            f"{len(leaking)} prompts"
        )
    target_pairs = (
        df["label_class"].astype(str) + "\u241f" + df["label"].astype(str)
    )
    target_counts = target_pairs.groupby(df["masked"].astype(str), sort=False).nunique()
    ambiguous = target_counts[target_counts > 1]
    if not ambiguous.empty:
        raise ValueError(
            "training.csv has masked prompts with conflicting class/label targets: "
            f"{len(ambiguous)} prompts"
        )
    expected_splits = {"train", "validation", "test"}
    actual_splits = set(df["split"].astype(str))
    if actual_splits != expected_splits:
        raise ValueError(
            f"training.csv splits must be {sorted(expected_splits)}; got {sorted(actual_splits)}"
        )


def _clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_common_metadata(source_dir: Path, target_dir: Path) -> None:
    config_path = source_dir / "generation_config.json"
    if config_path.is_file():
        shutil.copy2(config_path, target_dir / "generation_config.json")


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def stage_training_dataset(source_dir: Path, stage_dir: Path, *, include_readme: bool = True) -> Path:
    training_csv = source_dir / "training.csv"
    if not training_csv.is_file():
        raise FileNotFoundError(f"Missing generated training data: {training_csv}")
    if include_readme and not TRAINING_CARD.is_file():
        raise FileNotFoundError(f"Missing dataset card template: {TRAINING_CARD}")

    repo_dir = stage_dir / "en-pronouns"
    _clean_dir(repo_dir)
    if include_readme:
        shutil.copy2(TRAINING_CARD, repo_dir / "README.md")
    _copy_common_metadata(source_dir, repo_dir)

    df = pd.read_csv(training_csv)
    validate_training_frame(df)

    summary = {
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
        "rows_by_class_and_split": {
            str(class_id): {
                str(split): int(value)
                for split, value in group["split"].value_counts().sort_index().items()
            }
            for class_id, group in df.groupby("label_class", sort=True)
        },
        "quality_checks": {
            "duplicate_prediction_examples": 0,
            "masked_prompts_crossing_splits": 0,
            "ambiguous_masked_targets": 0,
        },
    }
    _write_json(repo_dir / "dataset_summary.json", summary)

    for split_name, split_df in df.groupby("split", sort=False):
        split_df = split_df.reset_index(drop=True)
        split_df.to_parquet(repo_dir / f"{split_name}.parquet", index=False)
    return repo_dir


def stage_neutral_dataset(source_dir: Path, stage_dir: Path, *, include_readme: bool = True) -> Path:
    neutral_csv = source_dir / "neutral.csv"
    if not neutral_csv.is_file():
        raise FileNotFoundError(f"Missing generated neutral data: {neutral_csv}")
    if include_readme and not NEUTRAL_CARD.is_file():
        raise FileNotFoundError(f"Missing dataset card template: {NEUTRAL_CARD}")

    repo_dir = stage_dir / "en-pronoun-neutral"
    _clean_dir(repo_dir)
    if include_readme:
        shutil.copy2(NEUTRAL_CARD, repo_dir / "README.md")
    _copy_common_metadata(source_dir, repo_dir)

    df = pd.read_csv(neutral_csv)
    if list(df.columns) != ["text"]:
        raise ValueError(f"neutral.csv must have exactly one column ['text']; got {list(df.columns)}")
    if df["text"].isna().any() or df["text"].astype(str).str.strip().eq("").any():
        raise ValueError("neutral.csv contains missing or empty text")
    duplicate_neutral = df["text"].duplicated(keep=False)
    if duplicate_neutral.any():
        raise ValueError(
            f"neutral.csv contains {int(duplicate_neutral.sum())} duplicate rows; regenerate it"
        )

    _write_json(
        repo_dir / "dataset_summary.json",
        {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "unique_text": int(df["text"].nunique()),
        },
    )
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
        raise ImportError("Install huggingface_hub to upload datasets: pip install huggingface_hub") from exc

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
    parser.add_argument("--token", default=None, help="HF token. Defaults to the logged-in huggingface_hub token.")
    parser.add_argument("--private", action="store_true", help="Create repos as private.")
    parser.add_argument("--dry-run", action="store_true", help="Only stage files; do not upload.")
    parser.add_argument(
        "--skip-readme",
        action="store_true",
        help="Do not stage or upload dataset-card README files; existing Hub cards stay unchanged.",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Stage/upload only a corrected labeled training dataset; leave the neutral repository untouched.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload English pronoun datasets",
        help="Commit message for both dataset uploads.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    stage_root = args.stage_dir.resolve()
    stage_root.mkdir(parents=True, exist_ok=True)

    training_dir = stage_training_dataset(
        args.source_dir, stage_root, include_readme=not args.skip_readme
    )
    neutral_dir = None
    if not args.training_only:
        neutral_dir = stage_neutral_dataset(
            args.source_dir, stage_root, include_readme=not args.skip_readme
        )

    print(f"Staged training dataset: {training_dir}")
    print(f"Staged neutral dataset:  {neutral_dir}")
    print(f"Training repo id: {args.training_repo_id}")
    if neutral_dir is not None:
        print(f"Neutral repo id:  {args.neutral_repo_id}")

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
    if neutral_dir is not None:
        upload_dataset_folder(
            repo_id=args.neutral_repo_id,
            folder=neutral_dir,
            private=args.private,
            token=args.token,
            commit_message=args.commit_message,
        )
    print("Upload complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
