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


def stage_training_dataset(source_dir: Path, stage_dir: Path) -> Path:
    training_csv = source_dir / "training.csv"
    if not training_csv.is_file():
        raise FileNotFoundError(f"Missing generated training data: {training_csv}")
    if not TRAINING_CARD.is_file():
        raise FileNotFoundError(f"Missing dataset card template: {TRAINING_CARD}")

    repo_dir = stage_dir / "en-pronouns"
    _clean_dir(repo_dir)
    shutil.copy2(TRAINING_CARD, repo_dir / "README.md")
    _copy_common_metadata(source_dir, repo_dir)

    df = pd.read_csv(training_csv)
    required = {"masked", "split", "label_class", "label", "feature_class_id"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"training.csv is missing required columns: {missing}")

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
    }
    _write_json(repo_dir / "dataset_summary.json", summary)

    for split_name, split_df in df.groupby("split", sort=False):
        split_df = split_df.reset_index(drop=True)
        split_df.to_parquet(repo_dir / f"{split_name}.parquet", index=False)
    return repo_dir


def stage_neutral_dataset(source_dir: Path, stage_dir: Path) -> Path:
    neutral_csv = source_dir / "neutral.csv"
    if not neutral_csv.is_file():
        raise FileNotFoundError(f"Missing generated neutral data: {neutral_csv}")
    if not NEUTRAL_CARD.is_file():
        raise FileNotFoundError(f"Missing dataset card template: {NEUTRAL_CARD}")

    repo_dir = stage_dir / "en-pronoun-neutral"
    _clean_dir(repo_dir)
    shutil.copy2(NEUTRAL_CARD, repo_dir / "README.md")
    _copy_common_metadata(source_dir, repo_dir)

    df = pd.read_csv(neutral_csv)
    if list(df.columns) != ["text"]:
        raise ValueError(f"neutral.csv must have exactly one column ['text']; got {list(df.columns)}")

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
        "--commit-message",
        default="Upload English pronoun datasets",
        help="Commit message for both dataset uploads.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    stage_root = args.stage_dir.resolve()
    stage_root.mkdir(parents=True, exist_ok=True)

    training_dir = stage_training_dataset(args.source_dir, stage_root)
    neutral_dir = stage_neutral_dataset(args.source_dir, stage_root)

    print(f"Staged training dataset: {training_dir}")
    print(f"Staged neutral dataset:  {neutral_dir}")
    print(f"Training repo id: {args.training_repo_id}")
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
