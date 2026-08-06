#!/usr/bin/env python3
"""Preload every external asset used by the configured example smoke suite.

Run this once with network access and name the same shared cache used by the
offline SLURM jobs::

    python scripts/prefetch_example_assets.py --cache-dir /shared/drechsel/hf-cache
    python scripts/prefetch_example_assets.py --cache-dir /shared/drechsel/hf-cache --verify-only

The offline job must likewise export ``HF_HOME=/shared/drechsel/hf-cache``.

Use ``--verify-only`` to prove that the resulting cache works offline. The
script processes every asset and reports all failures together at the end.
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Sequence


@dataclass(frozen=True)
class DatasetAsset:
    repo_id: str
    config: Optional[str] = None
    split: Optional[str] = None
    streaming_rows: Optional[int] = None
    reason: str = ""

    @property
    def label(self) -> str:
        parts = [self.repo_id]
        if self.config:
            parts.append(f"config={self.config}")
        if self.split:
            parts.append(f"split={self.split}")
        if self.streaming_rows:
            parts.append(f"first {self.streaming_rows:,} rows")
        return " | ".join(parts)


@dataclass(frozen=True)
class ModelAsset:
    repo_id: str
    kind: Literal["masked_lm", "causal_lm", "seq2seq"]
    reason: str = ""


# Keep configs explicit: config discovery itself needs the Hub and is therefore
# not a reliable way to prepare an offline cache.
EXAMPLE_DATASETS: tuple[DatasetAsset, ...] = (
    DatasetAsset("aieng-lab/genter", reason="train_gender_en training templates"),
    DatasetAsset("aieng-lab/gentypes", split="train", reason="train_gender_en decoder evaluation"),
    DatasetAsset("aieng-lab/namextend", split="train", reason="train_gender_en decoder evaluation"),
    DatasetAsset("aieng-lab/namexact", split="train", reason="train_gender_en name augmentation"),
    DatasetAsset("aieng-lab/biasneutral", split="train", reason="English/race neutral evaluation"),
    DatasetAsset("aieng-lab/de-gender-case-articles", config="masc_nom", reason="German gender examples"),
    DatasetAsset("aieng-lab/de-gender-case-articles", config="fem_nom", reason="German gender examples"),
    DatasetAsset(
        "aieng-lab/wortschatz-leipzig-de-grammar-neutral",
        split="train",
        reason="German neutral evaluation",
    ),
    DatasetAsset("aieng-lab/gradiend_race_data", config="white", reason="multi-seed stability"),
    DatasetAsset("aieng-lab/gradiend_race_data", config="black", reason="multi-seed stability"),
    DatasetAsset("aieng-lab/en-pronouns", split="train", reason="train_english_pronouns"),
    DatasetAsset("aieng-lab/en-pronouns", split="validation", reason="train_english_pronouns"),
    DatasetAsset("aieng-lab/en-pronouns", split="test", reason="train_english_pronouns"),
    DatasetAsset("aieng-lab/en-pronoun-neutral", split="train", reason="English pronoun neutral evaluation"),
    DatasetAsset(
        "aieng-lab/en-sentiment-nrc",
        config="split",
        split="train",
        reason="train_sentiment vocabulary-held-out",
    ),
    DatasetAsset(
        "aieng-lab/en-sentiment-nrc",
        config="split",
        split="validation",
        reason="train_sentiment vocabulary-held-out",
    ),
    DatasetAsset(
        "aieng-lab/en-sentiment-nrc",
        config="split",
        split="test",
        reason="train_sentiment vocabulary-held-out",
    ),
    DatasetAsset(
        "aieng-lab/en-sentiment-nrc-neutral",
        split="train",
        reason="English sentiment neutral evaluation",
    ),
    # Generation showcase (create_english_sentiment_data) still needs the raw sources:
    DatasetAsset("cardiffnlp/tweet_eval", config="sentiment", split="train", reason="create_english_sentiment_data"),
    DatasetAsset("cardiffnlp/tweet_eval", config="sentiment", split="validation", reason="create_english_sentiment_data"),
    DatasetAsset("cardiffnlp/tweet_eval", config="sentiment", split="test", reason="create_english_sentiment_data"),
    DatasetAsset("vladinc/nrc", split="train", reason="create_english_sentiment_data lexicon"),
    # Wikipedia is handled separately because streaming a few rows does not
    # populate a complete offline cache. _prepare_wikipedia_snapshot()
    # downloads the entire raw dataset repository without preprocessing it.
)


EXAMPLE_MODELS: tuple[ModelAsset, ...] = (
    ModelAsset("bert-base-uncased", "masked_lm", "start_workflow/train_english_pronouns"),
    ModelAsset("bert-base-cased", "masked_lm", "readme/train_sentiment"),
    ModelAsset("distilbert-base-cased", "masked_lm", "train_gender_en/multi-seed stability"),
    ModelAsset("bert-base-german-cased", "masked_lm", "train_gender_de"),
    ModelAsset("gpt2", "causal_lm", "train_gender_en decoder-only mode"),
    ModelAsset("dbmdz/german-gpt2", "causal_lm", "train_gender_de_decoder_only"),
    ModelAsset("t5-small", "seq2seq", "train_seq2seq_encoder_mlm"),
)


SPACY_MODELS: tuple[str, ...] = ("en_core_web_sm", "de_core_news_sm")

WIKIPEDIA_REPO = "wikimedia/wikipedia"
WIKIPEDIA_CONFIG = "20231101.en"
WIKIPEDIA_SHARD_COUNT = 41


def _configure_environment(*, verify_only: bool) -> None:
    if verify_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    else:
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

    from gradiend.util.hf_env import configure_hf_download_env

    configure_hf_download_env()


def _prefetch_dataset(asset: DatasetAsset, *, verify_only: bool) -> None:
    from datasets import load_dataset

    kwargs: dict[str, object] = {}
    if asset.split is not None:
        kwargs["split"] = asset.split
    if asset.streaming_rows is not None:
        kwargs["streaming"] = True

    if asset.config is None:
        dataset = load_dataset(asset.repo_id, **kwargs)
    else:
        dataset = load_dataset(asset.repo_id, asset.config, **kwargs)

    if asset.streaming_rows is not None:
        count = 0
        for _row in dataset:
            count += 1
            if count >= asset.streaming_rows:
                break
        if count < asset.streaming_rows:
            raise RuntimeError(
                f"Only {count:,} rows were available; expected {asset.streaming_rows:,}."
            )
        return

    # Force lazy/iterable metadata to resolve while online. Arrow datasets are
    # already materialized by load_dataset, so reading one row is sufficient.
    if hasattr(dataset, "items"):
        for split_name, split_dataset in dataset.items():
            if len(split_dataset):
                _ = split_dataset[0]
            print(f"      {split_name}: {len(split_dataset):,} rows")
    else:
        if len(dataset):
            _ = dataset[0]
        print(f"      rows: {len(dataset):,}")


def _prefetch_model(asset: ModelAsset, *, verify_only: bool) -> None:
    from transformers import (
        AutoConfig,
        AutoModelForCausalLM,
        AutoModelForMaskedLM,
        AutoModelForSeq2SeqLM,
        AutoTokenizer,
    )

    kwargs = {"local_files_only": verify_only}
    AutoConfig.from_pretrained(asset.repo_id, **kwargs)
    AutoTokenizer.from_pretrained(asset.repo_id, **kwargs)
    model_cls = {
        "masked_lm": AutoModelForMaskedLM,
        "causal_lm": AutoModelForCausalLM,
        "seq2seq": AutoModelForSeq2SeqLM,
    }[asset.kind]
    model = model_cls.from_pretrained(asset.repo_id, **kwargs)
    del model
    gc.collect()


def _prefetch_spacy(model_name: str, *, verify_only: bool) -> None:
    import spacy

    try:
        nlp = spacy.load(model_name)
    except OSError:
        if verify_only:
            raise
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", model_name],
            check=True,
        )
        nlp = spacy.load(model_name)
    del nlp


def _validate_wikipedia_snapshot(snapshot_path: str) -> None:
    parquet_files = sorted(Path(snapshot_path).rglob("train-*.parquet"))
    if len(parquet_files) != WIKIPEDIA_SHARD_COUNT:
        raise FileNotFoundError(
            f"Wikipedia snapshot has {len(parquet_files)} train shard(s); "
            f"expected {WIKIPEDIA_SHARD_COUNT}."
        )
    empty = [str(path) for path in parquet_files if path.stat().st_size <= 0]
    if empty:
        raise RuntimeError(f"Wikipedia snapshot contains empty shard(s): {empty}")


def _prepare_wikipedia_snapshot(*, verify_only: bool) -> None:
    """Cache the complete raw Wikipedia dataset repository, without preprocessing."""
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=WIKIPEDIA_REPO,
        repo_type="dataset",
        allow_patterns=[
            "README.md",
            ".gitattributes",
            "*.json",
            f"{WIKIPEDIA_CONFIG}/**",
            f"data/{WIKIPEDIA_CONFIG}/**",
            f"**/{WIKIPEDIA_CONFIG}/**",
        ],
        local_files_only=verify_only,
    )
    _validate_wikipedia_snapshot(snapshot_path)


def _run_group(title: str, assets: Sequence[object], loader, failures: list[str]) -> None:
    print(f"\n=== {title} ({len(assets)}) ===")
    for index, asset in enumerate(assets, start=1):
        label = getattr(asset, "label", None) or getattr(asset, "repo_id", None) or str(asset)
        reason = getattr(asset, "reason", "")
        print(f"[{index}/{len(assets)}] {label}" + (f" ({reason})" if reason else ""))
        try:
            loader(asset)
        except Exception as exc:  # continue so one run reveals every missing asset
            message = f"{title}: {label}: {type(exc).__name__}: {exc}"
            failures.append(message)
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        else:
            print("  OK")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Require every asset to be available offline; download nothing.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=(
            "Set HF_HOME explicitly for this run (and derive its datasets/hub caches). "
            "Use the identical path in offline jobs."
        ),
    )
    parser.add_argument("--skip-datasets", action="store_true")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-spacy", action="store_true")
    parser.add_argument(
        "--skip-wikipedia",
        "--skip-wikipedia-sample",
        "--skip-generated-data",
        dest="skip_wikipedia",
        action="store_true",
        help="Skip the complete raw Wikipedia dataset snapshot.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if args.cache_dir is not None:
        cache_dir = args.cache_dir.expanduser().resolve()
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_dir)
        os.environ["HF_DATASETS_CACHE"] = str(cache_dir / "datasets")
        os.environ["HF_HUB_CACHE"] = str(cache_dir / "hub")
    _configure_environment(verify_only=args.verify_only)

    print(f"Mode: {'offline verification' if args.verify_only else 'online prefetch'}")
    print(f"HF_HOME={os.environ.get('HF_HOME', '<unset>')}")
    print(f"HF_DATASETS_CACHE={os.environ.get('HF_DATASETS_CACHE', '<unset>')}")
    print(f"HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE', '<unset>')}")

    failures: list[str] = []
    if not args.skip_datasets:
        _run_group(
            "datasets",
            EXAMPLE_DATASETS,
            lambda asset: _prefetch_dataset(asset, verify_only=args.verify_only),
            failures,
        )
    if not args.skip_models:
        _run_group(
            "models",
            EXAMPLE_MODELS,
            lambda asset: _prefetch_model(asset, verify_only=args.verify_only),
            failures,
        )
    if not args.skip_spacy:
        _run_group(
            "spaCy models",
            SPACY_MODELS,
            lambda asset: _prefetch_spacy(asset, verify_only=args.verify_only),
            failures,
        )
    if not args.skip_wikipedia:
        print("\n=== complete dataset snapshots (1) ===")
        print("[1/1] Full raw English Wikipedia dataset (no preprocessing)")
        try:
            _prepare_wikipedia_snapshot(verify_only=args.verify_only)
        except Exception as exc:
            failures.append(f"dataset snapshot: Wikipedia: {type(exc).__name__}: {exc}")
            print(f"  FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        else:
            print("  OK")

    if failures:
        print(f"\nFAILED ASSETS ({len(failures)}):", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print("\nAll configured example assets are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
