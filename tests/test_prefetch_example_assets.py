"""Manifest and CLI checks for the example-asset prefetcher."""

import tempfile
from pathlib import Path

import pytest

from scripts import prefetch_example_assets as prefetch


def test_prefetch_manifest_covers_smoke_example_assets():
    datasets = {(asset.repo_id, asset.config, asset.split) for asset in prefetch.EXAMPLE_DATASETS}
    models = {asset.repo_id for asset in prefetch.EXAMPLE_MODELS}

    assert ("aieng-lab/gentypes", None, "train") in datasets
    assert ("cardiffnlp/tweet_eval", "sentiment", "test") in datasets
    assert ("vladinc/nrc", None, "train") in datasets
    assert {
        "bert-base-uncased",
        "bert-base-cased",
        "distilbert-base-cased",
        "bert-base-german-cased",
        "gpt2",
        "dbmdz/german-gpt2",
        "t5-small",
    }.issubset(models)
    assert set(prefetch.SPACY_MODELS) == {"en_core_web_sm", "de_core_news_sm"}


def test_prefetch_cli_can_skip_every_asset_group(monkeypatch):
    monkeypatch.setattr(prefetch, "_configure_environment", lambda **_kwargs: None)
    assert prefetch.main(
        ["--skip-datasets", "--skip-models", "--skip-spacy", "--skip-wikipedia"]
    ) == 0


def test_prefetch_cli_sets_one_explicit_shared_cache(monkeypatch):
    monkeypatch.setattr(prefetch, "_configure_environment", lambda **_kwargs: None)
    with tempfile.TemporaryDirectory() as tmp:
        assert prefetch.main(
            [
                "--cache-dir",
                tmp,
                "--skip-datasets",
                "--skip-models",
                "--skip-spacy",
                "--skip-wikipedia",
            ]
        ) == 0
        root = str(Path(tmp).resolve())
        assert prefetch.os.environ["HF_HOME"] == root
        assert prefetch.os.environ["HF_DATASETS_CACHE"] == str(Path(root) / "datasets")
        assert prefetch.os.environ["HF_HUB_CACHE"] == str(Path(root) / "hub")


def test_wikipedia_snapshot_validation_requires_all_nonempty_shards():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for index in range(prefetch.WIKIPEDIA_SHARD_COUNT):
            (root / f"train-{index:05d}-of-00041.parquet").write_bytes(b"parquet")

        prefetch._validate_wikipedia_snapshot(str(root))

        (root / "train-00040-of-00041.parquet").unlink()
        with pytest.raises(FileNotFoundError, match="40 train shard"):
            prefetch._validate_wikipedia_snapshot(str(root))
