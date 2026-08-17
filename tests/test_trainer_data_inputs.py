"""Tests for TextPredictionTrainer data inputs: Path, DataFrame, dict, eval_neutral_data path."""


import pandas as pd
import pytest
from unittest.mock import patch

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from gradiend.trainer.core.unified_data import (
    UNIFIED_ALTERNATIVE,
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_TRANSITION,
    resolve_dataframe,
    transition_id,
)
from tests.testing_mocks import MockTokenizer


def _merged_factual_df(rows=None):
    """Minimal merged (factual-only) DataFrame: masked, split, label_class, label."""
    if rows is None:
        rows = [
            {"masked": "[MASK] here", "split": "train", "label_class": "3SG", "label": "he"},
            {"masked": "[MASK] there", "split": "train", "label_class": "3PL", "label": "they"},
        ]
    return pd.DataFrame(rows)


def _merged_with_alternative_df():
    """Merged DataFrame with explicit alternative columns (label_class, label, alternative, alternative_class)."""
    return pd.DataFrame([
        {"masked": "[MASK] here", "split": "train", "label_class": "3SG", "label": "he", "alternative_class": "3PL", "alternative": "they"},
        {"masked": "[MASK] there", "split": "train", "label_class": "3PL", "label": "they", "alternative_class": "3SG", "alternative": "he"},
    ])


def _per_class_dict():
    """Per-class dict compatible with trainer."""
    return {
        "3SG": pd.DataFrame({
            "masked": ["[MASK] here"],
            "split": ["train"],
            "3SG": ["he"],
        }),
        "3PL": pd.DataFrame({
            "masked": ["[MASK] there"],
            "split": ["train"],
            "3PL": ["they"],
        }),
    }


def _per_class_dict_three_classes():
    """Per-class dict with three classes (3SG, 3PL, Other) for identity-for-other tests."""
    return {
        "3SG": pd.DataFrame({
            "masked": ["[MASK] here"],
            "split": ["train"],
            "3SG": ["he"],
        }),
        "3PL": pd.DataFrame({
            "masked": ["[MASK] there"],
            "split": ["train"],
            "3PL": ["they"],
        }),
        "Other": pd.DataFrame({
            "masked": ["[MASK] person"],
            "split": ["train"],
            "Other": ["they"],
        }),
    }


def _per_class_dict_four_classes():
    return {
        "good": pd.DataFrame({"masked": ["[MASK] one"], "split": ["test"], "good": ["good"]}),
        "bad": pd.DataFrame({"masked": ["[MASK] two"], "split": ["test"], "bad": ["bad"]}),
        "happy": pd.DataFrame({"masked": ["[MASK] three"], "split": ["test"], "happy": ["happy"]}),
        "sad": pd.DataFrame({"masked": ["[MASK] four"], "split": ["test"], "sad": ["sad"]}),
    }


class _DummyPredictionTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 103


class TestTrainerDataAsPath:
    """data=Path(...) or data=str path loads CSV/Parquet and builds unified data."""

    def test_data_as_path_loads_csv(self, tmp_path):
        path = tmp_path / "training.csv"
        _merged_factual_df().to_csv(path, index=False)
        config = TextPredictionConfig(
            data=path,
            target_classes=["3SG", "3PL"],
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        assert UNIFIED_FACTUAL_CLASS in trainer._combined_data.columns
        assert UNIFIED_ALTERNATIVE_CLASS in trainer._combined_data.columns
        assert UNIFIED_FACTUAL in trainer._combined_data.columns
        assert UNIFIED_ALTERNATIVE in trainer._combined_data.columns
        assert len(trainer._combined_data) == 2

    def test_data_as_str_path_loads_csv(self, tmp_path):
        path = str(tmp_path / "training.csv")
        _merged_factual_df().to_csv(path, index=False)
        config = TextPredictionConfig(data=path, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        assert len(trainer._combined_data) >= 1

    def test_data_as_path_reports_configured_missing_columns_early(self, tmp_path):
        path = tmp_path / "training.csv"
        pd.DataFrame(
            {
                "masked": ["[MASK] here"],
                "split": ["train"],
                "label": ["M"],
                "name": ["Alex"],
            }
        ).to_csv(path, index=False)
        config = TextPredictionConfig(
            data=path,
            target_classes=["M", "F"],
            masked_col="training_masked",
            label_col="target",
            label_class_col="label",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)

        with pytest.raises(ValueError) as exc:
            trainer._ensure_data()

        message = str(exc.value)
        assert "training data" in message
        assert "required column(s) ['training_masked', 'target']" in message
        assert "Available columns: ['masked', 'split', 'label', 'name']" in message
        assert "'masked_col': 'training_masked'" in message
        assert "'label_col': 'target'" in message

    def test_data_as_directory_loads_training_csv(self, tmp_path):
        _merged_factual_df().to_csv(tmp_path / "training.csv", index=False)
        config = TextPredictionConfig(data=tmp_path, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        assert len(trainer._combined_data) == 2

    def test_missing_data_path_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "training.csv"
        config = TextPredictionConfig(data=missing, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        with pytest.raises(FileNotFoundError, match="does not exist"):
            trainer._ensure_data()

    def test_missing_path_does_not_reach_merged_to_unified(self, tmp_path):
        """Regression: Path must not be passed to merged_to_unified (no AttributeError on .copy)."""
        missing = tmp_path / "training.csv"
        config = TextPredictionConfig(data=missing, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        with pytest.raises(FileNotFoundError) as exc_info:
            trainer._ensure_data()
        assert "copy" not in str(exc_info.value)
        assert "PosixPath" not in str(exc_info.value)
        assert "WindowsPath" not in str(exc_info.value)

    def test_empty_directory_raises_file_not_found(self, tmp_path):
        config = TextPredictionConfig(data=tmp_path, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        with pytest.raises(FileNotFoundError, match=r"training\.csv or training\.parquet"):
            trainer._ensure_data()

    def test_data_as_directory_parquet(self, tmp_path):
        pytest.importorskip("pyarrow")
        _merged_factual_df().to_parquet(tmp_path / "training.parquet", index=False)
        config = TextPredictionConfig(data=tmp_path, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        assert len(trainer._combined_data) == 2

    def test_invalid_data_type_raises_type_error(self):
        config = TextPredictionConfig(data=[1, 2, 3], target_classes=["3SG", "3PL"])  # type: ignore[arg-type]
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        with pytest.raises(TypeError, match="config.data"):
            trainer._ensure_data()


class TestTrainerHfDatasetInput:
    def test_hf_dataset_reports_configured_missing_columns_early(self):
        raw = pd.DataFrame(
            {
                "masked": ["[MASK] here"],
                "split": ["train"],
                "label": ["M"],
                "name": ["Alex"],
            }
        )
        config = TextPredictionConfig(
            hf_dataset="org/dataset",
            target_classes=["M", "F"],
            masked_col="training_masked",
            label_col="target",
            label_class_col="label",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)

        with patch.object(TextPredictionTrainer, "_load_hf_dataset", return_value=raw):
            with pytest.raises(ValueError) as exc:
                trainer._ensure_data()

        message = str(exc.value)
        assert "HF dataset 'org/dataset'" in message
        assert "required column(s) ['training_masked', 'target']" in message
        assert "Available columns: ['masked', 'split', 'label', 'name']" in message
        assert "'masked_col': 'training_masked'" in message
        assert "'label_col': 'target'" in message


class TestTrainerDataAsDataFrame:
    """data=DataFrame (merged) builds unified data."""

    def test_data_as_dataframe_merged(self):
        df = _merged_factual_df()
        config = TextPredictionConfig(data=df, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        assert UNIFIED_FACTUAL in trainer._combined_data.columns
        assert UNIFIED_ALTERNATIVE in trainer._combined_data.columns

    def test_training_args_custom_mask_placeholder_reaches_training_dataset(self):
        df = pd.DataFrame(
            [
                {"masked": "[PRONOUN] is here", "split": "train", "label_class": "3SG", "label": "he"},
                {"masked": "[PRONOUN] are here", "split": "train", "label_class": "3PL", "label": "they"},
            ]
        )
        config = TextPredictionConfig(data=df, target_classes=["3SG", "3PL"])
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            config=config,
            training_args=TrainingArguments(
                mask_placeholder="[PRONOUN]",
                add_neutral_identity_transitions=False,
            ),
        )

        training_data = trainer.create_training_data(MockTokenizer(), split="train", batch_size=1)
        item = training_data[0]

        assert training_data.mask_placeholder == "[PRONOUN]"
        assert item["input_text"].startswith("[MASK]")
        assert item["text"].startswith("he") or item["text"].startswith("they")

    def test_config_custom_mask_placeholder_reaches_training_dataset(self):
        df = pd.DataFrame(
            [
                {"masked": "[PRONOUN] is here", "split": "train", "label_class": "3SG", "label": "he"},
                {"masked": "[PRONOUN] are here", "split": "train", "label_class": "3PL", "label": "they"},
            ]
        )
        config = TextPredictionConfig(
            data=df,
            target_classes=["3SG", "3PL"],
            mask_placeholder="[PRONOUN]",
        )
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            config=config,
            training_args=TrainingArguments(add_neutral_identity_transitions=False),
        )

        training_data = trainer.create_training_data(MockTokenizer(), split="train", batch_size=1)
        item = training_data[0]

        assert training_data.mask_placeholder == "[PRONOUN]"
        assert item["input_text"].startswith("[MASK]")
        assert item["text"].startswith("he") or item["text"].startswith("they")

    def test_feature_class_ids_follow_target_class_order_not_row_order(self):
        df = pd.DataFrame(
            [
                {
                    "masked": "[MASK] first",
                    "split": "train",
                    "label_class": "negative",
                    "label": "bad",
                    "alternative_class": "positive",
                    "alternative": "good",
                },
                {
                    "masked": "[MASK] second",
                    "split": "train",
                    "label_class": "positive",
                    "label": "good",
                    "alternative_class": "negative",
                    "alternative": "bad",
                },
            ]
        )
        config = TextPredictionConfig(data=df, target_classes=["positive", "negative"])
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            config=config,
            args=TrainingArguments(
                source="alternative",
                target="diff",
                add_neutral_identity_transitions=False,
            ),
        )

        dataset = trainer.create_training_data(_DummyPredictionTokenizer(), split="train")
        by_source = dataset.data.set_index("factual_id")

        assert by_source.loc["positive", "label"] == 1
        assert by_source.loc["positive", "feature_pole"] == "pos"
        assert by_source.loc["positive", "feature_class_id"] == 0  # deprecated mirror
        assert by_source.loc["negative", "label"] == -1
        assert by_source.loc["negative", "feature_pole"] == "neg"
        assert by_source.loc["negative", "feature_class_id"] == 1  # deprecated mirror


class TestTrainerDataAsDict:
    """data=dict (per-class) builds merged then unified."""

    def test_data_as_per_class_dict(self):
        class_dfs = _per_class_dict()
        config = TextPredictionConfig(
            data=class_dfs,
            target_classes=["3SG", "3PL"],
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        assert set(trainer._all_classes) == {"3SG", "3PL"}
        assert UNIFIED_FACTUAL_CLASS in trainer._combined_data.columns

    def test_target_classes_not_in_data_raises_early(self):
        """When target_classes include a class not in the data, raise a clear ValueError before building unified data."""
        class_dfs = _per_class_dict()
        config = TextPredictionConfig(
            data=class_dfs,
            target_classes=["3SG", "NonExistent"],
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        with pytest.raises(ValueError) as exc_info:
            trainer._ensure_data()
        assert "target_classes" in str(exc_info.value)
        assert "NonExistent" in str(exc_info.value)
        assert "not present" in str(exc_info.value).lower() or "not in" in str(exc_info.value).lower()
        assert "3SG" in str(exc_info.value) or "3PL" in str(exc_info.value)

    def test_per_class_dict_without_target_classes_infers_from_data_and_run_id_none(self):
        """With data=dict and no explicit target_classes or run_id, target_classes are inferred and decoder eval targets can be inferred."""
        class_dfs = _per_class_dict()
        config = TextPredictionConfig(
            data=class_dfs,
            run_id=None,
            use_class_names_as_columns=True,
        )
        assert config.target_classes is None
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        assert trainer.run_id is None
        trainer._ensure_data()
        assert trainer._combined_data is not None
        assert trainer.target_classes is not None
        assert set(trainer.target_classes) == {"3SG", "3PL"}
        targets, has_overlap = trainer._infer_decoder_eval_targets()
        assert targets
        assert "3SG" in targets and "3PL" in targets
        assert has_overlap is False
        assert isinstance(targets["3SG"], list) and isinstance(targets["3PL"], list)
        assert len(targets["3SG"]) >= 1 and len(targets["3PL"]) >= 1

    def test_infer_decoder_eval_targets_marks_overlapping_tokens_for_row_wise_fallback(self):
        """Auto-inferred decoder eval targets mark overlap so decoder eval can use row-wise targets."""
        # Both classes use the same factual token "x", which should trigger row-wise fallback.
        class_dfs = {
            "A": pd.DataFrame(
                {
                    "masked": ["[MASK] x"],
                    "split": ["train"],
                    "A": ["x"],
                }
            ),
            "B": pd.DataFrame(
                {
                    "masked": ["[MASK] x"],
                    "split": ["train"],
                    "B": ["x"],
                }
            ),
        }
        config = TextPredictionConfig(
            data=class_dfs,
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        targets, has_overlap = trainer._infer_decoder_eval_targets()
        assert targets == {"A": ["x"], "B": ["x"]}
        assert has_overlap is True

    def test_per_class_dict_with_all_classes_keeps_pair_transitions_by_default(self):
        config = TextPredictionConfig(
            data=_per_class_dict_four_classes(),
            target_classes=["good", "bad"],
            all_classes=["good", "bad", "happy", "sad"],
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)

        trainer._ensure_data()

        transitions = set(trainer._combined_data[UNIFIED_TRANSITION])
        assert transition_id("good", "bad") in transitions
        assert transition_id("happy", "sad") not in transitions

    def test_per_class_dict_with_all_classes_and_include_other_materializes_suite_wide_transitions(self):
        config = TextPredictionConfig(
            data=_per_class_dict_four_classes(),
            target_classes=["good", "bad"],
            all_classes=["good", "bad", "happy", "sad"],
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            config=config,
            training_args=TrainingArguments(
                include_other_classes=True,
                add_neutral_identity_transitions=False,
            ),
        )

        trainer._ensure_data()

        transitions = set(trainer._combined_data[UNIFIED_TRANSITION])
        assert transition_id("good", "bad") in transitions
        assert transition_id("happy", "sad") in transitions

        eval_data = trainer.create_training_data(
            _DummyPredictionTokenizer(),
            split="test",
            include_other_classes=True,
            batch_size=1,
        )
        eval_pairs = set(zip(eval_data.data["factual_id"], eval_data.data["alternative_id"]))
        assert ("good", "bad") in eval_pairs
        assert ("happy", "sad") in eval_pairs

    def test_include_other_upgrades_pair_local_per_class_data_to_suite_wide_transitions(self):
        config = TextPredictionConfig(
            data=_per_class_dict_four_classes(),
            target_classes=["good", "bad"],
            all_classes=["good", "bad", "happy", "sad"],
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)

        trainer._ensure_data()
        assert transition_id("happy", "sad") not in set(trainer._combined_data[UNIFIED_TRANSITION])

        eval_data = trainer.create_training_data(
            _DummyPredictionTokenizer(),
            split="test",
            include_other_classes=True,
            batch_size=1,
        )

        assert transition_id("happy", "sad") in set(trainer._combined_data[UNIFIED_TRANSITION])
        eval_pairs = set(zip(eval_data.data["factual_id"], eval_data.data["alternative_id"]))
        assert ("happy", "sad") in eval_pairs

    def test_pair_local_encoder_cache_is_not_valid_for_include_other_request(self):
        config = TextPredictionConfig(
            data=_per_class_dict_four_classes(),
            target_classes=["good", "bad"],
            all_classes=["good", "bad", "happy", "sad"],
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        pair_local_df = pd.DataFrame(
            [
                {"type": "training", "source_id": "good", "target_id": "bad"},
                {"type": "training", "source_id": "bad", "target_id": "good"},
            ]
        )
        full_eval_df = pd.DataFrame(
            [
                {"type": "training", "source_id": "good", "target_id": "bad"},
                {"type": "training", "source_id": "happy", "target_id": "sad"},
            ]
        )
        trainer._ensure_data()

        assert not trainer._cached_encoder_df_matches_request(
            pair_local_df,
            include_other_classes=True,
        )
        assert trainer._cached_encoder_df_matches_request(
            full_eval_df,
            include_other_classes=True,
        )


class TestEvalNeutralDataAsPath:
    """eval_neutral_data=Path or str path is resolved by resolve_dataframe."""

    def test_eval_neutral_data_path_resolved(self, tmp_path):
        neutral_path = tmp_path / "neutral.csv"
        pd.DataFrame({"text": ["Neutral sentence one.", "Another neutral."]}).to_csv(neutral_path, index=False)
        config = TextPredictionConfig(
            data=_merged_factual_df(),
            target_classes=["3SG", "3PL"],
            eval_neutral_data=neutral_path,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        resolved = resolve_dataframe(config.eval_neutral_data)
        assert resolved is not None and "text" in resolved.columns and len(resolved) == 2


class TestStandardPipelinePerDataFormat:
    """Standard pipeline (ensure_data → create_training_data) works for each data input format."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        """Use mock tokenizer — pipeline tests only need [MASK] tokenization."""
        from tests.testing_mocks import MockTokenizer
        return MockTokenizer()

    def test_standard_pipeline_per_class_dict(self, tokenizer):
        """Per-class dict: trainer → ensure_data → create_training_data yields non-empty dataset."""
        config = TextPredictionConfig(
            data=_per_class_dict(),
            target_classes=["3SG", "3PL"],
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1

    def test_standard_pipeline_merged_dataframe_pair_mode(self, tokenizer):
        """Merged DataFrame (factual-only, pair): pipeline works."""
        config = TextPredictionConfig(
            data=_merged_factual_df(),
            target_classes=["3SG", "3PL"],
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1

    def test_standard_pipeline_merged_dataframe_with_alternative_cols(self, tokenizer):
        """Merged DataFrame with alternative/alternative_class columns: pipeline works."""
        config = TextPredictionConfig(
            data=_merged_with_alternative_df(),
            target_classes=["3SG", "3PL"],
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1

    def test_standard_pipeline_csv_path(self, tokenizer, tmp_path):
        """Data as CSV path (merged format with pair): pipeline works."""
        path = tmp_path / "train.csv"
        _merged_factual_df().to_csv(path, index=False)
        config = TextPredictionConfig(
            data=path,
            target_classes=["3SG", "3PL"],
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1

    def test_standard_pipeline_parquet_path(self, tokenizer, tmp_path):
        """Data as Parquet path (merged format with pair): pipeline works."""
        pytest.importorskip("pyarrow")
        path = tmp_path / "train.parquet"
        _merged_factual_df().to_parquet(path, index=False)
        config = TextPredictionConfig(
            data=path,
            target_classes=["3SG", "3PL"],
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer._combined_data is not None
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1

    def test_standard_pipeline_per_class_dict_no_explicit_target_classes(self, tokenizer):
        """Per-class dict without target_classes (inferred from data): pipeline works."""
        config = TextPredictionConfig(
            data=_per_class_dict(),
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer.target_classes is not None
        assert set(trainer.target_classes) == {"3SG", "3PL"}
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1


class TestAddIdentityForOtherClasses:
    """Identity transitions are added only for non-target classes (all_classes \\ target_classes)."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from tests.testing_mocks import MockTokenizer
        return MockTokenizer()

    def test_add_identity_not_added_when_all_classes_equal_target_classes(self, tokenizer):
        """When all_classes is set to target_classes, no identity rows are added."""
        config = TextPredictionConfig(
            data=_per_class_dict(),
            target_classes=["3SG", "3PL"],
            all_classes=["3SG", "3PL"],
            use_class_names_as_columns=True,
        )
        training_args = TrainingArguments(
            add_identity_for_other_classes=True,
            add_neutral_identity_transitions=False,
        )
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            config=config,
            training_args=training_args,
        )
        trainer._ensure_data()
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        df = training_data.data
        identity_rows = df[df["factual_id"] == df["alternative_id"]]
        assert len(identity_rows) == 0, (
            "Expected no identity rows when all_classes equals target_classes"
        )

    def test_add_identity_only_for_non_target_classes(self, tokenizer):
        """When all_classes has extra classes, identity rows are added only for non-target classes."""
        config = TextPredictionConfig(
            data=_per_class_dict_three_classes(),
            target_classes=["3SG", "3PL"],
            use_class_names_as_columns=True,
        )
        training_args = TrainingArguments(
            add_identity_for_other_classes=True,
            add_neutral_identity_transitions=False,
        )
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            config=config,
            training_args=training_args,
        )
        trainer._ensure_data()
        assert set(trainer.all_classes) == {"3SG", "3PL", "Other"}
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        df = training_data.data
        identity_rows = df[df["factual_id"] == df["alternative_id"]]
        assert len(identity_rows) >= 1, "Expected at least one identity row for non-target class Other"
        non_target = {"Other"}
        for _, row in identity_rows.iterrows():
            assert row["factual_id"] in non_target, (
                f"Identity row should be for non-target class only, got factual_id={row['factual_id']!r}"
            )


class TestAddNeutralIdentityTransitions:
    """Neutral identity transitions add zero-labeled training rows from neutral_data."""

    def test_add_neutral_identity_requires_neutral_data(self):
        config = TextPredictionConfig(
            data=_per_class_dict(),
            target_classes=["3SG", "3PL"],
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            config=config,
            training_args=TrainingArguments(add_neutral_identity_transitions=True),
        )

        with pytest.raises(ValueError, match="neutral_data"):
            trainer.create_training_data(_DummyPredictionTokenizer(), split="train", batch_size=1)

    def test_add_neutral_identity_rows_from_train_split(self):
        neutral = pd.DataFrame(
            [
                {"masked": "quiet [MASK]", "label": "stone", "split": "train"},
                {"masked": "bright [MASK]", "label": "cloud", "split": "test"},
            ]
        )
        config = TextPredictionConfig(
            data=_per_class_dict(),
            target_classes=["3SG", "3PL"],
            neutral_data=neutral,
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            config=config,
            training_args=TrainingArguments(add_neutral_identity_transitions=True),
        )

        training_data = trainer.create_training_data(_DummyPredictionTokenizer(), split="train", batch_size=1)
        df = training_data.data
        neutral_rows = df[df["neutral_variant"] == "neutral_identity"]

        assert len(neutral_rows) == 1
        row = neutral_rows.iloc[0]
        assert row["factual"] == "stone"
        assert row["alternative"] == "stone"
        assert row["factual_id"] == "neutral"
        assert row["alternative_id"] == "neutral"
        assert row["label"] == 0
        assert bool(row["is_identity_transition"]) is True

    def test_add_neutral_identity_rows_from_split_mapping(self):
        neutral = {
            "train": pd.DataFrame([{"masked": "quiet [MASK]", "label": "stone"}]),
            "test": pd.DataFrame([{"masked": "bright [MASK]", "label": "cloud"}]),
        }
        config = TextPredictionConfig(
            data=_per_class_dict(),
            target_classes=["3SG", "3PL"],
            neutral_data=neutral,
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            config=config,
            training_args=TrainingArguments(add_neutral_identity_transitions=True),
        )

        training_data = trainer.create_training_data(_DummyPredictionTokenizer(), split="train", batch_size=1)
        neutral_rows = training_data.data[training_data.data["neutral_variant"] == "neutral_identity"]

        assert neutral_rows["factual"].tolist() == ["stone"]
        assert neutral_rows["split"].tolist() == ["train"]

    def test_add_neutral_identity_requires_requested_split_in_mapping(self):
        neutral = {
            "test": pd.DataFrame([{"masked": "bright [MASK]", "label": "cloud"}]),
        }
        config = TextPredictionConfig(
            data=_per_class_dict(),
            target_classes=["3SG", "3PL"],
            neutral_data=neutral,
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(
            model="bert-base-uncased",
            config=config,
            training_args=TrainingArguments(add_neutral_identity_transitions=True),
        )

        with pytest.raises(ValueError, match="available splits"):
            trainer.create_training_data(_DummyPredictionTokenizer(), split="train", batch_size=1)

    def test_neutral_data_is_eval_fallback_with_requested_split(self):
        neutral = pd.DataFrame(
            [
                {"text": "train neutral", "split": "train"},
                {"text": "test neutral", "split": "test"},
            ]
        )
        config = TextPredictionConfig(
            data=_per_class_dict(),
            target_classes=["3SG", "3PL"],
            neutral_data=neutral,
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)

        resolved = trainer._resolve_eval_neutral_dataframe(split="test")

        assert resolved is not None
        assert resolved["text"].tolist() == ["test neutral"]

    def test_neutral_data_split_mapping_is_eval_fallback(self):
        neutral = {
            "train": pd.DataFrame([{"text": "train neutral"}]),
            "test": pd.DataFrame([{"text": "test neutral"}]),
        }
        config = TextPredictionConfig(
            data=_per_class_dict(),
            target_classes=["3SG", "3PL"],
            neutral_data=neutral,
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)

        resolved = trainer._resolve_eval_neutral_dataframe(split="test")

        assert resolved is not None
        assert resolved["text"].tolist() == ["test neutral"]
        assert resolved["split"].tolist() == ["test"]

    def test_eval_neutral_data_overrides_shared_split_mapping(self):
        config = TextPredictionConfig(
            data=_per_class_dict(),
            target_classes=["3SG", "3PL"],
            neutral_data={"test": pd.DataFrame([{"text": "shared neutral"}])},
            eval_neutral_data={"test": pd.DataFrame([{"text": "eval neutral"}])},
            use_class_names_as_columns=True,
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)

        resolved = trainer._resolve_eval_neutral_dataframe(split="test")

        assert resolved is not None
        assert resolved["text"].tolist() == ["eval neutral"]
