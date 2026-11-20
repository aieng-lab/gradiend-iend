"""Tests for class_merge_map: merging base classes into higher-level classes."""

import pandas as pd
import pytest

from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from gradiend.trainer.core.unified_data import (
    UNIFIED_ALTERNATIVE_CLASS,
    UNIFIED_FACTUAL_CLASS,
    UNIFIED_TRANSITION,
)


def _four_class_data():
    """Merged DataFrame with 4 base classes (1SG, 1PL, 3SG, 3PL) and explicit alternative columns."""
    return pd.DataFrame([
        {"masked": "[MASK] here", "split": "train", "label_class": "1SG", "label": "I", "alternative_class": "1PL", "alternative": "we"},
        {"masked": "[MASK] there", "split": "train", "label_class": "1PL", "label": "we", "alternative_class": "1SG", "alternative": "I"},
        {"masked": "[MASK] went", "split": "train", "label_class": "3SG", "label": "he", "alternative_class": "3PL", "alternative": "they"},
        {"masked": "[MASK] left", "split": "train", "label_class": "3PL", "label": "they", "alternative_class": "3SG", "alternative": "he"},
    ])


def _four_class_per_class_dict():
    """Per-class dict with 1SG, 1PL, 3SG, 3PL."""
    return {
        "1SG": pd.DataFrame({"masked": ["[MASK] here"], "split": ["train"], "1SG": ["I"]}),
        "1PL": pd.DataFrame({"masked": ["[MASK] there"], "split": ["train"], "1PL": ["we"]}),
        "3SG": pd.DataFrame({"masked": ["[MASK] went"], "split": ["train"], "3SG": ["he"]}),
        "3PL": pd.DataFrame({"masked": ["[MASK] left"], "split": ["train"], "3PL": ["they"]}),
    }


class TestClassMergeMapDataLayer:
    """class_merge_map applies at data layer; combined_data returns merged view."""

    def test_effective_data_has_merged_class_names(self):
        config = TextPredictionConfig(
            data=_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        cd = trainer.combined_data
        assert cd is not None
        merged_factual = set(cd[UNIFIED_FACTUAL_CLASS].unique())
        merged_alt = set(cd[UNIFIED_ALTERNATIVE_CLASS].unique())
        assert merged_factual <= {"singular", "plural"}
        assert merged_alt <= {"singular", "plural"}
        assert "1SG" not in merged_factual and "3SG" not in merged_factual

    def test_target_classes_inferred_when_merge_has_two_keys(self):
        config = TextPredictionConfig(
            data=_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        assert config.target_classes is None
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        assert trainer.target_classes is not None
        assert set(trainer.target_classes) == {"singular", "plural"}
        assert trainer.pair == ("singular", "plural")

    def test_all_classes_from_effective_data(self):
        """With class_merge_map and all_classes=None, raw base classes are preserved in all_classes."""
        config = TextPredictionConfig(
            data=_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert set(trainer.all_classes) == {"1SG", "1PL", "3SG", "3PL"}

    def test_transitions_use_merged_names(self):
        config = TextPredictionConfig(
            data=_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        cd = trainer.combined_data
        transitions = set(cd[UNIFIED_TRANSITION].unique())
        assert "singular→plural" in transitions or "plural→singular" in transitions
        assert "1SG→1PL" not in transitions


class TestClassMergeMapValidation:
    """Validation: duplicate base class, base in multiple merges."""

    def test_duplicate_base_class_raises(self):
        config = TextPredictionConfig(
            data=_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3SG"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        with pytest.raises(ValueError, match="3SG.*multiple merged"):
            trainer._ensure_data()

    def test_explicit_target_classes_with_merge(self):
        config = TextPredictionConfig(
            data=_four_class_data(),
            target_classes=["singular", "plural"],
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        assert trainer.target_classes == ["singular", "plural"]


class TestClassMergeMapNeutral:
    """Unmapped base classes are excluded from effective data (cross-group only)."""

    def test_unmapped_base_class_excluded_from_effective_data(self):
        """Class '2' (you) not in merge map is excluded; only singular<->plural kept."""
        df = pd.DataFrame([
            {"masked": "[MASK] you", "split": "train", "label_class": "2", "label": "you", "alternative_class": "1SG", "alternative": "I"},
            {"masked": "[MASK] here", "split": "train", "label_class": "1SG", "label": "I", "alternative_class": "2", "alternative": "you"},
            {"masked": "[MASK] went", "split": "train", "label_class": "1SG", "label": "I", "alternative_class": "1PL", "alternative": "we"},
        ])
        config = TextPredictionConfig(
            data=df,
            target_classes=["singular", "plural"],
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        cd = trainer.combined_data
        assert "2" not in set(cd[UNIFIED_FACTUAL_CLASS].unique())
        assert "2" not in set(cd[UNIFIED_ALTERNATIVE_CLASS].unique())
        assert "singular" in set(cd[UNIFIED_FACTUAL_CLASS].unique())
        transitions = set(cd[UNIFIED_TRANSITION].unique())
        assert transitions <= {"singular→plural", "plural→singular"}
        assert len(cd) == 1  # only 1SG->1PL row kept


class TestClassMergeMapCreateTrainingData:
    """create_training_data with class_merge_map yields correct labels."""

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from tests.testing_mocks import MockTokenizer
        return MockTokenizer()

    def test_create_training_data_merged_pipeline(self, tokenizer):
        config = TextPredictionConfig(
            data=_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1


class TestClassMergeMapGroupFiltering:
    """class_merge_map and transition groups keep only desired transitions."""

    def test_number_merge_map_keeps_only_singular_plural_transitions(self):
        """For number (singular vs plural), only 1SG/3SG <-> 1PL/3PL transitions, not 1SG->2 or 1SG->3SG."""
        df = pd.DataFrame([
            {"masked": "[MASK] here", "split": "train", "label_class": "1SG", "label": "I", "alternative_class": "1PL", "alternative": "we"},
            {"masked": "[MASK] here", "split": "train", "label_class": "1SG", "label": "I", "alternative_class": "2", "alternative": "you"},
            {"masked": "[MASK] here", "split": "train", "label_class": "1SG", "label": "I", "alternative_class": "3SG", "alternative": "he"},
            {"masked": "[MASK] here", "split": "train", "label_class": "1SG", "label": "I", "alternative_class": "3PL", "alternative": "they"},
        ])
        config = TextPredictionConfig(
            data=df,
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            class_merge_transition_groups=[["1SG", "3SG", "1PL", "3PL"]],
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        cd = trainer.combined_data
        transitions = set(cd[UNIFIED_TRANSITION].unique())
        # Only singular<->plural; 1SG->2 and 1SG->3SG must be dropped
        assert transitions == {"singular→plural"}
        assert len(cd) == 2  # 1SG->1PL, 1SG->3PL (both map to singular→plural)
        assert "singular→singular" not in transitions
        assert "singular→2" not in transitions
        assert "2" not in cd[UNIFIED_FACTUAL_CLASS].values and "2" not in cd[UNIFIED_ALTERNATIVE_CLASS].values

    def test_number_merge_map_both_directions(self):
        """Both singular->plural and plural->singular transitions are kept."""
        df = pd.DataFrame([
            {"masked": "[MASK] a", "split": "train", "label_class": "1SG", "label": "I", "alternative_class": "1PL", "alternative": "we"},
            {"masked": "[MASK] b", "split": "train", "label_class": "1PL", "label": "we", "alternative_class": "1SG", "alternative": "I"},
        ])
        config = TextPredictionConfig(
            data=df,
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            class_merge_transition_groups=[["1SG", "3SG", "1PL", "3PL"]],
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        cd = trainer.combined_data
        transitions = set(cd[UNIFIED_TRANSITION].unique())
        assert "singular→plural" in transitions
        assert "plural→singular" in transitions
        assert len(transitions) == 2

    def test_base_transition_clusters_limit_pairs(self):
        """transition_groups limit base-class transitions within clusters (e.g. (1SG,1PL) and (3SG,3PL))."""
        df = pd.DataFrame([
            {"masked": "[MASK] a", "split": "train", "label_class": "1SG", "label": "I", "alternative_class": "1PL", "alternative": "we"},
            {"masked": "[MASK] b", "split": "train", "label_class": "1SG", "label": "I", "alternative_class": "3PL", "alternative": "they"},
            {"masked": "[MASK] c", "split": "train", "label_class": "3SG", "label": "he", "alternative_class": "3PL", "alternative": "they"},
            {"masked": "[MASK] d", "split": "train", "label_class": "3SG", "label": "he", "alternative_class": "1PL", "alternative": "we"},
        ])
        # Clusters: (1SG,1PL) and (3SG,3PL) – only within-cluster base transitions kept.
        config = TextPredictionConfig(
            data=df,
            class_merge_map={"sg": ["1SG", "3SG"], "pl": ["1PL", "3PL"]},
            class_merge_transition_groups=[["1SG", "1PL"], ["3SG", "3PL"]],
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        cd = trainer.combined_data
        transitions = set(cd[UNIFIED_TRANSITION].unique())
        # After merge: only sg↔pl transitions corresponding to 1SG↔1PL and 3SG↔3PL remain (no cross-cluster like 1SG→3PL).
        assert transitions <= {"sg→pl", "pl→sg"}
        # We had 4 input rows, 2 should be dropped by clusters, leaving 2 merged rows
        assert len(cd) == 2

    def test_partial_merge_sg_vs_3pl(self):
        """Partial merge: SG from 1SG+3SG, 3PL unmapped; target_classes=['SG','3PL'] keeps SG<->3PL."""
        df = pd.DataFrame([
            {"masked": "[MASK] a", "split": "train", "label_class": "1SG", "label": "I", "alternative_class": "3PL", "alternative": "they"},
            {"masked": "[MASK] b", "split": "train", "label_class": "3SG", "label": "he", "alternative_class": "3PL", "alternative": "they"},
            {"masked": "[MASK] c", "split": "train", "label_class": "3PL", "label": "they", "alternative_class": "1SG", "alternative": "I"},
        ])
        config = TextPredictionConfig(
            data=df,
            target_classes=["SG", "3PL"],
            class_merge_map={"SG": ["1SG", "3SG"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        cd = trainer.combined_data
        transitions = set(cd[UNIFIED_TRANSITION].unique())
        assert "SG→3PL" in transitions
        assert "3PL→SG" in transitions
        assert len(cd) == 3
        assert "3PL" in set(cd[UNIFIED_FACTUAL_CLASS].unique()) | set(cd[UNIFIED_ALTERNATIVE_CLASS].unique())


class TestClassMergeMapSuiteIntegration:
    """TrainerSuite must pass class_merge_map to child trainers built from kwargs (no explicit config)."""

    def test_suite_child_trainer_applies_class_merge_map(self):
        from gradiend.trainer.core.arguments import TrainingArguments
        from gradiend.trainer.suite import SymmetricTrainerSuite, SuitePairDefinition

        pair_definitions = [
            SuitePairDefinition(
                target_classes=("singular", "plural"),
                child_id="pronoun_number_singular_plural",
                class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            )
        ]
        suite = SymmetricTrainerSuite(
            TextPredictionTrainer,
            data=_four_class_data(),
            masked_col="masked",
            split_col="split",
            alternative_col="alternative",
            alternative_class_col="alternative_class",
            pair_definitions=pair_definitions,
            model="bert-base-uncased",
            args=TrainingArguments(do_eval=False, output_dir="tmp_test_suite_merge"),
        )
        trainer = suite.get_trainer("pronoun_number_singular_plural")
        assert trainer.config.class_merge_map == {
            "singular": ["1SG", "3SG"],
            "plural": ["1PL", "3PL"],
        }
        trainer._ensure_data()
        cd = trainer.combined_data
        assert set(cd[UNIFIED_FACTUAL_CLASS].unique()) <= {"singular", "plural"}
        transitions = set(cd[UNIFIED_TRANSITION].unique())
        assert "singular\u2192plural" in transitions or "plural\u2192singular" in transitions

    def test_suite_child_create_training_data_with_merged_pair(self):
        from gradiend.trainer.core.arguments import TrainingArguments
        from gradiend.trainer.suite import SymmetricTrainerSuite, SuitePairDefinition
        from tests.testing_mocks import MockTokenizer

        pair_definitions = [
            SuitePairDefinition(
                target_classes=("singular", "plural"),
                child_id="pronoun_number_singular_plural",
                class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            )
        ]
        suite = SymmetricTrainerSuite(
            TextPredictionTrainer,
            data=_four_class_data(),
            masked_col="masked",
            split_col="split",
            alternative_col="alternative",
            alternative_class_col="alternative_class",
            pair_definitions=pair_definitions,
            model="bert-base-uncased",
            args=TrainingArguments(
                do_eval=False,
                output_dir="tmp_test_suite_merge",
                add_neutral_identity_transitions=False,
            ),
        )
        trainer = suite.get_trainer("pronoun_number_singular_plural")
        tokenizer = MockTokenizer()
        training_data = trainer.create_training_data(tokenizer, split="train", batch_size=1)
        assert training_data is not None
        assert len(training_data) >= 1


class TestClassMergeMapDecoderTargets:
    """Decoder eval targets work via effective data (merged class names)."""

    def test_infer_decoder_targets_merged_classes(self):
        config = TextPredictionConfig(
            data=_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        targets, has_overlap = trainer._infer_decoder_eval_targets()
        assert "singular" in targets
        assert "plural" in targets
        assert has_overlap is False
        assert "I" in targets["singular"] or "he" in targets["singular"]
        assert "we" in targets["plural"] or "they" in targets["plural"]


def _factual_only_four_class_data() -> pd.DataFrame:
    """Factual-only table like english_pronoun training.csv (_to_merged output)."""
    return pd.DataFrame([
        {"masked": "[MASK] here", "split": "train", "label_class": "1SG", "label": "I", "feature_class_id": "1SG"},
        {"masked": "[MASK] there", "split": "train", "label_class": "1PL", "label": "we", "feature_class_id": "1PL"},
        {"masked": "[MASK] went", "split": "train", "label_class": "3SG", "label": "he", "feature_class_id": "3SG"},
        {"masked": "[MASK] left", "split": "train", "label_class": "3PL", "label": "they", "feature_class_id": "3PL"},
    ])


class TestClassMergeMapFactualOnlyTable:
    """class_merge_map on factual-only CSV tables (no alternative_* columns)."""

    def test_trainer_ensure_data_merges_factual_only_table(self):
        config = TextPredictionConfig(
            data=_factual_only_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data()
        cd = trainer.combined_data
        assert cd is not None
        assert set(cd[UNIFIED_FACTUAL_CLASS].unique()) <= {"singular", "plural"}
        assert set(cd[UNIFIED_ALTERNATIVE_CLASS].unique()) <= {"singular", "plural"}

    def test_collect_unified_test_rows_accepts_merged_pronoun_trainer(self):
        from gradiend.comparison.cross_encoding import collect_unified_test_rows

        config = TextPredictionConfig(
            data=_factual_only_four_class_data(),
            class_merge_map={"singular": ["1SG", "3SG"], "plural": ["1PL", "3PL"]},
            run_id="pronoun_number_singular_plural",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        rows = collect_unified_test_rows({"pronoun_number_singular_plural": trainer}, split="train")
        assert not rows.empty
        assert "singular" in set(rows[UNIFIED_FACTUAL_CLASS].astype(str))


def _four_dialect_merged_data() -> pd.DataFrame:
    """Merged DataFrame with 4 raw dialect classes including intra-non-US (UK -> CA) transitions."""
    return pd.DataFrame([
        {"masked": "The [MASK] is nice.", "split": "train", "label_class": "US", "label": "color", "alternative_class": "UK", "alternative": "colour"},
        {"masked": "The [MASK] is nice.", "split": "train", "label_class": "UK", "label": "colour", "alternative_class": "US", "alternative": "color"},
        {"masked": "The [MASK] is nice.", "split": "train", "label_class": "CA", "label": "colour", "alternative_class": "US", "alternative": "color"},
        {"masked": "The [MASK] is nice.", "split": "train", "label_class": "AU", "label": "colour", "alternative_class": "US", "alternative": "color"},
        {"masked": "The [MASK] is nice.", "split": "train", "label_class": "UK", "label": "centre", "alternative_class": "CA", "alternative": "center"},
    ])


class TestMergedFullEvalAndCacheGuards:
    """Pure-CPU unit tests for raw all_classes preservation, intra-merged label=0, reload, and cache checks."""

    def test_all_classes_preserved_and_intra_merged_transitions_handled(self):
        from types import SimpleNamespace
        from tests.testing_mocks import MockTokenizer

        tokenizer = MockTokenizer()
        config = TextPredictionConfig(
            data=_four_dialect_merged_data(),
            target_classes=["US", "non_US"],
            all_classes=["US", "UK", "CA", "AU"],
            class_merge_map={"US": ["US"], "non_US": ["UK", "CA", "AU"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model=SimpleNamespace(tokenizer=tokenizer), config=config)
        trainer._ensure_data()
        assert trainer.all_classes == ["US", "UK", "CA", "AU"]

        ds_pair_only = trainer.create_training_data(
            tokenizer, split="train", batch_size=1, include_other_classes=False
        )
        pair_entries = ds_pair_only.data.to_dict("records")
        assert all(
            not (e.get("raw_factual_id") == "UK" and e.get("raw_alternative_id") == "CA")
            for e in pair_entries
        )

        ds_full = trainer.create_training_data(
            tokenizer, split="train", batch_size=1, include_other_classes=True
        )
        full_entries = ds_full.data.to_dict("records")
        uk_to_ca = [
            e
            for e in full_entries
            if e.get("raw_factual_id") == "UK" and e.get("raw_alternative_id") == "CA"
        ]
        assert len(uk_to_ca) == 1
        assert uk_to_ca[0]["label"] == 0
        assert uk_to_ca[0]["factual_id"] == "non_US"
        assert uk_to_ca[0]["alternative_id"] == "non_US"

    def test_ensure_data_reload_resets_clm_alignment_checked(self):
        config = TextPredictionConfig(
            data=_four_dialect_merged_data(),
            target_classes=["US", "non_US"],
            all_classes=["US", "UK", "CA", "AU"],
            class_merge_map={"US": ["US"], "non_US": ["UK", "CA", "AU"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        trainer = TextPredictionTrainer(model="bert-base-uncased", config=config)
        trainer._ensure_data(materialize_all_class_transitions=False)
        trainer._clm_alignment_checked = True

        trainer._ensure_data(materialize_all_class_transitions=True)
        assert trainer._clm_alignment_checked is False
        assert trainer._data_materialized_all_class_transitions is True

    def test_cached_encoder_df_matches_request_merged_and_one_pole(self):
        merged_cfg = TextPredictionConfig(
            data=_four_dialect_merged_data(),
            target_classes=["US", "non_US"],
            all_classes=["US", "UK", "CA", "AU"],
            class_merge_map={"US": ["US"], "non_US": ["UK", "CA", "AU"]},
            alternative_col="alternative",
            alternative_class_col="alternative_class",
        )
        merged_trainer = TextPredictionTrainer(model="bert-base-uncased", config=merged_cfg)

        missing_raw_df = pd.DataFrame([
            {"type": "training", "source_id": "US", "target_id": "non_US"},
            {"type": "training", "source_id": "non_US", "target_id": "US"},
        ])
        assert not merged_trainer._cached_encoder_df_matches_request(
            missing_raw_df, include_other_classes=True
        )

        target_pair_only_raw_df = pd.DataFrame([
            {"type": "training", "source_id": "US", "target_id": "non_US", "raw_source_id": "US", "raw_target_id": "UK"},
            {"type": "training", "source_id": "non_US", "target_id": "US", "raw_source_id": "UK", "raw_target_id": "US"},
        ])
        assert not merged_trainer._cached_encoder_df_matches_request(
            target_pair_only_raw_df, include_other_classes=True
        )

        full_eval_raw_df = pd.DataFrame([
            {"type": "training", "source_id": "US", "target_id": "non_US", "raw_source_id": "US", "raw_target_id": "UK"},
            {"type": "training", "source_id": "non_US", "target_id": "US", "raw_source_id": "UK", "raw_target_id": "US"},
            {"type": "training", "source_id": "non_US", "target_id": "non_US", "raw_source_id": "UK", "raw_target_id": "CA"},
        ])
        assert merged_trainer._cached_encoder_df_matches_request(
            full_eval_raw_df, include_other_classes=True
        )

        one_pole_cfg = TextPredictionConfig(
            target_classes=["US"],
            all_classes=["US", "UK", "CA"],
            counterfactual_classes="all",
        )
        one_pole_trainer = TextPredictionTrainer(model="bert-base-uncased", config=one_pole_cfg)
        assert not one_pole_trainer._cached_encoder_df_matches_request(
            pd.DataFrame([
                {"type": "training", "source_id": "US", "target_id": "UK"},
                {"type": "training", "source_id": "UK", "target_id": "US"},
            ]),
            include_other_classes=True,
        )
        assert one_pole_trainer._cached_encoder_df_matches_request(
            pd.DataFrame([
                {"type": "training", "source_id": "US", "target_id": "UK"},
                {"type": "training", "source_id": "UK", "target_id": "CA"},
            ]),
            include_other_classes=True,
        )

    def test_resplit_modes_also_split_other_df_rows(self):
        from gradiend.trainer.core.arguments import TrainingArguments

        rows = []
        for i in range(10):
            rows.append({"masked": f"US {i} [MASK].", "split": "train", "label_class": "US", "label": f"us{i}", "alternative_class": "UK", "alternative": f"uk{i}"})
            rows.append({"masked": f"UK {i} [MASK].", "split": "train", "label_class": "UK", "label": f"uk{i}", "alternative_class": "US", "alternative": f"us{i}"})
            rows.append({"masked": f"CA {i} [MASK].", "split": "train", "label_class": "CA", "label": f"ca{i}", "alternative_class": "AU", "alternative": f"au{i}"})
        df = pd.DataFrame(rows)

        for split_col in (None, "heldout"):
            trainer = TextPredictionTrainer(
                model="bert-base-uncased",
                data=df,
                target_classes=["US", "UK"],
                all_classes=["US", "UK", "CA", "AU"],
                alternative_col="alternative",
                alternative_class_col="alternative_class",
                split_col=split_col,
                split_ratios=(0.5, 0.25, 0.25),
                args=TrainingArguments(seed=7, experiment_dir=None),
            )
            trainer._ensure_data(materialize_all_class_transitions=True)
            other_rows = trainer.combined_data[trainer.combined_data[UNIFIED_FACTUAL_CLASS] == "CA"]
            assert set(other_rows["split"]) == {"train", "validation", "test"}

    def test_encode_training_rows_preserves_transition_and_input_type_for_anchor_alignment(self):
        from unittest.mock import MagicMock, PropertyMock, patch
        from gradiend.comparison.anchor_aligned import compute_anchor_aligned_encoding_matrix
        from gradiend.trainer.core.arguments import TrainingArguments
        from gradiend.util.encoding_rows import gradient_entry_to_encoder_row

        trainer = TextPredictionTrainer.__new__(TextPredictionTrainer)
        trainer._training_args = TrainingArguments(source="alternative", target="diff")
        trainer._eval_group = "feature_class_id"

        ab_alt = gradient_entry_to_encoder_row(
            {
                "label": -1.0,
                "factual_id": "A",
                "alternative_id": "B",
                "feature_class_id": "A→B",
                "raw_factual_id": "A_raw",
                "raw_alternative_id": "B_raw",
                "factual_token": "a_tok",
                "alternative_token": "b_tok",
                "text": "text ab",
                "template": "[MASK] ab",
                "data_split": "test",
            },
            encoded=-0.8,
            input_type="alternative",
        )
        ba_alt = gradient_entry_to_encoder_row(
            {
                "label": 1.0,
                "factual_id": "B",
                "alternative_id": "A",
                "feature_class_id": "B→A",
                "raw_factual_id": "B_raw",
                "raw_alternative_id": "A_raw",
                "factual_token": "b_tok",
                "alternative_token": "a_tok",
                "text": "text ba",
                "template": "[MASK] ba",
                "data_split": "test",
            },
            encoded=0.6,
            input_type="alternative",
        )
        comp_row = {
            **ab_alt,
            "component_index": 0,
            "component_id": "layer.0",
            "component_label": "layer.0",
        }
        mock_evaluator = MagicMock()
        mock_evaluator.evaluate_encoder.return_value = {
            "training_rows": [ab_alt, ba_alt],
            "component_training_rows": [comp_row],
        }

        with patch.object(TextPredictionTrainer, "evaluator", new_callable=PropertyMock, return_value=mock_evaluator):
            rows, comp_rows = trainer._encode_training_rows(
                model_with_gradiend=MagicMock(),
                train_eval_data=[],
                source_type="alternative",
                max_size=None,
                encoder_kwargs={"split": "test", "include_other_classes": False},
            )
        assert len(rows) == 2
        assert rows[0]["factual_id"] == "A"
        assert rows[0]["counterfactual_id"] == "B"
        assert rows[0]["transition_id"] == "A→B"
        assert rows[0]["input_type"] == "alternative"
        assert rows[0]["raw_source_id"] == "B_raw"
        assert rows[0]["raw_target_id"] == "A_raw"
        assert rows[0]["raw_factual_id"] == "A_raw"
        assert rows[0]["raw_counterfactual_id"] == "B_raw"

        assert len(comp_rows) == 1
        assert comp_rows[0]["factual_id"] == "A"
        assert comp_rows[0]["counterfactual_id"] == "B"
        assert comp_rows[0]["transition_id"] == "A→B"
        assert comp_rows[0]["input_type"] == "alternative"
        assert comp_rows[0]["component_id"] == "layer.0"

        encoder_df = pd.DataFrame(rows)
        cf_result = compute_anchor_aligned_encoding_matrix(
            pair_by_id={"ab": ("A", "B")},
            encoder_summary={"ab": {"encoder_df": encoder_df}},
            feature_classes=["A", "B"],
            alignment="counterfactual",
            source_by_id={"ab": "alternative"},
        )
        cf_matrix = pd.DataFrame(cf_result["matrix"], index=cf_result["rows"], columns=cf_result["columns"])
        # Under source="alternative", row A->B (encoded=-0.8) has counterfactual_id="B"
        # and row B->A (encoded=+0.6) has counterfactual_id="A".
        assert cf_matrix.loc["A", "B"] == pytest.approx(-0.8)
        assert cf_matrix.loc["A", "A"] == pytest.approx(0.6)

        fac_result = compute_anchor_aligned_encoding_matrix(
            pair_by_id={"ab": ("A", "B")},
            encoder_summary={"ab": {"encoder_df": encoder_df}},
            feature_classes=["A", "B"],
            alignment="factual",
            source_by_id={"ab": "alternative"},
        )
        fac_matrix = pd.DataFrame(fac_result["matrix"], index=fac_result["rows"], columns=fac_result["columns"])
        # Under alignment="factual" with source="alternative", view_sign=-1 flips signs and groups by factual_id.
        assert fac_matrix.loc["A", "A"] == pytest.approx(0.8)
        assert fac_matrix.loc["A", "B"] == pytest.approx(-0.6)
