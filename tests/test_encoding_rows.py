"""Tests for gradient_entry_to_encoder_row / encode_dataset_to_rows."""

from gradiend.evaluator.encoder import _rows_to_encoder_df
from gradiend.util.encoding_rows import gradient_entry_to_encoder_row


def test_gradient_entry_to_encoder_row_includes_masked_and_target_token():
    entry = {
        "label": 0,
        "factual_id": "neutral",
        "alternative_id": "neutral",
        "text": "The chef he added pepper quickly",
        "template": "The chef he added [MASK] quickly",
        "input_text": "The chef he added [MASK] quickly",
        "factual_token": "pepper",
        "alternative_token": "pepper",
    }
    row = gradient_entry_to_encoder_row(entry, encoded=0.12, input_type="factual")
    assert row["masked"] == "The chef he added [MASK] quickly"
    assert row["template"] == "The chef he added [MASK] quickly"
    assert row["factual_token"] == "pepper"
    assert row["source_token"] == "pepper"
    assert row["text"] == "The chef he added pepper quickly"


def test_gradient_entry_to_encoder_row_factual_and_alternative_ids_and_raw_ids():
    entry = {
        "label": 1.0,
        "factual_id": "US",
        "alternative_id": "UK",
        "raw_factual_id": "US",
        "raw_alternative_id": "UK",
        "factual_token": "color",
        "alternative_token": "colour",
    }

    factual_row = gradient_entry_to_encoder_row(entry, encoded=0.8, input_type="factual")
    assert factual_row["source_id"] == "US"
    assert factual_row["target_id"] == "UK"
    assert factual_row["raw_source_id"] == "US"
    assert factual_row["raw_target_id"] == "UK"
    assert factual_row["raw_factual_id"] == "US"
    assert factual_row["raw_counterfactual_id"] == "UK"

    alt_row = gradient_entry_to_encoder_row(entry, encoded=-0.7, input_type="alternative")
    assert alt_row["source_id"] == "UK"
    assert alt_row["target_id"] == "US"
    assert alt_row["raw_source_id"] == "UK"
    assert alt_row["raw_target_id"] == "US"
    assert alt_row["raw_factual_id"] == "US"
    assert alt_row["raw_counterfactual_id"] == "UK"


def test_gradient_entry_to_encoder_row_diff_omits_raw_source_id():
    entry = {
        "label": 1.0,
        "factual_id": "US",
        "alternative_id": "UK",
        "raw_factual_id": "US",
        "raw_alternative_id": "UK",
        "feature_class_id": "US_UK",
    }
    diff_row = gradient_entry_to_encoder_row(entry, encoded=0.5, input_type="diff")
    assert diff_row["source_id"] == "US_UK"
    assert diff_row["target_id"] == "UK"
    assert "raw_source_id" not in diff_row
    assert diff_row["raw_target_id"] == "UK"


def test_rows_to_encoder_df_preserves_raw_columns_only_when_present():
    unmerged_row = gradient_entry_to_encoder_row(
        {"label": 1.0, "factual_id": "US", "alternative_id": "UK"},
        encoded=0.4,
        input_type="factual",
    )
    unmerged_df = _rows_to_encoder_df([unmerged_row])
    for col in ("raw_source_id", "raw_target_id", "raw_factual_id", "raw_counterfactual_id"):
        assert col not in unmerged_df.columns

    merged_row = gradient_entry_to_encoder_row(
        {
            "label": 1.0,
            "factual_id": "US",
            "alternative_id": "non_US",
            "raw_factual_id": "US",
            "raw_alternative_id": "UK",
        },
        encoded=-0.4,
        input_type="alternative",
    )
    merged_df = _rows_to_encoder_df([merged_row])
    assert merged_df["source_id"].tolist() == ["non_US"]
    assert merged_df["target_id"].tolist() == ["US"]
    assert merged_df["raw_source_id"].tolist() == ["UK"]
    assert merged_df["raw_target_id"].tolist() == ["US"]
    assert merged_df["raw_factual_id"].tolist() == ["US"]
    assert merged_df["raw_counterfactual_id"].tolist() == ["UK"]
