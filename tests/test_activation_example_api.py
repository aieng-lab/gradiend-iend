import ast
from pathlib import Path

import pytest


EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "gradiend"
    / "examples"
    / "experimental"
    / "train_english_pronouns_activation.py"
)
SWEEP_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "gradiend"
    / "examples"
    / "experimental"
    / "sweep_english_pronouns_activation_interventions.py"
)
GENDER_EXAMPLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "gradiend"
    / "examples"
    / "train_gender_en.py"
)


def test_activation_example_delegates_decoder_grid_selection_to_evaluate_decoder():
    tree = ast.parse(EXAMPLE_PATH.read_text(encoding="utf-8"))
    evaluate_decoder_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "evaluate_decoder"
    ]

    assert len(evaluate_decoder_calls) == 1
    keyword_names = {keyword.arg for keyword in evaluate_decoder_calls[0].keywords}
    assert "target_class" in keyword_names
    assert "feature_factors" not in keyword_names


def test_activation_example_uses_modify_model_for_selected_decoder_result():
    tree = ast.parse(EXAMPLE_PATH.read_text(encoding="utf-8"))
    modify_model_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "modify_model"
    ]

    assert len(modify_model_calls) == 1
    keyword_names = {keyword.arg for keyword in modify_model_calls[0].keywords}
    assert {"decoder_results", "target_class"}.issubset(keyword_names)


def test_activation_intervention_sweep_delegates_decoder_selection():
    tree = ast.parse(SWEEP_EXAMPLE_PATH.read_text(encoding="utf-8"))
    evaluate_decoder_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "evaluate_decoder"
    ]

    assert len(evaluate_decoder_calls) >= 2
    keyword_sets = [{keyword.arg for keyword in call.keywords} for call in evaluate_decoder_calls]
    assert all("target_class" in names for names in keyword_sets)
    assert any({"target_class", "feature_factors", "lrs", "output_path", "plot_kwargs"}.issubset(names) for names in keyword_sets)


def test_activation_intervention_sweep_compares_selector_strategies():
    source = SWEEP_EXAMPLE_PATH.read_text(encoding="utf-8")

    assert "application_mode" in source
    assert "steering" in source
    assert "encoder-gated" in source
    assert "encoder_direction" in source
    assert "encoder_range" in source
    assert "encoder_abs" in source
    assert "prediction" in source
    assert '"all"' in source
    assert "actiend_intervention_sweep.csv" in source
    assert "actiend_lms_heatmap.png" in source
    assert "target_probability_delta" in source
    assert "best by P(" in source


def test_activation_intervention_sweep_labels_application_modes_in_plot_rows():
    from gradiend.examples.experimental import sweep_english_pronouns_activation_interventions as sweep

    rows = sweep._strategy_grid(1.0)
    labels = {row["label"] for row in rows}

    assert "steering | prediction slot | always" in labels
    assert "steering | all tokens | always" in labels
    # encoder_range "near target +-0.2" is intentionally omitted from the grid:
    # for |feature_factor|=1 it is identical to "direction gt 0.8" on encodings
    # in [-1, 1] (see the comment in _strategy_grid).
    assert "encoder-gated | prediction slot | near target +-0.2" not in labels
    assert "encoder-gated | all tokens | direction gt 0.8" in labels
    assert all(row["application_mode"] in {"steering", "encoder_gated"} for row in rows)
    assert {row["token_selector"] for row in rows} == {"prediction", "all"}
    # Steering before gated; within each mode, all tokens before prediction slot,
    # with all gates for a scope grouped together.
    assert [row["label"] for row in rows[:6]] == [
        "steering | all tokens | always",
        "steering | prediction slot | always",
        "encoder-gated | all tokens | direction gt 0.5",
        "encoder-gated | all tokens | direction gt 0.8",
        "encoder-gated | all tokens | abs gt 0.8",
        "encoder-gated | prediction slot | direction gt 0.5",
    ]
    assert rows[6]["label"] == "encoder-gated | prediction slot | direction gt 0.8"


def test_activation_intervention_heatmap_nests_single_site_under_parent():
    from gradiend.examples.experimental import sweep_english_pronouns_activation_interventions as sweep

    rows = [
        {
            "kind": "intervention",
            "selector_label": "encoder-gated | all tokens | direction gt 0.5",
            "application_mode": "encoder_gated",
            "token_scope": "all_tokens",
            "gate": "encoder_direction",
            "threshold": 0.5,
            "site_scope": "trained_sites",
            "activation_modules": None,
            "learning_rate": 1.0,
            "target_probability_delta": 0.0,
        },
        {
            "kind": "intervention",
            "selector_label": "steering | prediction slot | always | layer.1",
            "application_mode": "steering",
            "token_scope": "prediction",
            "gate": "always",
            "site_scope": "single:layer.1",
            "activation_modules": "layer.1",
            "learning_rate": 1.0,
            "target_probability_delta": 0.1,
        },
        {
            "kind": "intervention",
            "selector_label": "steering | all tokens | always",
            "application_mode": "steering",
            "token_scope": "all_tokens",
            "gate": "always",
            "site_scope": "trained_sites",
            "activation_modules": None,
            "learning_rate": 1.0,
            "target_probability_delta": 0.2,
        },
        {
            "kind": "intervention",
            "selector_label": "steering | prediction slot | always",
            "application_mode": "steering",
            "token_scope": "prediction",
            "gate": "always",
            "site_scope": "trained_sites",
            "activation_modules": None,
            "learning_rate": 1.0,
            "target_probability_delta": 0.3,
        },
        {
            "kind": "intervention",
            "selector_label": "steering | prediction slot | always | layer.0",
            "application_mode": "steering",
            "token_scope": "prediction",
            "gate": "always",
            "site_scope": "single:layer.0",
            "activation_modules": "layer.0",
            "learning_rate": 1.0,
            "target_probability_delta": 0.05,
        },
    ]

    ordered = sorted(rows, key=sweep._configuration_sort_key)
    assert [row["selector_label"] for row in ordered] == [
        "steering | all tokens | always",
        "steering | prediction slot | always",
        "steering | prediction slot | always | layer.0",
        "steering | prediction slot | always | layer.1",
        "encoder-gated | all tokens | direction gt 0.5",
    ]

def test_activation_intervention_sweep_uses_one_lr_grid():
    source = SWEEP_EXAMPLE_PATH.read_text(encoding="utf-8")

    assert "LR_GRID" in source
    assert "LR_MULTIPLIERS" not in source
    assert "DECODER_LRS" not in source
    assert "lr_multiplier" not in source
    # Best-method single-site ablation must reuse the full LR grid, not one reference LR.
    assert "lrs=list(LR_GRID)" in source
    assert 'lrs=[float(site_reference["learning_rate"])]' not in source
    assert "single-site ablation" in source
    assert '_decoder_plot_output_path(site_label, target_class=target_class)' in source
    # No redundant pre-sweep evaluate_decoder; FF comes from model metadata.
    assert 'plot_kwargs={"title": f"{task} | decoder selection | target={target_class}"}' not in source
    assert "selecting decoder direction via evaluate_decoder" not in source
    assert "derive_default_feature_factor" in source
    assert "decoder_eval_export_row_wise_csv" in source
    # CSV export must not force row-wise scoring mode.
    assert 'decoder_eval_targets="label"' not in source
    assert 'decoder_eval_targets = "label"' not in source
    assert "DECODER_MAX_SIZE" not in source
    assert "target_classes" in source
    assert "target_rank" in source
    assert "target_class in target_classes" in source


def test_activation_examples_enable_neutral_identity_training():
    train_source = EXAMPLE_PATH.read_text(encoding="utf-8")
    sweep_source = SWEEP_EXAMPLE_PATH.read_text(encoding="utf-8")
    gender_source = GENDER_EXAMPLE_PATH.read_text(encoding="utf-8")

    # Both examples load neutral data via load_english_pronoun_neutral_data()
    # into a `neutral_data` variable (not a raw ensure_english_pronoun_data()
    # path tuple) and pass it straight through as neutral_data= so identity
    # transitions are enabled at train time, not just eval time.
    assert "neutral_data=neutral_data" in train_source
    assert "eval_neutral_data=neutral_data" not in train_source
    assert "add_neutral_identity_transitions=True" in train_source
    assert "neutral_data=neutral_data" in sweep_source
    assert "add_neutral_identity_transitions=True" in sweep_source
    assert "neutral_data=neutral_df" in gender_source

def test_activation_intervention_sweep_reports_selector_coverage():
    source = SWEEP_EXAMPLE_PATH.read_text(encoding="utf-8")

    assert "activation_selector_coverage" in source
    assert "activation_gate" in source
    assert "activation_modules" in source
    assert "token_scope" in source
    assert "gate_label" in source
    assert "single-site ablation" in source
    assert "selector_target_coverage" in source
    assert "selector_other_coverage" in source
    assert "selector_neutral_coverage" in source
    assert "selector_target_scope_coverage" in source
    assert "selector_neutral_scope_coverage" in source
    assert "selector_target_minus_neutral" in source
    assert "selector_scope_target_minus_neutral" in source
    # Coverage / specificity metrics don't depend on LR, so they're merged into
    # one heatmap (_plot_selector_metrics_heatmap) instead of one PNG per
    # metric -- covers the same selector_neutral_coverage /
    # selector_neutral_scope_coverage columns as separate files used to.
    assert "actiend_selector_metrics_heatmap.png" in source
    assert "_plot_selector_metrics_heatmap" in source
    assert "probability_specificity_score" in source
    assert "actiend_probability_specificity_heatmap.png" in source
    assert "actiend_specificity_summary.md" in source
    assert "decoder_by_technique" in source
    assert "output_path" in source
    assert "_evaluate_intervention_row" not in source


def test_activation_intervention_sweep_flattens_probability_panel_cells():
    from gradiend.examples.experimental import sweep_english_pronouns_activation_interventions as sweep

    result = {
        "probs": {"3SG": 0.40, "3PL": 0.30},
        "probs_factual": {"3SG": 0.80, "3PL": 0.70},
        "probs_by_dataset": {
            "3SG": {"3SG": 0.80, "3PL": 0.20},
            "3PL": {"3SG": 0.40, "3PL": 0.60},
        },
        "lms": {"lms": 0.10, "perplexity": 10.0},
    }

    row = sweep._flatten_eval(result, target_class="3SG")

    assert row["other_class"] == "3PL"
    assert row["target_probability"] == 0.40
    assert row["target_factual_probability"] == 0.80
    assert row["target_probability_on_other_dataset"] == 0.40
    assert row["other_probability_on_other_dataset"] == 0.60
    assert row["target_margin_on_other_dataset"] == pytest.approx(-0.20)
    assert row["target_margin_on_target_dataset"] == pytest.approx(0.60)


def test_activation_intervention_sweep_adds_specificity_scores():
    from gradiend.examples.experimental import sweep_english_pronouns_activation_interventions as sweep

    baseline = {
        "target_probability": 0.40,
        "target_factual_probability": 0.80,
        "target_margin_on_other_dataset": -0.20,
        "target_margin_on_target_dataset": 0.60,
        "lms": 0.10,
    }
    row = {
        "target_probability": 0.50,
        "target_factual_probability": 0.83,
        "target_margin_on_other_dataset": 0.05,
        "target_margin_on_target_dataset": 0.64,
        "lms": 0.099,
    }

    sweep._add_baseline_deltas(row, baseline=baseline)

    assert row["target_probability_delta"] == pytest.approx(0.10)
    assert row["target_factual_probability_side_effect_abs"] == pytest.approx(0.03)
    assert row["probability_specificity_score"] == pytest.approx(0.07)
    assert row["target_margin_on_other_dataset_delta"] == pytest.approx(0.25)
    assert row["target_margin_side_effect_abs"] == pytest.approx(0.04)
    assert row["margin_specificity_score"] == pytest.approx(0.21)


def test_activation_intervention_sweep_table_skips_empty_generic_metrics(tmp_path):
    from gradiend.examples.experimental import sweep_english_pronouns_activation_interventions as sweep

    path = tmp_path / "table.md"
    rows = [
        {
            "rank": 1,
            "selector_label": "steering | all tokens | always",
            "application_mode": "steering",
            "token_scope": "all_tokens",
            "gate": "always",
            "site_scope": "trained_sites",
            "learning_rate": 1.0,
            "feature_factor": 1.0,
            "target_probability": 0.42,
            "target_probability_delta": 0.10,
            "target_factual_probability": 0.82,
            "target_factual_probability_delta": 0.01,
            "target_margin_on_other_dataset": -0.10,
            "target_margin_on_other_dataset_delta": 0.20,
            "lms": 0.09,
            "lms_ratio": 0.99,
            "lms_prefilter_pass": True,
        }
    ]

    sweep._write_markdown_table(path, rows)
    text = path.read_text(encoding="utf-8")

    assert "target_margin_on_other_dataset" in text
    assert "target_factual_probability_delta" in text
    assert "feature_score" not in text
    assert "accuracy" not in text
    assert "mean_probability" not in text
    assert "loss" not in text


def test_activation_intervention_sweep_rows_from_decoder_result():
    from gradiend.examples.experimental import sweep_english_pronouns_activation_interventions as sweep

    selector_config = {
        "label": "steering | all tokens | always",
        "token_selector": "all",
        "application_mode": "steering",
        "application_mode_label": "steering",
        "token_scope": "all_tokens",
        "gate": "always",
    }
    decoder_result = {
        "grid": {
            "base": {
                "probs": {"3SG": 0.40},
                "probs_factual": {"3SG": 0.80},
                "probs_by_dataset": {
                    "3SG": {"3SG": 0.80, "3PL": 0.20},
                    "3PL": {"3SG": 0.40, "3PL": 0.60},
                },
                "lms": {"lms": 0.10},
            },
            (1.0, 10.0): {
                "id": {"feature_factor": 1.0, "learning_rate": 10.0},
                "probs": {"3SG": 0.50},
                "probs_factual": {"3SG": 0.82},
                "probs_by_dataset": {
                    "3SG": {"3SG": 0.82, "3PL": 0.18},
                    "3PL": {"3SG": 0.50, "3PL": 0.50},
                },
                "lms": {"lms": 0.099},
            },
        }
    }
    baseline = sweep._baseline_row_from_decoder_result(
        decoder_result,
        target_class="3SG",
        feature_factor=1.0,
    )
    rows = sweep._rows_from_decoder_result(
        decoder_result,
        selector_config=selector_config,
        target_class="3SG",
        baseline_row=baseline,
        coverage_columns={"selector_neutral_coverage": 0.1},
    )

    assert len(rows) == 1
    assert rows[0]["learning_rate"] == 10.0
    assert rows[0]["selector_label"] == "steering | all tokens | always"
    assert rows[0]["target_probability_delta"] == pytest.approx(0.10)
    assert rows[0]["probability_specificity_score"] == pytest.approx(0.08)


def test_activation_intervention_sweep_writes_specificity_summary(tmp_path):
    from gradiend.examples.experimental import sweep_english_pronouns_activation_interventions as sweep

    path = tmp_path / "specificity.md"
    rows = [
        {
            "kind": "intervention",
            "rank": 1,
            "selector_label": "steering | all tokens | always",
            "application_mode": "steering",
            "site_scope": "trained_sites",
            "activation_modules": None,
            "learning_rate": 10.0,
            "target_probability_delta": 0.10,
            "target_factual_probability_delta": 0.02,
            "probability_specificity_score": 0.08,
            "target_margin_on_other_dataset_delta": 0.20,
            "target_margin_on_target_dataset_delta": 0.03,
            "margin_specificity_score": 0.17,
            "lms_ratio": 1.0,
            "lms_prefilter_pass": True,
            "selector_target_coverage": 0.8,
            "selector_other_coverage": 0.2,
            "selector_neutral_coverage": 0.1,
            "selector_target_minus_neutral": 0.7,
            "selector_specificity": 0.9,
        }
    ]

    sweep._write_specificity_summary(path, rows, target_class="3SG")
    text = path.read_text(encoding="utf-8")

    assert "Top Probability-Specific Rows" in text
    assert "Top Margin-Specific Rows" in text
    assert "Selector Coverage Specificity" in text
    assert "probability_specificity_score" in text


def test_activation_intervention_sweep_strategy_kwargs():
    from gradiend.examples.experimental import sweep_english_pronouns_activation_interventions as sweep

    strategy = next(row for row in sweep._strategy_grid(1.0) if row["label"] == "encoder-gated | all tokens | direction gt 0.8")
    kwargs = sweep._strategy_intervention_kwargs(strategy, activation_modules="transformer.h.9")

    assert kwargs["token_selector"] == "all"
    assert kwargs["activation_gate"] == "encoder_direction"
    assert kwargs["threshold"] == 0.8
    assert kwargs["direction"] == 1.0
    assert kwargs["activation_modules"] == "transformer.h.9"
