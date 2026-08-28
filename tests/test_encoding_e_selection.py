import pandas as pd

from gradiend.evaluator.encoder_metrics import (
    get_encoder_metrics_from_dataframe,
    get_encoding_e,
)
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.callbacks import (
    _current_step_selection_score,
    _selection_metric_from_config,
)
from gradiend.trainer.core.component_seed import summarize_component_seed_runs
from gradiend.trainer.trainer import (
    _selection_eval_source,
    _selection_metric_needs_rivals,
)


def test_encoding_e_uses_auc_rival_and_worst_rival_exclusivity():
    df = pd.DataFrame(
        {
            "encoded": [3.0, 2.5, 0.0, -0.2, -2.0, -1.5, 3.1, 2.9],
            "label": [1, 1, 0, 0, -1, -1, -1, -1],
            "source_id": ["target", "target", "neutral", "neutral", "good", "good", "bad", "bad"],
        }
    )

    metrics = get_encoding_e(df)

    assert metrics["roc_auc_neutral"] == 1.0
    assert metrics["auc_rival"] < 1.0
    assert metrics["class_exclusivity"] < 1.0
    assert metrics["encoding_e"] == min(
        metrics["roc_auc_neutral"],
        metrics["auc_rival"],
        metrics["class_exclusivity"],
    )


def test_selection_metric_is_independent_of_convergence_metric():
    args = TrainingArguments(
        convergent_metric="correlation",
        selection_metric="E",
    )

    assert args.convergent_metric == "correlation"
    assert args.selection_metric == "encoding_e"
    assert _selection_metric_from_config(args.to_dict()) == "encoding_e"
    assert _current_step_selection_score(
        step=10,
        metric="encoding_e",
        training_stats={},
        eval_result={"encoding_e": 0.73},
    ) == 0.73


def test_rival_encoding_is_only_required_by_auc_or_e_selectors():
    assert not _selection_metric_needs_rivals("correlation")
    assert _selection_metric_needs_rivals("encoding_e")
    assert _selection_metric_needs_rivals("min_auc_n_o")
    assert _selection_eval_source(
        one_pole=True,
        training_source="both",
        metric="correlation",
    ) == "factual"
    assert _selection_eval_source(
        one_pole=True,
        training_source="both",
        metric="encoding_e",
    ) == "both"


def test_correlation_metric_profile_skips_auc_rival_computation():
    df = pd.DataFrame(
        {
            "encoded": [2.0, 1.5, -1.0, -1.5, 0.0, 0.1],
            "label": [1, 1, -1, -1, 0, 0],
            "type": ["training"] * 4 + ["neutral_dataset"] * 2,
            "source_id": ["target", "target", "rival", "rival", "neutral", "neutral"],
        }
    )

    metrics = get_encoder_metrics_from_dataframe(df, compute_rival_metrics=False)

    assert metrics["correlation"] > 0.9
    assert metrics["roc_auc_other"] is None
    assert metrics["auc_rival"] is None
    assert metrics["class_exclusivity"] is None
    assert metrics["encoding_e"] is None


def test_component_seed_selection_can_rank_by_encoding_e():
    runs = [
        {
            "seed": 1,
            "output_dir": "seed_1",
            "component_convergence": {
                "L0": {"converged": True, "correlation": 0.99, "global_step": 10}
            },
            "component_metrics": {"L0": {"correlation": 0.99, "encoding_e": 0.60}},
        },
        {
            "seed": 2,
            "output_dir": "seed_2",
            "component_convergence": {
                "L0": {"converged": True, "correlation": 0.80, "global_step": 20}
            },
            "component_metrics": {"L0": {"correlation": 0.80, "encoding_e": 0.92}},
        },
    ]

    summary = summarize_component_seed_runs(
        runs,
        min_convergent_seeds=1,
        selection_metric="encoding_e",
    )

    assert summary["selected_components"]["L0"]["seed"] == 2
    assert summary["selection_metric"] == "encoding_e"
