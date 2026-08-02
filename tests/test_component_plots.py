import json
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.stats import load_training_stats
from gradiend.trainer.text.prediction.trainer import TextPredictionConfig, TextPredictionTrainer
from gradiend.visualizer.convergence import plot_training_convergence


def _encoder_df() -> pd.DataFrame:
    return pd.DataFrame({
        "encoded": [0.9, -0.8, 0.7, -0.6],
        "label": [1.0, -1.0, 1.0, -1.0],
        "source_id": ["3SG", "3PL", "3SG", "3PL"],
        "target_id": ["3PL", "3SG", "3PL", "3SG"],
        "feature_class_id": ["3SG", "3PL", "3SG", "3PL"],
        "type": ["training", "training", "training", "training"],
    })


def _component_df() -> pd.DataFrame:
    rows = []
    for component_index, component_id, values in [
        (0, "activation:emb", [0.9, -0.8, 0.8, -0.7]),
        (1, "activation:block", [0.6, -0.5, 0.7, -0.6]),
    ]:
        df = _encoder_df().copy()
        df["encoded"] = values
        df["component_index"] = component_index
        df["component_id"] = component_id
        df["component_label"] = component_id.removeprefix("activation:")
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def test_evaluate_encoder_plot_writes_component_sidecars_by_default(tmp_path):
    pytest.importorskip("matplotlib")
    pytest.importorskip("seaborn")

    config = TextPredictionConfig(
        data=pd.DataFrame({
            "masked": ["[MASK] here"],
            "split": ["train"],
            "label_class": ["3SG"],
            "label": ["he"],
        }),
        target_classes=["3SG", "3PL"],
        img_format="png",
    )
    args = TrainingArguments(experiment_dir=str(tmp_path))
    trainer = TextPredictionTrainer(model="bert-base-uncased", config=config, args=args)
    trainer.get_model = MagicMock(return_value=None)

    with patch("matplotlib.pyplot.show"):
        result = trainer.evaluate_encoder(
            encoder_df={"encoder_df": _encoder_df(), "component_df": _component_df()},
            plot=True,
            plot_kwargs={"show": False, "output": str(tmp_path / "encoder_analysis_split_test.pdf")},
        )

    component_paths = result.get("component_plot_paths") or []
    assert len(component_paths) == 3
    assert (tmp_path / "components" / "encoder_analysis_split_test_components.png").exists()
    assert (tmp_path / "components" / "encoder_analysis_split_test_component_000.png").exists()
    assert (tmp_path / "components" / "encoder_analysis_split_test_component_001.png").exists()


def test_plot_training_convergence_writes_component_sidecars_by_default(tmp_path, caplog):
    pytest.importorskip("matplotlib")

    run_info = {
        "training_stats": {
            "scores": {0: 0.1, 10: 0.8},
            "mean_by_class": {
                0: {"-1.0": -0.1, "1.0": 0.1},
                10: {"-1.0": -0.8, "1.0": 0.8},
            },
            "components": {
                0: {
                    "summary": {"n_components": 2, "n_converged": 0},
                    "metrics_by_component": {
                        "emb": {
                            "component_index": 0,
                            "component_label": "emb",
                            "correlation": 0.2,
                            "mean_by_class": {"-1.0": -0.2, "1.0": 0.2},
                        },
                        "block": {
                            "component_index": 1,
                            "component_label": "block",
                            "correlation": -0.1,
                            "mean_by_class": {"-1.0": 0.1, "1.0": -0.1},
                        },
                    },
                },
                10: {
                    "summary": {"n_components": 2, "n_converged": 1},
                    "metrics_by_component": {
                        "emb": {
                            "component_index": 0,
                            "component_label": "emb",
                            "correlation": 0.9,
                            "mean_by_class": {"-1.0": -0.9, "1.0": 0.9},
                        },
                        "block": {
                            "component_index": 1,
                            "component_label": "block",
                            "correlation": 0.4,
                            "mean_by_class": {"-1.0": -0.4, "1.0": 0.4},
                        },
                    },
                },
            },
            "component_summary": {
                0: {"n_components": 2, "n_converged": 0},
                10: {"n_components": 2, "n_converged": 1},
            },
        },
        "best_score_checkpoint": {"global_step": 10, "correlation": 0.8},
    }

    caplog.set_level("INFO")
    with patch("matplotlib.pyplot.show"):
        path = plot_training_convergence(
            training_stats=run_info,
            output=str(tmp_path / "training_convergence.pdf"),
            show=False,
            img_format="png",
        )

    assert path == str(tmp_path / "training_convergence.png")
    assert (tmp_path / "components" / "training_convergence_components.png").exists()
    assert (tmp_path / "components" / "training_convergence_component_000.png").exists()
    assert (tmp_path / "components" / "training_convergence_component_001.png").exists()
    messages = [record.getMessage() for record in caplog.records]
    assert sum(message.startswith("Saved convergence plot:") for message in messages) == 1
    component_messages = [
        message for message in messages
        if "component convergence" in message
    ]
    assert component_messages == [
        f"Saved 3 component convergence plot(s) under {tmp_path / 'components'}"
    ]


def test_plot_training_convergence_uses_canonical_stitched_component_stats(tmp_path):
    pytest.importorskip("matplotlib")

    run_info = {
        "training_stats": {
            "synthetic_component_stitching": True,
            "scores": {0: 0.15, 10: 0.8},
            "components": {
                0: {
                    "summary": {"n_components": 2, "n_converged": 0},
                    "metrics_by_component": {
                        "left": {
                            "component_index": 0,
                            "component_label": "left",
                            "correlation": 0.2,
                            "mean_by_class": {"-1.0": -0.2, "1.0": 0.2},
                        },
                        "right": {
                            "component_index": 1,
                            "component_label": "right",
                            "correlation": 0.1,
                            "mean_by_class": {"-1.0": -0.1, "1.0": 0.1},
                        },
                    },
                },
                10: {
                    "summary": {"n_components": 2, "n_converged": 2},
                    "metrics_by_component": {
                        "left": {
                            "component_index": 0,
                            "component_label": "left",
                            "correlation": 0.9,
                            "mean_by_class": {"-1.0": -0.9, "1.0": 0.9},
                        },
                        "right": {
                            "component_index": 1,
                            "component_label": "right",
                            "correlation": 0.7,
                            "mean_by_class": {"-1.0": -0.7, "1.0": 0.7},
                        },
                    },
                },
            },
            "component_summary": {
                0: {"n_components": 2, "n_converged": 0},
                10: {"n_components": 2, "n_converged": 2},
            },
        },
        "best_score_checkpoint": {"global_step": 10, "correlation": 0.8},
        "convergence_info": {
            "component_stitching": {
                "applied": True,
                "status": "applied",
                "source_seed_by_component": {"left": 0, "right": 1},
            }
        },
    }

    with patch("matplotlib.pyplot.show"):
        path = plot_training_convergence(
            training_stats=run_info,
            output=str(tmp_path / "training_convergence.png"),
            show=False,
        )

    assert path == str(tmp_path / "training_convergence.png")
    assert (tmp_path / "components" / "training_convergence_components.png").exists()
    assert (tmp_path / "components" / "training_convergence_component_000.png").exists()
    assert (tmp_path / "components" / "training_convergence_component_001.png").exists()


def test_load_training_stats_collapses_legacy_nested_stitched_history(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    selected_component_training = {
        "training_stats": {
            "synthetic_component_stitching": True,
            "scores": {"10": 0.8},
            "components": {
                "10": {
                    "summary": {"n_components": 1, "n_converged": 1},
                    "metrics_by_component": {
                        "left": {
                            "component_index": 0,
                            "component_label": "left",
                            "correlation": 0.8,
                        }
                    },
                }
            },
        },
        "best_score_checkpoint": {"global_step": 10, "correlation": 0.8},
    }
    with open(model_dir / "training.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "training_stats": {"scores": {"10": -1.0}},
                "best_score_checkpoint": {"global_step": 10, "correlation": -1.0},
                "training_args": {},
                "convergence_info": {
                    "component_stitching": {
                        "applied": True,
                        "status": "applied",
                        "selected_component_training": selected_component_training,
                    }
                },
            },
            handle,
        )

    loaded = load_training_stats(str(model_dir))

    assert loaded["training_stats"]["synthetic_component_stitching"] is True
    assert loaded["training_stats"]["scores"]["10"] == 0.8
    assert loaded["best_score_checkpoint"]["correlation"] == 0.8
    stitching = loaded["convergence_info"]["component_stitching"]
    assert "selected_component_training" not in stitching
    assert stitching["selected_component_training_collapsed"] is True
