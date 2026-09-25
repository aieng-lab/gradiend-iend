import pandas as pd
import pytest
import torch

from gradiend.evaluator.encoder_metrics import get_encoder_metrics_from_dataframe
from gradiend.model import GradiendModel
from gradiend.trainer.core.callbacks import NormalizationCallback
from gradiend.trainer.core.component_seed import (
    apply_saved_component_best_states,
    build_stitched_component_training_run,
    component_run_from_training_stats,
    finalize_component_best_states,
    load_component_best_states,
    save_component_best_states,
    stitch_gradiend_components,
    stitch_gradiend_component_states,
    summarize_component_seed_runs,
    update_component_best_states,
)
from gradiend.util.encoding_rows import encode_dataset_to_rows


class _TinyModelWithGradiend:
    def __init__(self, gradiend):
        self.gradiend = gradiend


def _dataset():
    return [
        {
            "source": torch.tensor([2.0, 0.0, 0.0, -2.0]),
            "label": 1.0,
            "factual_id": "3SG",
            "alternative_id": "3PL",
            "feature_class_id": "3SG",
        },
        {
            "source": torch.tensor([-2.0, 0.0, 0.0, 2.0]),
            "label": -1.0,
            "factual_id": "3PL",
            "alternative_id": "3SG",
            "feature_class_id": "3PL",
        },
    ]


def test_default_split_emits_no_public_component_rows():
    gradiend = GradiendModel(
        input_dim=4,
        latent_dim=1,
        bias_encoder=False,
        bias_decoder=False,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        gradiend.encoder[0].linear.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, -1.0]]))

    rows, component_rows = encode_dataset_to_rows(
        _TinyModelWithGradiend(gradiend),
        _dataset(),
        return_component_rows=True,
    )

    assert rows
    assert component_rows == []
    assert "component_id" not in rows[0]


def test_explicit_split_emits_component_rows_and_aggregate_metrics():
    gradiend = GradiendModel(
        input_dim=4,
        latent_dim=1,
        bias_encoder=False,
        bias_decoder=False,
        device=torch.device("cpu"),
        component_slices=[
            {"id": "activation:left", "start": 0, "end": 2},
            {"id": "activation:right", "start": 2, "end": 4},
        ],
        component_split_mode="tensors",
    )
    with torch.no_grad():
        gradiend.encoder[0].linear.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, -1.0]]))

    rows, component_rows = encode_dataset_to_rows(
        _TinyModelWithGradiend(gradiend),
        _dataset(),
        return_component_rows=True,
    )
    result = get_encoder_metrics_from_dataframe(
        pd.DataFrame(rows),
        component_df=pd.DataFrame(component_rows),
    )

    assert "component_id" not in rows[0]
    assert {row["component_id"] for row in component_rows} == {"activation:left", "activation:right"}
    assert result["correlation"] > 0.99
    assert result["components"]["summary"]["n_components"] == 2
    assert result["components"]["summary"]["correlation_mean"] > 0.99


def test_component_normalization_flips_only_negative_component():
    gradiend = GradiendModel(
        input_dim=4,
        latent_dim=1,
        bias_encoder=False,
        bias_decoder=False,
        activation_decoder="id",
        device=torch.device("cpu"),
        component_slices=[
            {"id": "left", "start": 0, "end": 2},
            {"id": "right", "start": 2, "end": 4},
        ],
        component_split_mode="tensors",
    )
    with torch.no_grad():
        gradiend.encoder[0].linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
        gradiend.decoder[0].linear.weight.copy_(torch.ones(4, 1))

    eval_result = {
        "components": {
            "summary": {"n_components": 2},
            "metrics_by_component": {
                "left": {
                    "component_index": 0,
                    "correlation": -0.9,
                    "mean_by_class": {-1.0: 0.8, 1.0: -0.8},
                },
                "right": {
                    "component_index": 1,
                    "correlation": 0.9,
                    "mean_by_class": {-1.0: -0.8, 1.0: 0.8},
                },
            },
        }
    }
    training_stats = {
        "components": {},
        "component_summary": {},
        "scores": {},
    }

    NormalizationCallback().on_step_end(
        step=100,
        loss=1.0,
        model=_TinyModelWithGradiend(gradiend),
        config={"convergent_score_threshold": 0.6, "convergent_mean_by_class_threshold": 0.5},
        eval_result=eval_result,
        training_stats=training_stats,
    )

    torch.testing.assert_close(gradiend.encoder[0].linear.weight[:, :2], torch.tensor([[-1.0, -2.0]]))
    torch.testing.assert_close(gradiend.encoder[0].linear.weight[:, 2:], torch.tensor([[3.0, 4.0]]))
    assert eval_result["components"]["metrics_by_component"]["left"]["correlation"] == 0.9
    assert eval_result["component_summary"]["n_converged"] == 2


def test_component_run_uses_best_step_per_component_not_global_checkpoint():
    stats = {
        "best_score_checkpoint": {"global_step": 100},
        "training_stats": {
            "components": {
                "100": {
                    "metrics_by_component": {
                        "left": {
                            "component_index": 0,
                            "component_id": "left",
                            "correlation": 0.9,
                            "mean_by_class": {"-1.0": -0.9, "1.0": 0.9},
                        },
                        "right": {
                            "component_index": 1,
                            "component_id": "right",
                            "correlation": 0.2,
                            "mean_by_class": {"-1.0": -0.9, "1.0": 0.9},
                        },
                    }
                },
                "200": {
                    "metrics_by_component": {
                        "left": {
                            "component_index": 0,
                            "component_id": "left",
                            "correlation": 0.7,
                            "mean_by_class": {"-1.0": -0.7, "1.0": 0.7},
                        },
                        "right": {
                            "component_index": 1,
                            "component_id": "right",
                            "correlation": 0.95,
                            "mean_by_class": {"-1.0": -0.95, "1.0": 0.95},
                        },
                    }
                },
            }
        },
    }

    run = component_run_from_training_stats(
        stats,
        threshold=0.5,
        mean_threshold=0.5,
    )

    assert run["summary"]["n_components"] == 2
    assert run["summary"]["n_converged"] == 2
    assert run["convergence_by_component"]["left"]["converged"] is True
    assert run["convergence_by_component"]["left"]["best_component_global_step"] == 100
    assert run["convergence_by_component"]["right"]["converged"] is True
    assert run["convergence_by_component"]["right"]["best_component_global_step"] == 200
    assert run["summary"]["correlation_mean"] == pytest.approx(0.925)


def test_component_seed_summary_combines_partial_convergent_seeds():
    runs = [
        {
            "seed": 0,
            "output_dir": "seed_0",
            "converged": False,
            "component_convergence": {
                "left": {
                    "component_index": 0,
                    "converged": True,
                    "correlation": 0.91,
                    "min_target_class_abs_mean": 0.9,
                },
                "right": {
                    "component_index": 1,
                    "converged": False,
                    "correlation": 0.2,
                },
            },
        },
        {
            "seed": 1,
            "output_dir": "seed_1",
            "converged": False,
            "component_convergence": {
                "left": {
                    "component_index": 0,
                    "converged": False,
                    "correlation": 0.1,
                },
                "right": {
                    "component_index": 1,
                    "converged": True,
                    "correlation": 0.87,
                    "min_target_class_abs_mean": 0.8,
                },
            },
        },
    ]

    summary = summarize_component_seed_runs(runs, min_convergent_seeds=1)

    assert summary["converged"] is True
    assert summary["min_convergent_runs_per_component"] == 1
    assert summary["component_convergent_count_by_id"] == {"left": 1, "right": 1}
    assert summary["selected_components"]["left"]["seed"] == 0
    assert summary["selected_components"]["right"]["seed"] == 1


def test_build_stitched_component_training_run_uses_selected_source_histories():
    selected = {
        "left": {
            "seed": 0,
            "output_dir": "seed_0",
            "component_index": 0,
            "component_label": "left",
        },
        "right": {
            "seed": 1,
            "output_dir": "seed_1",
            "component_index": 1,
            "component_label": "right",
        },
    }
    source_runs = {
        "left": {
            "training_stats": {
                "components": {
                    0: {
                        "metrics_by_component": {
                            "left": {
                                "component_index": 0,
                                "component_label": "left",
                                "correlation": 0.2,
                                "mean_by_class": {"-1.0": -0.2, "1.0": 0.2},
                            }
                        },
                        "convergence_by_component": {"left": {"converged": False}},
                    },
                    10: {
                        "metrics_by_component": {
                            "left": {
                                "component_index": 0,
                                "component_label": "left",
                                "correlation": 0.9,
                                "mean_by_class": {"-1.0": -0.9, "1.0": 0.9},
                            }
                        },
                        "convergence_by_component": {
                            "left": {
                                "converged": True,
                                "min_target_class_abs_mean": 0.9,
                            }
                        },
                    },
                }
            }
        },
        "right": {
            "training_stats": {
                "components": {
                    0: {
                        "metrics_by_component": {
                            "right": {
                                "component_index": 1,
                                "component_label": "right",
                                "correlation": 0.1,
                                "mean_by_class": {"-1.0": -0.1, "1.0": 0.1},
                            }
                        },
                        "convergence_by_component": {"right": {"converged": False}},
                    },
                    10: {
                        "metrics_by_component": {
                            "right": {
                                "component_index": 1,
                                "component_label": "right",
                                "correlation": 0.7,
                                "mean_by_class": {"-1.0": -0.7, "1.0": 0.7},
                            }
                        },
                        "convergence_by_component": {
                            "right": {
                                "converged": True,
                                "min_target_class_abs_mean": 0.7,
                            }
                        },
                    },
                }
            }
        },
    }

    run = build_stitched_component_training_run(selected, source_runs)

    assert run is not None
    ts = run["training_stats"]
    assert ts["synthetic_component_stitching"] is True
    assert ts["scores"][10] == 0.8
    assert ts["mean_by_class"][10] == {"-1.0": -0.8, "1.0": 0.8}
    assert ts["components"][10]["summary"]["n_converged"] == 2
    assert ts["components"][10]["metrics_by_component"]["left"]["source_seed"] == 0
    assert ts["components"][10]["metrics_by_component"]["right"]["source_seed"] == 1


def test_stitch_gradiend_components_copies_only_selected_component_slices():
    def make_model(fill: float) -> _TinyModelWithGradiend:
        gradiend = GradiendModel(
            input_dim=4,
            latent_dim=1,
            bias_encoder=False,
            bias_decoder=True,
            activation_decoder="id",
            device=torch.device("cpu"),
            component_slices=[
                {"id": "left", "start": 0, "end": 2},
                {"id": "right", "start": 2, "end": 4},
            ],
            component_split_mode="tensors",
        )
        with torch.no_grad():
            gradiend.encoder[0].linear.weight.fill_(fill)
            gradiend.decoder[0].linear.weight.fill_(fill)
            gradiend.decoder[0].linear.bias.fill_(fill)
        return _TinyModelWithGradiend(gradiend)

    target = make_model(0.0)
    source_left = make_model(1.0)
    source_right = make_model(2.0)

    rows = stitch_gradiend_components(
        target,
        {
            "left": source_left,
            "right": source_right,
        },
    )

    assert [row["component_id"] for row in rows] == ["left", "right"]
    torch.testing.assert_close(target.gradiend.encoder[0].linear.weight[:, :2], torch.ones(1, 2))
    torch.testing.assert_close(target.gradiend.encoder[0].linear.weight[:, 2:], torch.full((1, 2), 2.0))
    torch.testing.assert_close(target.gradiend.decoder[0].linear.weight[:2, :], torch.ones(2, 1))
    torch.testing.assert_close(target.gradiend.decoder[0].linear.weight[2:, :], torch.full((2, 1), 2.0))
    torch.testing.assert_close(target.gradiend.decoder[0].linear.bias[:2], torch.ones(2))
    torch.testing.assert_close(target.gradiend.decoder[0].linear.bias[2:], torch.full((2,), 2.0))


def test_component_best_state_roundtrip_and_stitching(tmp_path):
    def make_model(fill: float) -> _TinyModelWithGradiend:
        gradiend = GradiendModel(
            input_dim=4,
            latent_dim=1,
            bias_encoder=False,
            bias_decoder=True,
            activation_decoder="id",
            device=torch.device("cpu"),
            component_slices=[
                {"id": "left", "start": 0, "end": 2},
                {"id": "right", "start": 2, "end": 4},
            ],
            component_split_mode="tensors",
        )
        with torch.no_grad():
            gradiend.encoder[0].linear.weight.fill_(fill)
            gradiend.decoder[0].linear.weight.fill_(fill)
            gradiend.decoder[0].linear.bias.fill_(fill)
        return _TinyModelWithGradiend(gradiend)

    source = make_model(1.0)
    best_states = {}
    eval_result = {
        "components": {
            "metrics_by_component": {
                "left": {"component_index": 0, "correlation": 0.8},
                "right": {"component_index": 1, "correlation": 0.7},
            },
            "convergence_by_component": {
                "left": {"converged": True, "min_target_class_abs_mean": 0.8},
                "right": {"converged": True, "min_target_class_abs_mean": 0.7},
            },
        }
    }

    update_component_best_states(best_states, model=source, eval_result=eval_result, step=10)
    with torch.no_grad():
        source.gradiend.encoder[0].linear.weight[:, :2].fill_(3.0)
        source.gradiend.decoder[0].linear.weight[:2, :].fill_(3.0)
        source.gradiend.decoder[0].linear.bias[:2].fill_(3.0)
    eval_result["components"]["metrics_by_component"]["left"]["correlation"] = 0.9
    update_component_best_states(best_states, model=source, eval_result=eval_result, step=20)

    run_dir = str(tmp_path / "seed_0")
    save_component_best_states(run_dir, best_states)
    finalize_component_best_states(run_dir)
    payload = load_component_best_states(run_dir)

    target = make_model(0.0)
    rows = stitch_gradiend_component_states(
        target,
        {
            "left": payload["components"]["left"],
            "right": payload["components"]["right"],
        },
    )

    assert {row["component_id"] for row in rows} == {"left", "right"}
    torch.testing.assert_close(target.gradiend.encoder[0].linear.weight[:, :2], torch.full((1, 2), 3.0))
    torch.testing.assert_close(target.gradiend.encoder[0].linear.weight[:, 2:], torch.full((1, 2), 1.0))
    torch.testing.assert_close(target.gradiend.decoder[0].linear.weight[:2, :], torch.full((2, 1), 3.0))
    torch.testing.assert_close(target.gradiend.decoder[0].linear.weight[2:, :], torch.full((2, 1), 1.0))

    # Single-seed finalize path: apply local-best slices onto whatever aggregate checkpoint remains.
    again = make_model(0.0)
    applied = apply_saved_component_best_states(again, run_dir)
    assert applied is not None
    assert {row["component_id"] for row in applied} == {"left", "right"}
    torch.testing.assert_close(again.gradiend.encoder[0].linear.weight[:, :2], torch.full((1, 2), 3.0))
    torch.testing.assert_close(again.gradiend.encoder[0].linear.weight[:, 2:], torch.full((1, 2), 1.0))
