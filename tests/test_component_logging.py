import json

from gradiend.model import model_with_gradiend as mwg_module
from gradiend.trainer.core.callbacks import LoggingCallback
from gradiend.trainer.core.training import format_non_convergence_error
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.util.component_logging import (
    format_component_convergence_fragment,
    format_component_seed_summary_fragment,
    format_component_stitching_fragment,
)


def _components(n_components, converged_indices):
    metrics = {}
    convergence = {}
    for index in range(n_components):
        component_id = f"activation:h.{index}"
        metrics[component_id] = {"component_index": index}
        convergence[component_id] = {"converged": index in converged_indices}
    return {
        "summary": {
            "n_components": n_components,
            "n_converged": len(converged_indices),
        },
        "metrics_by_component": metrics,
        "convergence_by_component": convergence,
    }


def test_component_convergence_fragment_names_low_converged_edge():
    fragment = format_component_convergence_fragment(_components(13, {1, 3}))

    assert fragment == "components: 2/13 converged [activation:h.1, activation:h.3]"


def test_component_convergence_fragment_names_low_remaining_edge():
    fragment = format_component_convergence_fragment(_components(13, set(range(11))))

    assert fragment == "components: 11/13 converged, remaining [activation:h.11, activation:h.12]"


def test_component_convergence_fragment_can_force_blockers_for_warning():
    fragment = format_component_convergence_fragment(
        _components(13, {0, 1, 2, 3}),
        show_blockers=True,
    )

    assert fragment == (
        "components: 4/13 converged, "
        "remaining 9 components (first 3: [activation:h.4, activation:h.5, activation:h.6])"
    )


def test_component_convergence_fragment_shows_all_for_tiny_splits():
    fragment = format_component_convergence_fragment(_components(3, {0, 2}))

    assert fragment == (
        "components: 2/3 converged "
        "[activation:h.0: ok; activation:h.1: pending; activation:h.2: ok]"
    )


def test_component_seed_summary_fragment_names_remaining_edge():
    fragment = format_component_seed_summary_fragment({
        "n_components": 13,
        "n_satisfied_components": 11,
        "min_convergent_seeds": 1,
        "component_convergent_count_by_id": {
            **{f"activation:h.{i}": 1 for i in range(11)},
            "activation:h.11": 0,
            "activation:h.12": 0,
        },
        "missing_component_ids": ["activation:h.11", "activation:h.12"],
    })

    assert fragment == (
        "components: 11/13 have required convergent seed(s), "
        "remaining [activation:h.11, activation:h.12]"
    )


def test_component_seed_summary_fragment_can_force_blockers_for_warning():
    fragment = format_component_seed_summary_fragment(
        {
            "n_components": 13,
            "n_satisfied_components": 4,
            "min_convergent_seeds": 1,
            "component_convergent_count_by_id": {
                **{f"activation:h.{i}": 1 for i in range(4)},
                **{f"activation:h.{i}": 0 for i in range(4, 13)},
            },
        },
        show_blockers=True,
    )

    assert fragment == (
        "components: 4/13 have required convergent seed(s), "
        "remaining 9 components (first 3: [activation:h.4, activation:h.5, activation:h.6])"
    )


def test_component_stitching_fragment_reports_skipped_status():
    fragment = format_component_stitching_fragment({
        "applied": False,
        "status": "skipped",
        "reason": "component convergence policy not satisfied",
    })

    assert fragment == "component stitching skipped: component convergence policy not satisfied"


def test_logging_callback_adds_component_ids_near_edges(caplog):
    callback = LoggingCallback(n_loss_report=1)
    caplog.set_level("INFO")

    callback.on_step_end(
        step=100,
        loss=1.25,
        model=None,
        config={"do_eval": True},
        training_stats={
            "mean_by_class": {100: {-1.0: -0.5, 1.0: 0.6}},
        },
        eval_result={
            "correlation": 0.9,
            "component_summary": {"n_components": 13, "n_converged": 2},
            "components": _components(13, {1, 3}),
        },
        last_losses=[1.25],
    )

    assert "components: 2/13 converged [activation:h.1, activation:h.3]" in caplog.text


def test_non_convergence_error_includes_component_seed_summary():
    args = TrainingArguments(
        fail_on_non_convergence=True,
        min_convergent_seeds=1,
        convergent_metric="correlation",
        convergent_score_threshold=0.5,
    )

    message = format_non_convergence_error(
        actual=0,
        min_required=1,
        training_args=args,
        component_seed_summary={
            "n_components": 13,
            "n_satisfied_components": 11,
            "min_convergent_seeds": 1,
            "missing_component_ids": ["activation:h.11", "activation:h.12"],
        },
    )

    assert "component convergence: 11/13 have required convergent seed(s)" in message
    assert "remaining [activation:h.11, activation:h.12]" in message


def test_load_warning_is_component_aware(tmp_path, caplog):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with open(model_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump({
            "architecture": {"input_dim": 4},
            "metadata": {"mapping_kind": "activation"},
        }, handle)
    # Local-best summary: 11/13. Global best step (100) only has 2 converged —
    # load warning must use the local-best payload, not the global step snapshot.
    training_json = {
        "training_stats": {
            "components": {
                "50": _components(13, set(range(11))),
                "100": _components(13, {0, 1}),
            },
        },
        "best_score_checkpoint": {"global_step": 100, "correlation": 0.95},
        "convergence_info": {
            "converged": False,
            "convergent_count": 0,
            "min_convergent_seeds": 1,
            "convergence_metric": "correlation",
            "threshold": 0.5,
            "convergence_unit": "component",
            "component_summary": {
                "n_components": 13,
                "n_converged": 11,
            },
            "components": _components(13, set(range(11))),
        },
    }
    with open(model_dir / "training.json", "w", encoding="utf-8") as handle:
        json.dump(training_json, handle)

    mwg_module._convergence_warning_logged.clear()
    caplog.set_level("WARNING")
    mwg_module._check_convergence_warning(str(model_dir))

    assert "non-convergent component-split ACTIEND training" in caplog.text
    assert "11/13 converged, remaining [activation:h.11, activation:h.12]" in caplog.text
    assert "remaining 11 components" not in caplog.text
    assert "remaining 12 components" not in caplog.text


def test_load_warning_rebuilds_local_best_when_components_payload_missing(tmp_path, caplog):
    """Old training.json without convergence_info.components must still use local bests."""
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with open(model_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump({
            "architecture": {"input_dim": 4},
            "metadata": {"mapping_kind": "activation"},
        }, handle)

    def _step_payload(step_converged):
        metrics = {}
        convergence = {}
        for index in range(4):
            component_id = f"activation:h.{index}"
            ok = index in step_converged
            metrics[component_id] = {
                "component_index": index,
                "correlation": 0.9 if ok else 0.1,
                "mean_by_class": {-1.0: -0.8, 1.0: 0.8} if ok else {-1.0: -0.01, 1.0: 0.01},
            }
            convergence[component_id] = {"converged": ok}
        return {
            "summary": {"n_components": 4, "n_converged": len(step_converged)},
            "metrics_by_component": metrics,
            "convergence_by_component": convergence,
        }

    training_json = {
        "training_stats": {
            "components": {
                # Local bests: h0/h1/h2 converge here; h3 never converges.
                "10": _step_payload({0, 1, 2}),
                # Global best step alone would only mark h0 converged.
                "20": _step_payload({0}),
            },
        },
        "best_score_checkpoint": {"global_step": 20, "correlation": 0.5},
        "convergence_info": {
            "converged": False,
            "convergent_count": 0,
            "min_convergent_seeds": 1,
            "convergence_metric": "correlation",
            "threshold": 0.5,
            "convergent_mean_by_class_threshold": 0.5,
            "convergence_unit": "component",
            "component_summary": {"n_components": 4, "n_converged": 3},
        },
    }
    with open(model_dir / "training.json", "w", encoding="utf-8") as handle:
        json.dump(training_json, handle)

    mwg_module._convergence_warning_logged.clear()
    caplog.set_level("WARNING")
    mwg_module._check_convergence_warning(str(model_dir))

    assert "3/4 converged" in caplog.text
    assert "remaining [activation:h.3]" in caplog.text
    assert "remaining [activation:h.1, activation:h.2, activation:h.3]" not in caplog.text


def test_multiseed_load_warning_fires_when_component_count_is_below_required(tmp_path, caplog):
    model_dir = tmp_path / "run" / "model"
    seeds_dir = tmp_path / "run" / "seeds"
    model_dir.mkdir(parents=True)
    seeds_dir.mkdir()
    seed_report = {
        "convergent_count": 1,
        "min_convergent_seeds": 2,
        "convergence_metric": "correlation",
        "threshold": 0.5,
        "component_seed_summary": {
            "n_components": 3,
            "n_satisfied_components": 2,
            "min_convergent_seeds": 2,
            "component_convergent_count_by_id": {
                "activation:h.0": 2,
                "activation:h.1": 2,
                "activation:h.2": 1,
            },
            "missing_component_ids": ["activation:h.2"],
            "converged": False,
        },
    }
    with open(seeds_dir / "seed_report.json", "w", encoding="utf-8") as handle:
        json.dump(seed_report, handle)

    mwg_module._convergence_warning_logged.clear()
    caplog.set_level("WARNING")
    mwg_module._check_convergence_warning(str(model_dir))

    assert "component-split multi-seed GRADIEND training" in caplog.text
    assert "2/3 have required convergent seed(s)" in caplog.text


def test_training_json_multiseed_load_warning_names_blockers_and_stitch_status(tmp_path, caplog):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    with open(model_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump({
            "architecture": {"input_dim": 4},
            "metadata": {"signal_space": {"kind": "activation"}},
        }, handle)
    training_json = {
        "training_stats": {},
        "best_score_checkpoint": {"global_step": 100, "correlation": 0.95},
        "convergence_info": {
            "converged": False,
            "convergent_count": 0,
            "min_convergent_seeds": 1,
            "convergence_metric": "correlation",
            "threshold": 0.5,
            "convergence_unit": "component_seed",
            "component_seed_summary": {
                "n_components": 13,
                "n_satisfied_components": 11,
                "min_convergent_seeds": 1,
                "component_convergent_count_by_id": {
                    **{f"activation:h.{i}": 1 for i in range(11)},
                    "activation:wte": 0,
                    "activation:h.11": 0,
                },
                "missing_component_ids": ["activation:wte", "activation:h.11"],
                "converged": False,
                "stitched": False,
                "stitching_status": "skipped",
                "stitching_reason": "component convergence policy not satisfied",
            },
            "component_stitching": {
                "applied": False,
                "status": "skipped",
                "reason": "component convergence policy not satisfied",
                "missing_component_ids": ["activation:wte", "activation:h.11"],
            },
        },
    }
    with open(model_dir / "training.json", "w", encoding="utf-8") as handle:
        json.dump(training_json, handle)

    mwg_module._convergence_warning_logged.clear()
    caplog.set_level("WARNING")
    mwg_module._check_convergence_warning(str(model_dir))

    assert "component-split multi-seed ACTIEND training" in caplog.text
    assert "remaining [activation:wte, activation:h.11]" in caplog.text
    assert "Stitching status: component stitching skipped" in caplog.text
