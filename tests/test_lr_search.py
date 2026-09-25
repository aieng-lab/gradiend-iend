"""LR search: classification of probe runs and the bracket/bisect controller (pure logic, no GPU)."""

import json
import math

import numpy as np
import pytest

from gradiend.trainer.core.signal_checks import SignalNotDiverseError, assert_signal_diverse
from gradiend.trainer.core.lr_search import (
    LRSearch,
    ProbeResult,
    RunState,
    classify_probe,
    nice_lr_grid,
    snap_to_grid,
)

STEPS = [0, 50, 100, 150, 200, 250]


def _result(scores, pos, neg, converged=False, losses=None):
    return ProbeResult(
        steps=STEPS[: len(scores)],
        scores=list(scores),
        converged=converged,
        mean_by_class=[{"1.0": p, "-1.0": n, "0.0": 0.0} for p, n in zip(pos, neg)],
        losses=losses or [0.1] * 10,
    )


COLLAPSE = ([0.5, 0.3, 0.05], [-0.95, -1.0, -1.0], [-1.0, -1.0, -1.0])
GOOD = ([0.1, 0.6, 0.95], [0.1, 0.5, 0.9], [-0.1, -0.5, -0.9])
FROZEN = ([0.56] * 4, [0.09] * 4, [-0.14] * 4)


class TestClassify:
    def test_converged_trusts_the_package_decision(self):
        assert classify_probe(_result(*GOOD, converged=True)) is RunState.CONVERGED

    def test_high_initial_correlation_is_not_convergence(self):
        # gemma-2 @1e-7: correlation 0.56 at init decays to 0.19, class means never separate
        scores = [0.56, 0.55, 0.54, 0.53, 0.50, 0.45, 0.40, 0.33, 0.26, 0.22, 0.19]
        pos = [0.096 - 0.01 * i for i in range(11)]
        neg = [-0.14 - 0.01 * i for i in range(11)]
        r = ProbeResult(
            steps=list(range(0, 550, 50)), scores=scores, converged=False,
            mean_by_class=[{"1.0": p, "-1.0": n} for p, n in zip(pos, neg)],
        )
        assert classify_probe(r) is RunState.DEGRADING

    def test_frozen_when_nothing_moves(self):
        assert classify_probe(_result(*FROZEN)) is RunState.FROZEN

    def test_collapse_both_poles_same_side(self):
        assert classify_probe(_result(*COLLAPSE)) is RunState.COLLAPSED

    def test_slow_when_still_improving(self):
        r = _result([0.0, 0.1, 0.2, 0.3], [0.0, 0.05, 0.1, 0.2], [0.0, -0.05, -0.1, -0.2])
        assert classify_probe(r) is RunState.SLOW

    def test_shrinking_pole_gap_is_drift_even_before_the_score_drops(self):
        r = _result([0.56, 0.56, 0.55], [0.3, 0.2, 0.1], [-0.3, -0.2, -0.1])
        assert classify_probe(r) is RunState.DEGRADING

    def test_nonfinite_loss_is_divergence(self):
        r = _result([0.5, 0.5], [0.1, 0.1], [-0.1, -0.1], losses=[0.1, float("nan")])
        assert classify_probe(r) is RunState.DIVERGED

    def test_direction(self):
        assert RunState.FROZEN.direction == +1 and RunState.SLOW.direction == +1
        assert RunState.COLLAPSED.direction == -1 and RunState.DEGRADING.direction == -1
        assert RunState.CONVERGED.direction == 0


class TestGrid:
    def test_coarse_grid_is_one_two_five(self):
        assert nice_lr_grid(1e-8, 1e-7) == pytest.approx([2e-8, 5e-8])

    def test_fine_grid_adds_intermediates(self):
        assert nice_lr_grid(1e-8, 1e-7, "fine") == pytest.approx([1.5e-8, 2e-8, 3e-8, 5e-8, 7e-8])

    def test_open_interval(self):
        assert nice_lr_grid(2e-8, 5e-8) == []

    def test_snap(self):
        assert snap_to_grid(1.1e-6) == pytest.approx(1e-6)
        assert snap_to_grid(3.4e-7) in (pytest.approx(2e-7), pytest.approx(5e-7))

    def test_bad_resolution(self):
        with pytest.raises(ValueError):
            nice_lr_grid(1e-8, 1e-7, "nope")


def _window_run(good_lo, good_hi):
    """LR below the window is frozen, above it collapses, inside it converges."""
    calls = []

    def run(lr):
        calls.append(lr)
        if lr < good_lo:
            return _result(*FROZEN)
        if lr > good_hi:
            return _result(*COLLAPSE)
        return _result(*GOOD, converged=True)

    return run, calls


class TestSearch:
    def test_finds_a_convergent_lr_expanding_upward(self):
        run, calls = _window_run(3e-7, 3e-6)
        res = LRSearch(run, 1e-9).search()
        assert res.status == "converged"
        assert 3e-7 <= res.best_lr <= 3e-6
        assert calls[:3] == pytest.approx([1e-9, 1e-8, 1e-7])

    def test_finds_a_convergent_lr_expanding_downward(self):
        run, _ = _window_run(3e-7, 3e-6)
        res = LRSearch(run, 1e-3).search()
        assert res.status == "converged" and 3e-7 <= res.best_lr <= 3e-6

    def test_bisects_a_narrow_window_between_the_decades(self):
        run, _ = _window_run(4e-7, 6e-7)  # only 5e-7 converges on the coarse grid
        res = LRSearch(run, 1e-6).search()
        assert res.status == "converged" and res.best_lr == pytest.approx(5e-7)

    def test_no_convergent_lr_closes_the_bracket_with_a_verdict(self):
        # frozen below 1e-8, drifting from 1e-8 up: nothing converges anywhere
        def run(lr):
            if lr < 1e-8:
                return _result(*FROZEN)
            return _result([0.56, 0.4, 0.1], [0.05, 0.02, 0.0], [-0.2, -0.1, -0.05])

        res = LRSearch(run, 1e-6, max_probes=20).search()
        assert res.status == "no_convergent_lr"
        low, high = res.bracket
        assert high / low <= 2.0 + 1e-9
        assert res.best_lr is None
        assert len(res.probes) < 20  # it stopped by itself, well under the budget

    def test_never_probes_the_same_lr_twice(self):
        run, calls = _window_run(1e-5, 2e-5)
        LRSearch(run, 1e-9, max_probes=20).search()
        assert len(calls) == len(set(calls))

    def test_budget_exhaustion_is_reported_not_disguised(self):
        run, _ = _window_run(1e30, 1e31)  # frozen everywhere, bounds far away
        res = LRSearch(run, 1e-9, max_probes=3).search()
        assert res.status == "budget_exhausted" and len(res.probes) == 3

    def test_lr_bounds_stop_the_expansion(self):
        run, _ = _window_run(1e30, 1e31)
        res = LRSearch(run, 1e-3, lr_bounds=(1e-12, 1e-1), max_probes=20).search()
        assert res.status == "no_convergent_lr"
        assert max(p.lr for p in res.probes) <= 1e-1 * (1 + 1e-9)

    def test_failed_confirmation_demotes_the_probe_and_continues(self):
        run, _ = _window_run(1e-7, 1e-5)

        def confirm(lr):  # the short probe converged, the full budget collapses above 1e-6
            if lr > 1e-6:
                return _result(*COLLAPSE)
            return _result(*GOOD, converged=True)

        res = LRSearch(run, 1e-4, confirm=confirm).search()
        assert res.status == "converged" and res.best_lr <= 1e-6
        demoted = [p for p in res.probes if p.confirmed is False]
        assert demoted and all(p.state is RunState.COLLAPSED for p in demoted)

    def test_validation(self):
        with pytest.raises(ValueError):
            LRSearch(lambda lr: None, 0.0)
        with pytest.raises(ValueError):
            LRSearch(lambda lr: None, 1e-6, factor=1.0)


class TestSignalDiversity:
    def test_identical_rows_of_a_class_are_rejected(self):
        x = np.vstack([np.ones((5, 4)), -np.ones((5, 4))])
        y = [1] * 5 + [-1] * 5
        with pytest.raises(SignalNotDiverseError, match="identical"):
            assert_signal_diverse(x, y, where="unit")

    def test_varied_rows_pass(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(10, 4))
        assert_signal_diverse(x, [1] * 5 + [-1] * 5)


class _FakeTrainer:
    """Mimics Trainer.train(): afterwards it points at the checkpoint it wrote and holds the model."""

    def __init__(self, tmp_path):
        self._base_model_arg = "base-model"
        self._model_arg = "base-model"
        self._model_instance = None
        self._training_args = type("A", (), {"max_steps": 500, "eval_steps": 50})()
        self.started_from = []
        self.tmp_path = tmp_path

    def train(self, **kw):
        self.started_from.append((self._model_arg, self._model_instance))
        out = kw["experiment_dir"] + "/model"
        import os

        os.makedirs(out, exist_ok=True)
        good = kw["learning_rate"] >= 1e-6
        payload = {
            "training_stats": {
                "scores": {"0": 0.1, "50": 0.9 if good else 0.1},
                "mean_by_class": {"0": {"1.0": 0.1, "-1.0": -0.1},
                                  "50": {"1.0": 0.9, "-1.0": -0.9} if good else {"1.0": 0.1, "-1.0": -0.1}},
            },
            "losses": [0.1],
            "convergence_info": {"converged": good},
        }
        with open(out + "/training.json", "w") as fh:
            json.dump(payload, fh)
        with open(out + "/model.safetensors", "wb") as fh:
            fh.write(b"x" * 1024)  # stands in for a checkpoint
        self._model_arg = out  # what the real Trainer does
        self._model_instance = object()  # the trained model stays resident
        return self


def test_every_probe_starts_from_the_base_model_and_the_trainer_is_left_untouched(tmp_path):
    from gradiend.trainer.core.lr_search import tune_learning_rate

    trainer = _FakeTrainer(tmp_path)
    res = tune_learning_rate(
        trainer, 1e-8, experiment_dir=str(tmp_path / "exp"), probe_steps=100, confirm=False, max_probes=6,
    )
    assert res.status == "converged" and res.best_lr >= 1e-6
    assert len(trainer.started_from) >= 2
    assert all(arg == "base-model" and inst is None for arg, inst in trainer.started_from), (
        "a later probe continued from the previous probe's checkpoint"
    )
    assert trainer._model_arg == "base-model" and trainer._model_instance is None
    assert (tmp_path / "exp" / "lr_search.json").exists()


def test_confirmation_reruns_at_the_full_budget(tmp_path):
    from gradiend.trainer.core.lr_search import tune_learning_rate

    seen = []
    trainer = _FakeTrainer(tmp_path)
    original = trainer.train

    def spy(**kw):
        seen.append(kw.get("max_steps"))
        return original(**kw)

    trainer.train = spy
    tune_learning_rate(
        trainer, 1e-6, experiment_dir=str(tmp_path / "exp"), probe_steps=100, confirm=True,
        center_window=False,
    )
    assert seen == [100, 500]


def test_adapter_installs_the_guard_always_and_a_fresh_stop_callback_unless_disabled(tmp_path):
    from gradiend.trainer.core.lr_search import ProbeStopCallback, tune_learning_rate
    from gradiend.trainer.core.signal_checks import SignalDiversityCallback

    seen = []
    trainer = _FakeTrainer(tmp_path)
    original = trainer.train
    trainer.train = lambda **kw: (seen.append(kw.get("callbacks")), original(**kw))[1]
    tune_learning_rate(trainer, 1e-6, experiment_dir=str(tmp_path / "a"), probe_steps=100, confirm=False,
                       center_window=False)
    assert [type(c) for c in seen[0]] == [SignalDiversityCallback, ProbeStopCallback]
    seen.clear()
    tune_learning_rate(trainer, 1e-6, experiment_dir=str(tmp_path / "b"), probe_steps=100, confirm=False,
                       center_window=False, early_stop=False)
    assert [[type(c) for c in cbs] for cbs in seen] == [[SignalDiversityCallback]]


class TestWindowCentering:
    def test_returns_the_centre_of_the_convergent_window_not_its_first_hit(self):
        run, calls = _window_run(1e-6, 1e-4)  # convergent: 1e-6 .. 1e-4 (grid: 1,2,5 x)
        res = LRSearch(run, 1e-6).search()  # first hit is the lower edge
        assert res.status == "converged"
        assert res.window == pytest.approx((1e-6, 1e-4)) or res.window[0] >= 1e-6
        centre = (res.window[0] * res.window[1]) ** 0.5
        assert res.best_lr not in (res.window[0], res.window[1])
        assert abs(math.log(res.best_lr / centre)) < math.log(2.6)

    def test_window_walk_stops_at_the_first_non_convergent_neighbour(self):
        run, calls = _window_run(4e-7, 6e-7)  # only 5e-7 converges
        res = LRSearch(run, 5e-7).search()
        assert res.best_lr == pytest.approx(5e-7) and res.window == pytest.approx((5e-7, 5e-7))
        assert sorted(calls) == pytest.approx([2e-7, 5e-7, 1e-6])  # exactly the two neighbours were checked

    def test_window_probes_have_their_own_budget(self):
        run, calls = _window_run(1e-9, 1e9)  # everything converges
        res = LRSearch(run, 1e-6, max_probes=1, max_window_probes=3).search()
        assert res.status == "converged" and len(calls) == 1 + 3

    def test_both_sides_are_explored_even_when_the_window_is_wider_than_the_budget(self):
        run, calls = _window_run(1e-9, 1e9)  # everything converges: the walk is only limited by its budget
        res = LRSearch(run, 1e-6, max_window_probes=4).search()
        assert sorted(c for c in calls if c != 1e-6) == pytest.approx([2e-7, 5e-7, 2e-6, 5e-6])  # two per side
        assert res.window == pytest.approx((2e-7, 5e-6))

    def test_disabled(self):
        run, calls = _window_run(1e-9, 1e9)
        res = LRSearch(run, 1e-6, center_window=False).search()
        assert res.best_lr == pytest.approx(1e-6) and len(calls) == 1

    def test_ties_go_to_the_lower_lr(self):
        run, _ = _window_run(1e-6, 2e-6)  # two convergent grid points, equal distance to the centre
        res = LRSearch(run, 1e-6).search()
        assert res.best_lr == pytest.approx(1e-6)

    def test_reported_window_probes_are_never_repeated(self):
        run, calls = _window_run(1e-7, 1e-5)
        LRSearch(run, 1e-9).search()
        assert len(calls) == len(set(round(c, 15) for c in calls))


class TestGridResolution:
    def test_bracket_closes_on_the_grid_not_on_a_ratio(self):
        # frozen below 3e-8, collapsed from 3e-8 up; fine grid has interior points down to ratio 1.33
        def run(lr):
            return _result(*FROZEN) if lr < 3e-8 else _result(*COLLAPSE)

        res = LRSearch(run, 1e-6, resolution="fine", max_probes=30).search()
        assert res.status == "no_convergent_lr"
        low, high = res.bracket
        assert nice_lr_grid(low, high, "fine") == []  # nothing untried left inside
        assert high / low < 1.7  # finer than the old ratio-2 stop

    def test_coarse_grid_closes_at_two_to_two_and_a_half(self):
        def run(lr):
            return _result(*FROZEN) if lr < 3e-8 else _result(*COLLAPSE)

        res = LRSearch(run, 1e-6, max_probes=30).search()
        low, high = res.bracket
        assert nice_lr_grid(low, high) == [] and high / low <= 2.5 + 1e-9

    def test_optional_min_ratio_still_stops_earlier(self):
        def run(lr):
            return _result(*FROZEN) if lr < 3e-8 else _result(*COLLAPSE)

        res = LRSearch(run, 1e-6, resolution="fine", min_ratio=2.0, max_probes=30).search()
        low, high = res.bracket
        assert high / low <= 2.0 + 1e-9

    def test_expansion_by_division_does_not_reprobe_grid_values(self):
        # 1e-6 / 10 is not bit-identical to the grid literal 1e-7; it must still count as tried
        run, calls = _window_run(1e30, 1e31)
        LRSearch(run, 1e-3, lr_bounds=(1e-12, 1e-2), max_probes=30).search()
        assert len(calls) == len(set(round(c, 15) for c in calls))

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            LRSearch(lambda lr: None, 1e-6, resolution="nope")
        with pytest.raises(ValueError):
            LRSearch(lambda lr: None, 1e-6, min_ratio=1.0)


def test_grid_neighbor():
    from gradiend.trainer.core.lr_search import grid_neighbor

    assert grid_neighbor(1e-7, +1) == pytest.approx(2e-7)
    assert grid_neighbor(5e-7, +1) == pytest.approx(1e-6)
    assert grid_neighbor(1e-7, -1) == pytest.approx(5e-8)
    assert grid_neighbor(1e-7, +1, "fine") == pytest.approx(1.5e-7)
    assert grid_neighbor(2e-7, -1, "fine") == pytest.approx(1.5e-7)
    with pytest.raises(ValueError):
        grid_neighbor(1e-7, 0)


class TestProbeStopCallback:
    @staticmethod
    def _stats(scores, pos, neg):
        return {
            "scores": {i * 50: s for i, s in enumerate(scores)},
            "mean_by_class": {i * 50: {1.0: p, -1.0: n} for i, (p, n) in enumerate(zip(pos, neg))},
        }

    def _step(self, cb, stats, loss=0.1, step=0):
        control = {"should_stop": False}
        cb.on_step_end(step=step, loss=loss, model=None, config={}, control=control, training_stats=stats)
        return control["should_stop"]

    def test_stops_on_collapse_but_not_on_drift_or_slow_progress(self):
        from gradiend.trainer.core.lr_search import ProbeStopCallback

        cb = ProbeStopCallback()
        assert not self._step(cb, self._stats([0.5], [0.1], [-0.1]))  # below min_evals
        assert self._step(cb, self._stats([0.5, 0.1], [0.1, -0.9], [-0.1, -1.0]))
        assert cb.stopped_state is RunState.COLLAPSED

        cb = ProbeStopCallback()
        self._step(cb, self._stats([0.56], [0.1], [-0.14]))
        assert not self._step(cb, self._stats([0.56, 0.2], [0.1, 0.02], [-0.14, -0.2]))  # drift: not decisive

    def test_only_judges_when_a_new_evaluation_exists(self):
        from gradiend.trainer.core.lr_search import ProbeStopCallback

        calls = []

        def classify(result):
            calls.append(len(result.scores))
            return RunState.SLOW

        cb = ProbeStopCallback(classify=classify)
        stats = self._stats([0.1, 0.2], [0.0, 0.1], [0.0, -0.1])
        self._step(cb, stats)
        self._step(cb, stats)  # same evaluations again (a step without evaluation)
        assert calls == [2]

    def test_nonfinite_loss_stops_immediately(self):
        from gradiend.trainer.core.lr_search import ProbeStopCallback

        cb = ProbeStopCallback()
        assert self._step(cb, {}, loss=float("nan"), step=7)
        assert cb.stopped_state is RunState.DIVERGED and cb.stopped_step == 7

    def test_configurable_stop_states(self):
        from gradiend.trainer.core.lr_search import ProbeStopCallback

        cb = ProbeStopCallback(stop_states=(RunState.DEGRADING,))
        self._step(cb, self._stats([0.56], [0.1], [-0.14]))
        assert self._step(cb, self._stats([0.56, 0.2], [0.1, 0.02], [-0.14, -0.2]))

    def test_accepts_integer_and_string_step_keys(self):
        stats = {"scores": {"0": 0.1, "50": 0.2}, "mean_by_class": {"0": {"1.0": 0.0}, "50": {"1.0": 0.1}}}
        r = ProbeResult.from_training_stats(stats)
        assert r.steps == [0, 50] and r.mean_by_class[1] == {"1.0": 0.1}


class TestTrainingLoopHonoursShouldStop:
    def test_step_loop_stops_mid_epoch_and_still_writes_the_results(self, tmp_path):
        import torch
        from torch.utils.data import DataLoader

        from gradiend.trainer.core.arguments import TrainingArguments
        from gradiend.trainer.core.callbacks import TrainingCallback
        from gradiend.trainer.core.dataset import GradientTrainingDataset
        from gradiend.trainer.core.stats import load_training_stats
        from gradiend.trainer.core.training import train
        from tests.test_training_loop import MockModelWithGradiend, MockTrainingData

        data = MockTrainingData([{"factual": torch.randn(10), "alternative": torch.randn(10), "label": 1.0}] * 50)
        dataset = GradientTrainingDataset(
            training_data=data, gradient_creator=lambda inputs: torch.randn(100), source="factual", target="diff",
        )
        steps = []

        class StopAtThree(TrainingCallback):
            def on_step_end(self, step, loss, model, config, **kwargs):
                steps.append(step)
                if step >= 3:
                    kwargs["control"]["should_stop"] = True

        args = TrainingArguments(
            output_dir=str(tmp_path), max_steps=40, eval_steps=100, train_batch_size=1,
            num_train_epochs=1, convergent_score_threshold=None,
        )
        out = train(
            model_with_gradiend=MockModelWithGradiend(), data=DataLoader(dataset, batch_size=1),
            training_args=args, callbacks=[StopAtThree()],
        )
        assert isinstance(out, str)
        assert max(steps) == 3, f"training continued past the stop request: {steps}"
        assert load_training_stats(str(tmp_path))["training_stats"]["global_step"] == 3


def test_probe_result_reads_training_json(tmp_path):
    payload = {
        "training_stats": {
            "scores": {"0": 0.5, "50": 0.6},
            "mean_by_class": {"0": {"1.0": 0.1, "-1.0": -0.1}, "50": {"1.0": 0.5, "-1.0": -0.5}},
        },
        "losses": [0.1, 0.2],
        "convergence_info": {"converged": True, "threshold": 0.5, "convergence_metric": "correlation"},
    }
    path = tmp_path / "training.json"
    path.write_text(json.dumps(payload))
    r = ProbeResult.from_training_json(str(path))
    assert r.steps == [0, 50] and r.scores == [0.5, 0.6] and r.converged
    assert r.mean_by_class[1]["1.0"] == 0.5 and r.metric == "correlation"
    assert classify_probe(r) is RunState.CONVERGED


class TestSignalDiversityCallback:
    @staticmethod
    def _eval(std, n=20):
        return {"std_by_class": {1.0: std, -1.0: std, 0.0: 0.3}, "n_by_class": {1.0: n, -1.0: n, 0.0: n}}

    def test_identical_inputs_raise_at_the_first_evaluation(self):
        from gradiend.trainer.core.signal_checks import SignalDiversityCallback

        cb = SignalDiversityCallback()
        with pytest.raises(SignalNotDiverseError, match="identical"):
            cb.on_step_end(step=0, loss=0.1, model=None, config={}, eval_result=self._eval(0.0))

    def test_spread_encodings_pass_and_the_check_is_then_inert(self):
        from gradiend.trainer.core.signal_checks import SignalDiversityCallback

        cb = SignalDiversityCallback()
        cb.on_step_end(step=0, loss=0.1, model=None, config={}, eval_result=self._eval(0.05))
        assert cb.checked
        # a later, saturated evaluation (collapse) is the LR search's business, not a signal bug
        cb.on_step_end(step=50, loss=0.1, model=None, config={}, eval_result=self._eval(0.0))

    def test_one_diverse_class_is_enough(self):
        from gradiend.trainer.core.signal_checks import SignalDiversityCallback

        result = {"std_by_class": {1.0: 0.0, -1.0: 0.2}, "n_by_class": {1.0: 20, -1.0: 20}}
        SignalDiversityCallback().on_step_end(step=0, loss=0.1, model=None, config={}, eval_result=result)

    def test_tiny_classes_and_neutral_rows_are_not_evidence(self):
        from gradiend.trainer.core.signal_checks import SignalDiversityCallback

        cb = SignalDiversityCallback(min_examples=4)
        result = {"std_by_class": {1.0: 0.0, -1.0: 0.0, 0.0: 0.0}, "n_by_class": {1.0: 2, -1.0: 3, 0.0: 50}}
        cb.on_step_end(step=0, loss=0.1, model=None, config={}, eval_result=result)  # nothing judged: no error

    def test_waits_for_an_evaluation_that_carries_the_statistic(self):
        from gradiend.trainer.core.signal_checks import SignalDiversityCallback

        cb = SignalDiversityCallback()
        cb.on_step_end(step=0, loss=0.1, model=None, config={})  # no eval_result
        cb.on_step_end(step=1, loss=0.1, model=None, config={}, eval_result={"correlation": 0.1})
        assert not cb.checked


class TestDiversityCallbackInTheRealLoop:
    @staticmethod
    def _run(tmp_path, std):
        import torch
        from torch.utils.data import DataLoader

        from gradiend.trainer.core.arguments import TrainingArguments
        from gradiend.trainer.core.dataset import GradientTrainingDataset
        from gradiend.trainer.core.signal_checks import SignalDiversityCallback
        from gradiend.trainer.core.training import train
        from tests.test_training_loop import MockModelWithGradiend, MockTrainingData

        data = MockTrainingData([{"factual": torch.randn(10), "alternative": torch.randn(10), "label": 1.0}] * 20)
        dataset = GradientTrainingDataset(
            training_data=data, gradient_creator=lambda inputs: torch.randn(100), source="factual", target="diff",
        )

        def evaluate_fn(config=None, training_stats=None, **kw):
            return {
                "correlation": 0.5,
                "mean_by_class": {1.0: 0.1, -1.0: -0.1},
                "std_by_class": {1.0: std, -1.0: std},
                "n_by_class": {1.0: 10, -1.0: 10},
            }

        args = TrainingArguments(
            output_dir=str(tmp_path), max_steps=5, eval_steps=100, train_batch_size=1, num_train_epochs=1,
            evaluate_fn=evaluate_fn, do_eval=True, convergent_score_threshold=None,
        )
        return train(
            model_with_gradiend=MockModelWithGradiend(), data=DataLoader(dataset, batch_size=1),
            training_args=args, callbacks=[SignalDiversityCallback()],
        )

    def test_constant_signal_aborts_the_run(self, tmp_path):
        with pytest.raises(SignalNotDiverseError):
            self._run(tmp_path, std=0.0)

    def test_diverse_signal_trains_normally(self, tmp_path):
        assert isinstance(self._run(tmp_path, std=0.2), str)


def test_probe_checkpoints_are_deleted_and_only_training_json_stays(tmp_path):
    from gradiend.trainer.core.lr_search import tune_learning_rate

    tune_learning_rate(
        _FakeTrainer(tmp_path), 1e-6, experiment_dir=str(tmp_path / "exp"), probe_steps=100, confirm=False,
        center_window=False,
    )
    files = [f for _r, _d, fs in __import__("os").walk(tmp_path / "exp" / "lr_search") for f in fs]
    assert files and set(files) == {"training.json"}
    assert (tmp_path / "exp" / "lr_search.json").exists()


def test_probe_checkpoints_can_be_kept(tmp_path):
    import os

    from gradiend.trainer.core.lr_search import tune_learning_rate

    tune_learning_rate(
        _FakeTrainer(tmp_path), 1e-6, experiment_dir=str(tmp_path / "exp"), probe_steps=100, confirm=False,
        center_window=False, keep_probe_artifacts=True,
    )
    files = {f for _r, _d, fs in os.walk(tmp_path / "exp" / "lr_search") for f in fs}
    assert "model.safetensors" in files


def test_no_convergent_learning_rate_carries_the_trace():
    from gradiend.trainer.core.lr_search import LRSearchResult, NoConvergentLearningRate

    result = LRSearchResult("no_convergent_lr", None, [], bracket=(1e-8, 2e-8))
    err = NoConvergentLearningRate("boom", result)
    assert isinstance(err, RuntimeError) and err.result is result and "boom" in str(err)


def test_search_context_is_stored_next_to_the_result(tmp_path):
    from gradiend.trainer.core.lr_search import tune_learning_rate

    tune_learning_rate(
        _FakeTrainer(tmp_path), 1e-6, experiment_dir=str(tmp_path / "exp"), probe_steps=100, confirm=False,
        center_window=False, metadata={"backend": "gradiend", "probe_steps": 100},
    )
    saved = json.loads((tmp_path / "exp" / "lr_search.json").read_text())
    assert saved["backend"] == "gradiend" and saved["probe_steps"] == 100 and saved["initial_lr"] == 1e-6
    assert saved["status"] == "converged" and saved["best_lr"] >= 1e-6
