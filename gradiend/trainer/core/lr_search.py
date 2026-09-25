"""
Automatic learning-rate search for GRADIEND-style training.

A short probe run is *classified* (not just scored) and the classification decides the next
learning rate: raise it when the run is too slow to learn, lower it when it overshoots or
degrades, stop when it converges. Once one LR is "too low" and a larger one is "too high" the
bracket is bisected in log space on a small grid of readable values (1/2/5 x 10^k by default)
until it is closed. If the bracket closes without a convergent run the search returns
``no_convergent_lr`` with the full trace instead of continuing forever: that verdict means no
learning rate can fix the run (e.g. the reconstruction target is not representable by the
latent), so the cause has to be found elsewhere.

The bracket closes when no untried grid value remains strictly inside it, so the grid alone
defines the finest spacing tried (``resolution="coarse"``: neighbours 2-2.5x apart, ``"fine"``:
1.33-1.67x). When a run converges the search additionally walks outward along the grid to find the
edges of the convergent window and returns the LR closest (in log space) to its centre, so the
result does not sit on the edge of the stable range.

The search logic (:class:`LRSearch`, :func:`classify_probe`) is pure and takes any callable
``run(lr) -> ProbeResult``; :func:`tune_learning_rate` is the adapter that drives a real
:class:`~gradiend.trainer.trainer.Trainer`. :class:`ProbeStopCallback` ends a probe as soon as it
is decisively collapsed or diverged instead of training the remaining steps.

Notes:
    * Probes are short. GRADIEND is schedule-sensitive (equal ``lr * steps`` does not give an
      equal optimisation), so pass ``confirm=True`` to re-run a converged probe at the
      configured full budget before accepting it.
    * The classification uses the package's own convergence decision
      (``convergence_info['converged']``, which includes the class-mean check) as ground truth for
      "converged". A run can have a high correlation at initialisation without being converged.
"""

from __future__ import annotations

import contextlib
import enum
import gc
import json
import math
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from gradiend.trainer.core.callbacks import TrainingCallback
from gradiend.trainer.core.signal_checks import SignalDiversityCallback
from gradiend.util.logging import get_logger

logger = get_logger(__name__)

__all__ = [
    "RunState",
    "ProbeResult",
    "Probe",
    "LRSearchResult",
    "NoConvergentLearningRate",
    "LRSearch",
    "classify_probe",
    "nice_lr_grid",
    "snap_to_grid",
    "grid_neighbor",
    "ProbeStopCallback",
    "tune_learning_rate",
]

COARSE_GRID: Tuple[float, ...] = (1.0, 2.0, 5.0)
FINE_GRID: Tuple[float, ...] = (1.0, 1.5, 2.0, 3.0, 5.0, 7.0)


class RunState(str, enum.Enum):
    """What a probe run did. Decides the direction of the next learning rate."""

    CONVERGED = "converged"
    FROZEN = "frozen"  # nothing moved: LR too low
    SLOW = "slow"  # improving but not converged in the budget: LR too low
    DEGRADING = "degrading"  # improved/started well, then got worse: LR too high (slow drift)
    COLLAPSED = "collapsed"  # both poles on the same side / saturated: LR too high
    DIVERGED = "diverged"  # non-finite loss: LR too high

    @property
    def direction(self) -> int:
        """+1: raise the LR, -1: lower it, 0: stop."""
        if self in (RunState.FROZEN, RunState.SLOW):
            return +1
        if self in (RunState.DEGRADING, RunState.COLLAPSED, RunState.DIVERGED):
            return -1
        return 0


@dataclass
class ProbeResult:
    """The evidence one probe run leaves behind (all histories are aligned to ``steps``)."""

    steps: List[int]
    scores: List[float]  # convergence-metric history (correlation, or min_auc_n_o for one-pole)
    converged: bool
    threshold: Optional[float] = None
    metric: Optional[str] = None
    mean_by_class: List[Dict[str, float]] = field(default_factory=list)  # per eval, {"-1.0": m, "1.0": m}
    losses: List[float] = field(default_factory=list)

    @classmethod
    def from_training_stats(
        cls,
        stats: Mapping[str, Any],
        *,
        converged: bool = False,
        threshold: Optional[float] = None,
        metric: Optional[str] = None,
        losses: Optional[Sequence[float]] = None,
    ) -> "ProbeResult":
        """Build from a ``training_stats`` mapping (live in memory or loaded from ``training.json``).

        Step keys are ints while training runs and strings once written to JSON; both are accepted.
        """
        scores_by_step: Dict[int, float] = {int(k): float(v) for k, v in (stats.get("scores") or {}).items()}
        means_by_step: Dict[int, Mapping[Any, float]] = {
            int(k): v for k, v in (stats.get("mean_by_class") or {}).items()
        }
        steps = sorted(scores_by_step)
        return cls(
            steps=steps,
            scores=[scores_by_step[s] for s in steps],
            converged=converged,
            threshold=threshold,
            metric=metric,
            mean_by_class=[
                {str(k): float(v) for k, v in (means_by_step.get(s) or {}).items()} for s in steps
            ],
            losses=[float(x) for x in (losses or [])],
        )

    @classmethod
    def from_training_json(cls, path: str) -> "ProbeResult":
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        info = data.get("convergence_info") or {}
        return cls.from_training_stats(
            data.get("training_stats") or {},
            converged=bool(info.get("converged", False)),
            threshold=info.get("threshold"),
            metric=info.get("convergence_metric"),
            losses=data.get("losses"),
        )


def _signed_pole_means(means: Mapping[str, float]) -> Tuple[Optional[float], Optional[float]]:
    """(mean of the +1 class, mean of the -1 class) or None where absent."""
    pos = neg = None
    for key, val in means.items():
        try:
            label = float(key)
        except (TypeError, ValueError):
            continue
        if label > 0:
            pos = float(val)
        elif label < 0:
            neg = float(val)
    return pos, neg


def classify_probe(
    result: ProbeResult,
    *,
    frozen_range: float = 0.05,
    degrade_margin: float = 0.15,
    collapse_abs_mean: float = 0.5,
) -> RunState:
    """Classify a probe run.

    Args:
        frozen_range: total movement (max - min) of the score history, and of the class-mean gap,
            below which nothing is considered to have moved.
        degrade_margin: how far the final score must sit below the best score to count as degrading.
        collapse_abs_mean: both poles on the same side, each with ``|mean|`` above this, is a collapse.
    """
    if result.losses and not all(math.isfinite(x) for x in result.losses):
        return RunState.DIVERGED
    if result.scores and not all(math.isfinite(x) for x in result.scores):
        return RunState.DIVERGED
    if result.converged:
        return RunState.CONVERGED

    # Collapse: the +1 and -1 classes end on the same side, saturated.
    if result.mean_by_class:
        pos, neg = _signed_pole_means(result.mean_by_class[-1])
        if pos is not None and neg is not None:
            if pos * neg > 0 and min(abs(pos), abs(neg)) >= collapse_abs_mean:
                return RunState.COLLAPSED

    scores = result.scores
    if len(scores) >= 2:
        if scores[-1] < max(scores) - degrade_margin:
            return RunState.DEGRADING
        score_range = max(scores) - min(scores)
    else:
        score_range = 0.0

    gap_range = 0.0
    gaps = []
    for means in result.mean_by_class:
        pos, neg = _signed_pole_means(means)
        if pos is not None and neg is not None:
            gaps.append(pos - neg)
    if len(gaps) >= 2:
        gap_range = max(gaps) - min(gaps)

    if score_range < frozen_range and gap_range < frozen_range:
        return RunState.FROZEN
    # Moved, did not degrade, did not converge: still learning when the budget ended. A run whose
    # class-pole gap is *shrinking* is drifting to a common pole even if the score has not dropped yet.
    if len(gaps) >= 2 and gaps[-1] < gaps[0] - frozen_range:
        return RunState.DEGRADING
    return RunState.SLOW


def nice_lr_grid(lo: float, hi: float, resolution: str = "coarse") -> List[float]:
    """Readable learning rates ``m * 10^k`` strictly inside ``(lo, hi)``, ascending."""
    if resolution not in ("coarse", "fine"):
        raise ValueError(f"resolution must be 'coarse' or 'fine', got {resolution!r}")
    mantissas = COARSE_GRID if resolution == "coarse" else FINE_GRID
    out: List[float] = []
    k_lo = math.floor(math.log10(lo)) - 1
    k_hi = math.ceil(math.log10(hi)) + 1
    for k in range(k_lo, k_hi + 1):
        for m in mantissas:
            value = float(f"{m}e{k}")
            if lo * (1 + 1e-9) < value < hi * (1 - 1e-9):
                out.append(value)
    return sorted(set(out))


def snap_to_grid(value: float, resolution: str = "coarse") -> float:
    """Nearest readable learning rate in log space."""
    candidates = nice_lr_grid(value / 10.0, value * 10.0, resolution)
    return min(candidates, key=lambda c: abs(math.log(c) - math.log(value)))


def grid_neighbor(lr: float, direction: int, resolution: str = "coarse") -> float:
    """The adjacent readable learning rate above (``direction > 0``) or below (``< 0``) ``lr``."""
    if direction > 0:
        return nice_lr_grid(lr, lr * 10.5, resolution)[0]
    if direction < 0:
        return nice_lr_grid(lr / 10.5, lr, resolution)[-1]
    raise ValueError("direction must be non-zero")


class ProbeStopCallback(TrainingCallback):
    """End a probe as soon as its fate is decided.

    After every new evaluation the run so far is classified with the same rules as the final
    verdict; if the state is one of ``stop_states`` the training loop is told to stop
    (``control['should_stop']``) and the epoch-end hooks still write the checkpoint and
    ``training.json``. Only *decisive* states stop by default: a collapse (both poles saturated on
    one side) and a non-finite loss cannot recover, whereas a slow or drifting run might. A
    non-finite loss stops immediately, without waiting for an evaluation.

    Args:
        stop_states: states that end the run.
        min_evals: number of evaluations (including the initial one) required before judging.
    """

    def __init__(
        self,
        stop_states: Sequence[RunState] = (RunState.COLLAPSED, RunState.DIVERGED),
        *,
        min_evals: int = 2,
        classify: Callable[[ProbeResult], RunState] = classify_probe,
    ) -> None:
        self.stop_states = frozenset(stop_states)
        self.min_evals = int(min_evals)
        self.classify = classify
        self.stopped_state: Optional[RunState] = None
        self.stopped_step: Optional[int] = None
        self._seen_evals = 0

    def _stop(self, control: Dict[str, Any], state: RunState, step: int) -> None:
        self.stopped_state, self.stopped_step = state, int(step)
        control["should_stop"] = True
        logger.info("LR probe stopped early at step %s: %s", step, state.value)

    def on_step_end(self, step, loss, model, config, **kwargs):
        control = kwargs.get("control")
        if control is None or self.stopped_state is not None:
            return None
        if loss is not None and not math.isfinite(float(loss)):
            self._stop(control, RunState.DIVERGED, step)
            return None
        stats = kwargs.get("training_stats") or {}
        n_evals = len(stats.get("scores") or {})
        if n_evals == self._seen_evals:
            return None  # no evaluation happened at this step
        self._seen_evals = n_evals
        if n_evals < self.min_evals:
            return None
        state = self.classify(ProbeResult.from_training_stats(stats))
        if state in self.stop_states:
            self._stop(control, state, step)
        return None


@dataclass
class Probe:
    lr: float
    state: RunState
    final_score: Optional[float] = None
    best_score: Optional[float] = None
    confirmed: Optional[bool] = None
    steps_run: Optional[int] = None  # last evaluated step (shows early stops)
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        return d


@dataclass
class LRSearchResult:
    status: str  # "converged" | "no_convergent_lr" | "budget_exhausted"
    best_lr: Optional[float]
    probes: List[Probe]
    bracket: Optional[Tuple[float, float]] = None  # (highest "too low", lowest "too high")
    window: Optional[Tuple[float, float]] = None  # (lowest, highest) convergent LR probed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "best_lr": self.best_lr,
            "bracket": list(self.bracket) if self.bracket else None,
            "window": list(self.window) if self.window else None,
            "probes": [p.to_dict() for p in self.probes],
        }


class NoConvergentLearningRate(RuntimeError):
    """The search ended without a convergent learning rate; ``result`` carries the full trace."""

    def __init__(self, message: str, result: "LRSearchResult") -> None:
        super().__init__(message)
        self.result = result


class LRSearch:
    """Bracket-and-bisect controller over a probe function.

    Args:
        run: ``run(lr) -> ProbeResult``; must be deterministic enough that repeating an LR is not needed.
        initial_lr: first guess (use a readable value such as ``1e-6``).
        factor: expansion step while no bracket exists (10 by default).
        min_ratio: optional extra stop: the bracket is also closed once ``high / low <= min_ratio``.
            ``None`` (default) closes it only when no untried grid value lies strictly inside, so the
            grid (``resolution``) alone sets the finest spacing.
        max_probes: budget for the bracketing phase. On exhaustion the status is ``budget_exhausted``.
        resolution: ``"coarse"`` (1/2/5) or ``"fine"`` (1/1.5/2/3/5/7) grid for bisection candidates and
            window edges.
        lr_bounds: never propose an LR outside ``(lo, hi)``.
        confirm: optional ``confirm(lr) -> ProbeResult`` re-run (e.g. at the full budget) of a converged
            probe. A failed confirmation is classified like any probe and the search continues.
        center_window: once a probe converges, walk outward along the grid until the neighbours stop
            converging and return the convergent LR closest (in log space) to the window's centre,
            instead of the first hit, which may sit on the edge of the stable range.
        max_window_probes: extra probes allowed for that walk (separate from ``max_probes``), split
            evenly between the two sides (upward gets the odd one); the worst case is therefore
            ``max_probes + max_window_probes`` probes.
    """

    def __init__(
        self,
        run: Callable[[float], ProbeResult],
        initial_lr: float,
        *,
        factor: float = 10.0,
        min_ratio: Optional[float] = None,
        max_probes: int = 12,
        resolution: str = "coarse",
        lr_bounds: Tuple[float, float] = (1e-12, 1.0),
        confirm: Optional[Callable[[float], ProbeResult]] = None,
        classify: Callable[[ProbeResult], RunState] = classify_probe,
        center_window: bool = True,
        max_window_probes: int = 4,
    ) -> None:
        if initial_lr <= 0:
            raise ValueError("initial_lr must be positive")
        if factor <= 1:
            raise ValueError("factor must be > 1")
        if resolution not in ("coarse", "fine"):
            raise ValueError(f"resolution must be 'coarse' or 'fine', got {resolution!r}")
        if min_ratio is not None and min_ratio <= 1:
            raise ValueError("min_ratio must be > 1 (or None)")
        self.run = run
        self.initial_lr = self._norm(initial_lr)
        self.factor = float(factor)
        self.min_ratio = None if min_ratio is None else float(min_ratio)
        self.max_probes = int(max_probes)
        self.resolution = resolution
        self.lr_bounds = lr_bounds
        self.confirm = confirm
        self.classify = classify
        self.center_window = bool(center_window)
        self.max_window_probes = int(max_window_probes)
        self.probes: List[Probe] = []

    @staticmethod
    def _norm(lr: float) -> float:
        """Canonical float for an LR, so 1e-6 / 10 and the grid value 1e-7 compare equal."""
        return float(f"{lr:.9g}")

    # ------------------------------------------------------------------ bracket
    def _tried(self) -> Dict[float, Probe]:
        return {self._norm(p.lr): p for p in self.probes}

    def _bracket(self) -> Tuple[Optional[float], Optional[float]]:
        """(highest 'too low' LR below the lowest 'too high' LR, lowest 'too high' LR)."""
        highs = [p.lr for p in self.probes if p.state.direction < 0]
        high = min(highs) if highs else None
        lows = [p.lr for p in self.probes if p.state.direction > 0 and (high is None or p.lr < high)]
        low = max(lows) if lows else None
        return low, high

    def _next_lr(self) -> Optional[float]:
        low, high = self._bracket()
        tried = self._tried()
        lo_b, hi_b = self.lr_bounds
        if low is None and high is None:
            return self.initial_lr
        if high is None:  # only "too low" so far: expand upward
            nxt = self._norm(low * self.factor)
            return nxt if nxt <= hi_b and nxt not in tried else None
        if low is None:  # only "too high" so far: expand downward
            nxt = self._norm(high / self.factor)
            return nxt if nxt >= lo_b and nxt not in tried else None
        # bracket exists: bisect on the readable grid
        if self.min_ratio is not None and high / low <= self.min_ratio * (1 + 1e-9):
            return None
        inside = [c for c in nice_lr_grid(low, high, self.resolution) if self._norm(c) not in tried]
        if not inside:
            return None
        target = math.sqrt(low * high)
        return min(inside, key=lambda c: abs(math.log(c) - math.log(target)))

    # -------------------------------------------------------------------- probe
    def _probe(self, lr: float) -> Probe:
        result = self.run(lr)
        state = self.classify(result)
        best = max(result.scores) if result.scores else None
        final = result.scores[-1] if result.scores else None
        probe = Probe(
            lr=lr, state=state, final_score=final, best_score=best,
            steps_run=result.steps[-1] if result.steps else None,
        )
        if state is RunState.CONVERGED and self.confirm is not None:
            confirmed = self.confirm(lr)
            cstate = self.classify(confirmed)
            probe.confirmed = cstate is RunState.CONVERGED
            if not probe.confirmed:
                probe.state = cstate
                probe.note = f"probe converged but the confirmation run was {cstate.value}"
                probe.final_score = confirmed.scores[-1] if confirmed.scores else None
                probe.best_score = max(confirmed.scores) if confirmed.scores else None
        logger.info(
            "lr_search: lr=%g -> %s (final=%s best=%s)%s",
            lr, probe.state.value, probe.final_score, probe.best_score,
            f" [{probe.note}]" if probe.note else "",
        )
        return probe

    # ------------------------------------------------------------------- window
    def _walk_window(self, first_lr: float, direction: int, budget: int) -> Tuple[List[float], int]:
        """Follow the grid from ``first_lr`` while neighbours keep converging; return (LRs, budget left)."""
        found: List[float] = []
        edge = first_lr
        lo_b, hi_b = self.lr_bounds
        while True:
            nxt = self._norm(grid_neighbor(edge, direction, self.resolution))
            if not lo_b <= nxt <= hi_b:
                break
            known = self._tried().get(nxt)
            if known is None:
                if budget <= 0:
                    break
                known = self._probe(nxt)
                self.probes.append(known)
                budget -= 1
            if known.state is not RunState.CONVERGED:
                break
            found.append(nxt)
            edge = nxt
        return found, budget

    def _converged_result(self, first_lr: float) -> LRSearchResult:
        convergent = [first_lr]
        if self.center_window:
            # each side gets its own budget: a walk that spends everything upward would return the centre of
            # a truncated window (found on Qwen-2B: 1e-6 .. 2e-5 probed only upward)
            budgets = {+1: (self.max_window_probes + 1) // 2, -1: self.max_window_probes // 2}
            for direction in (+1, -1):
                found, _ = self._walk_window(first_lr, direction, budgets[direction])
                convergent.extend(found)
        lo, hi = min(convergent), max(convergent)
        centre = 0.5 * (math.log(lo) + math.log(hi))
        # ties go to the lower LR: the upper edge of the window is where runs collapse
        best = min(convergent, key=lambda c: (round(abs(math.log(c) - centre), 9), c))
        return LRSearchResult("converged", best, list(self.probes), window=(lo, hi))

    def search(self) -> LRSearchResult:
        while len(self.probes) < self.max_probes:
            lr = self._next_lr()
            if lr is None:
                break
            probe = self._probe(lr)
            self.probes.append(probe)
            if probe.state is RunState.CONVERGED:
                return self._converged_result(probe.lr)
        low, high = self._bracket()
        bracket = (low, high) if low is not None and high is not None else None
        exhausted = len(self.probes) >= self.max_probes and self._next_lr() is not None
        status = "budget_exhausted" if exhausted else "no_convergent_lr"
        return LRSearchResult(status, None, list(self.probes), bracket)


# --------------------------------------------------------------------- trainer adapter
def tune_learning_rate(
    trainer: Any,
    initial_lr: float,
    *,
    experiment_dir: str,
    probe_steps: Optional[int] = 150,
    probe_eval_steps: Optional[int] = None,
    confirm: bool = True,
    early_stop: bool = True,
    max_probes: int = 12,
    resolution: str = "coarse",
    factor: float = 10.0,
    min_ratio: Optional[float] = None,
    lr_bounds: Tuple[float, float] = (1e-12, 1.0),
    center_window: bool = True,
    max_window_probes: int = 4,
    precheck: Optional[Callable[[], None]] = None,
    keep_probe_artifacts: bool = False,
    metadata: Optional[Mapping[str, Any]] = None,
    **train_overrides: Any,
) -> LRSearchResult:
    """Search the learning rate of ``trainer`` and write ``lr_search.json`` into ``experiment_dir``.

    Each probe trains one seed into ``<experiment_dir>/lr_search/<probe|confirm>_<lr>`` (no cache
    reuse), always from the trainer's original base model, with ``max_steps=probe_steps``. With
    ``confirm=True`` a converged probe is re-run at the trainer's configured ``max_steps`` before it
    is accepted. With ``early_stop=True`` a probe ends as soon as it is decisively collapsed or
    diverged (:class:`ProbeStopCallback`). ``precheck`` runs once before the first probe (use
    :func:`~gradiend.trainer.core.signal_checks.assert_signal_diverse` on stacked inputs) for extra checks.
    Probe checkpoints are deleted after each probe (only ``training.json`` is kept) unless
    ``keep_probe_artifacts=True``. ``metadata`` (e.g. the first guess and probe length) is stored next to the
    result in ``lr_search.json`` so a later run can tell whether a finished search still applies. The remaining arguments are those of
    :class:`LRSearch`.
    """
    os.makedirs(experiment_dir, exist_ok=True)
    if precheck is not None:
        precheck()

    base_args = getattr(trainer, "training_args", None) or getattr(trainer, "_training_args", None)
    full_steps = getattr(base_args, "max_steps", None)
    full_eval = getattr(base_args, "eval_steps", None)

    def _train_and_read(lr: float, steps: Optional[int], eval_steps: Optional[int], tag: str) -> ProbeResult:
        probe_dir = os.path.join(experiment_dir, "lr_search", f"{tag}_{lr:.3g}")
        overrides: Dict[str, Any] = dict(train_overrides)
        overrides.update(
            learning_rate=lr, experiment_dir=probe_dir, max_seeds=1, use_cache=False,
        )
        if steps is not None:
            overrides["max_steps"] = int(steps)
        if eval_steps is not None:
            overrides["eval_steps"] = int(eval_steps)
        # the diversity guard is free (it reads statistics every evaluation computes) and stays on
        overrides["callbacks"] = [SignalDiversityCallback()] + ([ProbeStopCallback()] if early_stop else [])
        with _isolated_probe(trainer):
            trainer.train(**overrides)
            result = _read_probe(probe_dir)
        if not keep_probe_artifacts:
            _prune_probe_dir(probe_dir)
        return result

    eval_steps = probe_eval_steps
    if eval_steps is None and probe_steps is not None:
        eval_steps = max(1, int(probe_steps) // 6)

    def run(lr: float) -> ProbeResult:
        return _train_and_read(lr, probe_steps, eval_steps, "probe")

    confirm_fn = None
    if confirm and probe_steps is not None and probe_steps != full_steps:
        def confirm_fn(lr: float) -> ProbeResult:  # noqa: E306
            return _train_and_read(lr, full_steps, full_eval, "confirm")

    search = LRSearch(
        run, initial_lr, factor=factor, min_ratio=min_ratio, max_probes=max_probes,
        resolution=resolution, lr_bounds=lr_bounds, confirm=confirm_fn,
        center_window=center_window, max_window_probes=max_window_probes,
    )
    result = search.search()
    payload = {**dict(metadata or {}), "initial_lr": float(initial_lr), **result.to_dict()}
    with open(os.path.join(experiment_dir, "lr_search.json"), "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    return result


@contextlib.contextmanager
def _isolated_probe(trainer: Any) -> Iterator[None]:
    """Make one probe an independent run.

    ``Trainer.train()`` leaves the trainer pointing at the checkpoint it just wrote
    (``_model_arg = out_path``) with that model resident on the GPU. Calling ``train()`` again
    would therefore continue from the previous probe's weights instead of the base model (and
    keep both models in memory). Each probe starts from the original base model and, afterwards,
    the trainer is handed back as it was found with the probe's model released.
    """
    if not hasattr(trainer, "_base_model_arg"):
        raise AttributeError("trainer has no _base_model_arg; cannot start each LR probe from the base model")
    saved_arg = trainer._model_arg
    trainer._model_arg = trainer._base_model_arg
    trainer._model_instance = None
    try:
        yield
    finally:
        trainer._model_arg = saved_arg
        trainer._model_instance = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # pragma: no cover - torch is a hard dependency; never mask the probe's own error
            pass


def _prune_probe_dir(probe_dir: str) -> None:
    """Keep only ``training.json`` of a probe: its checkpoint is as large as a real model."""
    for root, _dirs, files in os.walk(probe_dir):
        for name in files:
            if name != "training.json":
                try:
                    os.remove(os.path.join(root, name))
                except OSError:  # pragma: no cover - best effort, never masks the probe result
                    pass
    for root, dirs, files in os.walk(probe_dir, topdown=False):
        if root != probe_dir and not os.listdir(root):
            try:
                os.rmdir(root)
            except OSError:  # pragma: no cover
                pass


def _read_probe(probe_dir: str) -> ProbeResult:
    for root, _dirs, files in os.walk(probe_dir):
        if "training.json" in files:
            return ProbeResult.from_training_json(os.path.join(root, "training.json"))
    raise FileNotFoundError(f"no training.json under {probe_dir}")
