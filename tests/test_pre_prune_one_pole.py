"""Pre-prune must work for one-pole training.

One-pole requests the task's full class list while its frame carries only the
trained pole's ids, so the requested classes matched nothing and stratification
raised. Callers worked around that by disabling pre-prune entirely -- which on a
multi-billion-parameter model means the encoder is built at FULL width
(lazy_init is gated on pre_prune_config being set), i.e. 26 GiB at 8B instead of
the ~280 MB a pruned encoder needs. That is why GRADIEND could not run the
`none` split at 8B.

Stratification here exists only to draw a representative gradient sample, so the
classes actually present serve that purpose.
"""

from __future__ import annotations

import pytest

from gradiend.trainer.core.pruning import _stratified_indices


class _DS:
    """Minimal dataset exposing a feature-class key per item."""

    def __init__(self, classes):
        self._classes = list(classes)

    def __len__(self):
        return len(self._classes)

    def __getitem__(self, idx):
        return {"factual_id": self._classes[idx], "x": idx}


def _indices(dataset_classes, requested, n_samples=8, seed=0):
    return _stratified_indices(
        _DS(dataset_classes),
        n_samples=n_samples,
        feature_class_key="factual_id",
        target_feature_class_ids=requested,
        seed=seed,
    )


class TestOnePoleFallback:
    def test_no_requested_class_present_falls_back(self):
        """The regression: one-pole frame has only 'F', task asks for F and M."""
        out = _indices(["F"] * 10, requested=["X", "Y"])
        assert len(out) == 8
        assert all(0 <= i < 10 for i in out)

    def test_fallback_warns_naming_both_sets(self, caplog):
        import logging

        with caplog.at_level(logging.WARNING):
            _indices(["F"] * 10, requested=["X", "Y"])
        text = caplog.text
        assert "pre_prune" in text
        assert "X" in text and "F" in text

    def test_partial_match_still_uses_only_matching_classes(self):
        """A genuine partial match must not trigger the fallback."""
        out = _indices(["F"] * 6 + ["M"] * 6, requested=["F", "ZZZ"], n_samples=4)
        assert len(out) == 4
        assert all(i < 6 for i in out), "should draw only from the F block"

    def test_normal_multiclass_is_unchanged(self):
        out = _indices(["a"] * 5 + ["b"] * 5, requested=["a", "b"], n_samples=6)
        assert len(out) == 6
        assert any(i < 5 for i in out) and any(i >= 5 for i in out)


class TestStillRaisesWhenItShould:
    def test_empty_dataset_raises(self):
        """Caught earlier, with its own clearer message; the fallback must not
        swallow it."""
        with pytest.raises(ValueError, match="[Dd]ataset is empty"):
            _indices([], requested=["a"], n_samples=4)

    def test_missing_key_still_raises(self):
        class _Bad:
            def __len__(self):
                return 3

            def __getitem__(self, idx):
                return {"other": 1}

        with pytest.raises(KeyError, match="factual_id"):
            _stratified_indices(
                _Bad(), n_samples=2, feature_class_key="factual_id",
                target_feature_class_ids=["a"], seed=0,
            )


class TestScaleConsequence:
    def test_lazy_init_is_gated_on_pre_prune_config(self):
        """Why disabling pre-prune is catastrophic at scale, pinned in a test."""
        from pathlib import Path

        import gradiend.model.model_with_gradiend as mwg

        src = Path(mwg.__file__).read_text(encoding="utf-8")
        assert 'lazy_init = bool(create_kwargs.get("pre_prune_config") is not None)' in src
