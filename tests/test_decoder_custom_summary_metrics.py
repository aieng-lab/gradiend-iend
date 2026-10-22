"""Explicit summary_metrics: a custom extractor defines its own metric names, the default extractor uses class names."""

from gradiend.evaluator.decoder import DecoderEvaluator, default_extract_candidates
from tests.test_decoder_strengthen_weaken_dataset import TrainerForStrengthenWeakenTest


def _evaluate(tmp_path, **kwargs):
    trainer = TrainerForStrengthenWeakenTest()
    trainer.experiment_dir = str(tmp_path)
    return DecoderEvaluator().evaluate_decoder(
        trainer, target_class="3SG", feature_factors=[-1.0], lrs=[1e-2], plot=False, use_cache=False, **kwargs
    )


def _with_derived_metric(results):
    """Like the gender example: add a derived metric to every grid entry, then use the default conversion."""
    derived = {cid: entry if cid == "base" else {**entry, "_derived": 0.5} for cid, entry in results.items()}
    return default_extract_candidates(derived)


def test_metrics_of_a_custom_extractor_are_summarized_as_requested(tmp_path):
    result = _evaluate(tmp_path, summary_extractor=_with_derived_metric, summary_metrics=["_derived"])
    assert result["_derived"]["value"] == 0.5
    assert "3SG" not in result


def test_default_extractor_still_limits_metrics_to_the_evaluated_classes(tmp_path):
    result = _evaluate(tmp_path, summary_metrics=["_derived"])
    assert "3SG" in result
    assert "_derived" not in result
