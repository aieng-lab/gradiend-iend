"""Masked templates must carry the dataset placeholder, never a tokenizer special.

create_masked_pair_from_text has two branches. The decoder-only branch wrote the
literal "[MASK]"; the in-place (MLM) branch wrote tokenizer.mask_token. Both feed
the same dataframe, which validate_masked_templates_in_dataframe then checks for
the dataset placeholder -- so the second branch produced rows the validator
rejected. It stayed invisible because every model used so far is a causal LM
with no mask token, making the branches agree by accident.

Substituting the tokenizer's mask token is a tokenization-stage concern
(mask_placeholder_for_tokenizer) applied after the dataframe, not before it.
"""

from __future__ import annotations

import pytest

from gradiend.trainer.text.prediction.dataset import (
    DEFAULT_DATASET_MASK_PLACEHOLDER,
    create_masked_pair_from_text,
    validate_masked_templates_in_dataframe,
)


class _Tok:
    """Minimal whitespace tokenizer, optionally owning an MLM mask special."""

    def __init__(self, mask_token=None):
        self.mask_token = mask_token

    def tokenize(self, text):
        return str(text).split()

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)


TEXT = "the museum is still closed today but it reopens on monday for everyone"


class TestInPlaceBranch:
    """The branch that regressed: model with a real mask token."""

    def test_template_carries_the_dataset_placeholder(self):
        pair = create_masked_pair_from_text(
            TEXT, _Tok(mask_token="<mask>"), is_decoder_only_model=False,
            mask_token="<mask>",
        )
        assert pair is not None
        masked, _ = pair
        assert DEFAULT_DATASET_MASK_PLACEHOLDER in masked
        assert "<mask>" not in masked, "tokenizer special leaked into the template"

    def test_output_passes_the_schema_validator(self):
        """The regression, end to end: the builder fed its own validator."""
        pd = pytest.importorskip("pandas")
        rows = []
        for _ in range(8):
            pair = create_masked_pair_from_text(
                TEXT, _Tok(mask_token="<mask>"), is_decoder_only_model=False,
                mask_token="<mask>",
            )
            if pair:
                rows.append(pair[0])
        assert rows
        frame = pd.DataFrame({"masked": rows})
        validate_masked_templates_in_dataframe(
            frame, DEFAULT_DATASET_MASK_PLACEHOLDER, masked_col="masked"
        )

    def test_target_token_is_the_replaced_word(self):
        pair = create_masked_pair_from_text(
            TEXT, _Tok(mask_token="<mask>"), is_decoder_only_model=False,
            mask_token="<mask>",
        )
        masked, target = pair
        assert target in TEXT.split()
        assert target not in masked.split()


class TestDecoderOnlyBranchUnchanged:
    def test_still_emits_the_placeholder(self):
        pair = create_masked_pair_from_text(
            TEXT, _Tok(), is_decoder_only_model=True, mask_token=None
        )
        assert pair is not None
        assert pair[0].endswith(DEFAULT_DATASET_MASK_PLACEHOLDER)

    def test_a_mask_token_does_not_change_the_decoder_branch(self):
        pair = create_masked_pair_from_text(
            TEXT, _Tok(mask_token="<mask>"), is_decoder_only_model=True,
            mask_token="<mask>",
        )
        assert "<mask>" not in pair[0]
        assert DEFAULT_DATASET_MASK_PLACEHOLDER in pair[0]


class TestCustomPlaceholder:
    def test_both_branches_honour_an_explicit_placeholder(self):
        for decoder_only in (True, False):
            pair = create_masked_pair_from_text(
                TEXT, _Tok(mask_token="<mask>"),
                is_decoder_only_model=decoder_only,
                mask_token="<mask>", mask_placeholder="[PRONOUN]",
            )
            assert pair is not None
            assert "[PRONOUN]" in pair[0], decoder_only
            assert "<mask>" not in pair[0], decoder_only


class TestCallSitesUseTheCanonicalClassifier:
    """gradiend.model.utils.is_decoder_only_model calls itself the "single place
    of truth" and already handles Gemma-3's mask-token false negative. Two call
    sites bypassed it with an inline ``tokenizer.mask_token_id is None``, which
    routed Gemma-3 -- a causal LM -- down the masked-LM branch.
    """

    SITES = (
        "gradiend/trainer/text/prediction/trainer.py",
        "gradiend/comparison/cross_encoding.py",
    )

    def test_no_inline_mask_token_heuristic_remains(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for rel in self.SITES:
            text = (root / rel).read_text(encoding="utf-8")
            assert "else tokenizer.mask_token_id is None" not in text, rel

    def test_both_sites_call_the_canonical_classifier(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        for rel in self.SITES:
            text = (root / rel).read_text(encoding="utf-8")
            assert "_is_decoder_only(" in text, rel

    def test_classifier_keeps_gemma3_causal(self):
        from gradiend.model.utils import is_decoder_only_model

        gemma = type("Gemma3ForCausalLM", (object,), {"mask_token": "<mask>"})()
        gemma.config = type(
            "C", (object,), {"model_type": "gemma3",
                             "architectures": ["Gemma3ForCausalLM"]}
        )()
        assert is_decoder_only_model(gemma) is True

    def test_classifier_keeps_a_real_masked_lm_masked(self):
        from gradiend.model.utils import is_decoder_only_model

        bert = type("BertForMaskedLM", (object,), {"mask_token": "[MASK]"})()
        assert is_decoder_only_model(bert) is False


class TestTruncationAnchorsOnFirstMask:
    """With more than one placeholder, anchor on the first, not the last.

    Anchoring on the last leaves earlier placeholders in the prefix as literal
    "[MASK]" text that the model attends to as context. Truncating after the
    first yields a prefix with no stray placeholder. Single-placeholder
    templates are unaffected.
    """

    @staticmethod
    def _truncate(template, max_length=64):
        from gradiend.trainer.text.prediction.dataset import (
            _left_truncate_template_keeping_mask,
        )

        class _Tok:
            def __call__(self, text, **kw):
                return {"input_ids": str(text).split()}

            def tokenize(self, text):
                return str(text).split()

            def convert_tokens_to_string(self, tokens):
                return " ".join(tokens)

        return _left_truncate_template_keeping_mask(
            _Tok(), template, mask_placeholder="[MASK]", max_length=max_length
        )

    def test_second_placeholder_is_cut_off(self):
        out = self._truncate("alpha [MASK] beta [MASK] gamma")
        assert out.count("[MASK]") == 1
        assert out.endswith("[MASK]")
        assert "beta" not in out

    def test_prefix_before_the_first_mask_is_kept(self):
        out = self._truncate("alpha [MASK] beta [MASK] gamma")
        assert out.startswith("alpha")

    def test_single_placeholder_is_unchanged_in_behaviour(self):
        out = self._truncate("alpha beta [MASK] gamma delta")
        assert out == "alpha beta [MASK]"

    def test_template_without_a_placeholder_is_returned_as_is(self):
        assert self._truncate("alpha beta gamma") == "alpha beta gamma"

    def test_source_no_longer_anchors_on_the_last_mask(self):
        from pathlib import Path

        import gradiend.trainer.text.prediction.dataset as ds

        text = Path(ds.__file__).read_text(encoding="utf-8")
        block = text[text.index("def _left_truncate_template_keeping_mask") :][:1600]
        assert "rfind" not in block
        assert "template.find(mask_placeholder)" in block
