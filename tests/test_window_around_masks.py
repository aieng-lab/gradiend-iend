"""A long masked-LM row keeps its mask: a window around it replaces the naive truncation that cut the mask off."""

import re

from gradiend.trainer.text.prediction.dataset import _window_around_masks


class _WordTokenizer:
    """Whitespace tokenizer with character offsets (no download)."""

    mask_token = "[MASK]"

    def __init__(self, offsets=True):
        self.vocab = {"[MASK]": 1}
        self.offsets = offsets

    def num_special_tokens_to_add(self, pair=False):
        return 2

    def convert_tokens_to_ids(self, token):
        return self.vocab.get(token, 0)

    def __call__(self, text, add_special_tokens=True, truncation=False, return_offsets_mapping=False, **kwargs):
        spans = [(m.group(), m.start(), m.end()) for m in re.finditer(r"\S+", text)]
        encoded = {"input_ids": [self.vocab.setdefault(word, len(self.vocab) + 1) for word, _, _ in spans]}
        if return_offsets_mapping:
            if not self.offsets:
                raise NotImplementedError("slow tokenizer")
            encoded["offset_mapping"] = [(start, end) for _, start, end in spans]
        return encoded


def _text(mask_at, n=40):
    words = [f"w{i}" for i in range(n)]
    words[mask_at] = "[MASK]"
    return " ".join(words)


def _window(text, tokenizer=None, max_length=10):
    return _window_around_masks(tokenizer or _WordTokenizer(), text, mask_token="[MASK]", max_length=max_length)


def test_the_mask_stays_with_context_on_both_sides():
    result = _window(_text(25))
    words = result.split()
    assert "[MASK]" in words
    assert len(words) == 8  # max_length 10 minus two special tokens
    assert 0 < words.index("[MASK]") < len(words) - 1


def test_text_that_fits_is_unchanged():
    short = _text(3, n=6)
    assert _window(short) == short


def test_a_mask_near_the_start_spends_the_room_on_the_right():
    words = _window(_text(1)).split()
    assert words[:2] == ["w0", "[MASK]"] and len(words) == 8


def test_a_mask_near_the_end_spends_the_room_on_the_left():
    words = _window(_text(38)).split()
    assert words[-2:] == ["[MASK]", "w39"] and len(words) == 8


def test_all_masks_of_a_span_are_kept():
    words = [f"w{i}" for i in range(40)]
    words[20:23] = ["[MASK]"] * 3
    result = _window(" ".join(words)).split()
    assert result.count("[MASK]") == 3 and len(result) == 8


def test_tokenizer_without_offsets_returns_the_text_unchanged():
    text = _text(25)
    assert _window(text, _WordTokenizer(offsets=False)) == text


def test_text_without_a_mask_is_unchanged():
    text = " ".join(f"w{i}" for i in range(40))
    assert _window(text) == text
