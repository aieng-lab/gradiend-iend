import copy
import pytest
from gradiend.trainer.text.common.loading import InstructTokenizerWrapper


class DummyTokenizer:
    def __init__(self):
        self.pad_token = "<pad>"
        self.mask_token = None
        self.custom_attr = 42

    def __call__(self, text, **kwargs):
        return {"input_ids": [1, 2, 3]}

    def tokenize(self, text, **kwargs):
        return ["hello", "world"]

    def convert_tokens_to_ids(self, tokens):
        return [1, 2]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        return ["hello", "world"]

    def decode(self, token_ids, **kwargs):
        return "hello world"


def test_instruct_tokenizer_wrapper_delegation():
    base = DummyTokenizer()
    wrapper = InstructTokenizerWrapper(base)
    assert wrapper.pad_token == "<pad>"
    assert wrapper.custom_attr == 42
    assert wrapper.tokenize("hello") == ["hello", "world"]
    assert wrapper.convert_tokens_to_ids(["hello"]) == [1, 2]
    assert wrapper.convert_ids_to_tokens([1]) == ["hello", "world"]
    assert wrapper.decode([1, 2]) == "hello world"


def test_instruct_tokenizer_wrapper_deepcopy_no_recursion():
    base = DummyTokenizer()
    wrapper = InstructTokenizerWrapper(base, user_prompt_header="u_hdr", assistant_prompt_header="a_hdr")
    
    # Check hasattr for standard special attributes without recursion
    assert not hasattr(wrapper, "__setstate__")
    assert not hasattr(wrapper, "__missing_special__")
    
    # Deepcopy must succeed and maintain wrapper invariants
    wrapper_copy = copy.deepcopy(wrapper)
    assert isinstance(wrapper_copy, InstructTokenizerWrapper)
    assert wrapper_copy.pad_token == "<pad>"
    assert wrapper_copy.user_prompt_header == "u_hdr"
    assert wrapper_copy.assistant_prompt_header == "a_hdr"
    assert wrapper_copy.tokenizer.custom_attr == 42
    
    # Shallow copy
    wrapper_shallow = copy.copy(wrapper)
    assert isinstance(wrapper_shallow, InstructTokenizerWrapper)
    assert wrapper_shallow.pad_token == "<pad>"
