"""gpt-oss (harmony format): the generic template chat format renders its prompt, no dedicated format needed.

Uses the real tokenizer of ``openai/gpt-oss-20b`` (a 27 MB download, no weights); skipped when the Hub is not reachable.
"""
import pytest

transformers = pytest.importorskip("transformers")

from gradiend.trainer.text.common.chat_format import (
    TemplateChatFormat,
    chat_format_for_tokenizer,
    uses_chat_format,
)
from gradiend.trainer.text.common.loading import InstructTokenizerWrapper

GPT_OSS = "openai/gpt-oss-20b"
SYSTEM = "Predict the next word."


@pytest.fixture(scope="module")
def gpt_oss_tokenizer():
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(GPT_OSS)
    except Exception as error:  # noqa: BLE001 - offline machine or unreachable Hub
        pytest.skip(f"tokenizer of {GPT_OSS} not available: {type(error).__name__}")


def test_the_generic_template_format_handles_gpt_oss(gpt_oss_tokenizer):
    chat_format = chat_format_for_tokenizer(gpt_oss_tokenizer)
    assert type(chat_format) is TemplateChatFormat
    prefix, suffix = chat_format.frame(SYSTEM)
    assert prefix.startswith("<|start|>system<|message|>")
    assert prefix.endswith("<|start|>user<|message|>")
    assert SYSTEM in prefix
    # the generation prompt opens the assistant turn; the model's answer follows it
    assert suffix == "<|end|><|start|>assistant"


def test_the_prompt_carries_the_user_text_verbatim_and_a_fixed_date(gpt_oss_tokenizer):
    chat_format = chat_format_for_tokenizer(gpt_oss_tokenizer)
    chat_format.configure("2025-01-02")
    prompt = chat_format.render(SYSTEM, "The nurse said that")
    assert chat_format.inserts_date  # the harmony template writes the current date
    assert "Current date: 2025-01-02" in prompt
    assert prompt.endswith("<|start|>user<|message|>The nurse said that<|end|><|start|>assistant")
    # the same prompt on another day: the format never reads the system clock
    assert chat_format.render(SYSTEM, "The nurse said that") == prompt


def test_the_wrapper_tokenizes_the_harmony_prompt(gpt_oss_tokenizer):
    wrapper = InstructTokenizerWrapper(gpt_oss_tokenizer)
    assert uses_chat_format(wrapper)
    ids = wrapper("The nurse said that")["input_ids"]
    expected = gpt_oss_tokenizer(wrapper.chat_format.render(wrapper.system_prompt, "The nurse said that"), add_special_tokens=False)["input_ids"]
    assert list(ids) == list(expected)
    assert gpt_oss_tokenizer.convert_tokens_to_ids("<|start|>") in ids
