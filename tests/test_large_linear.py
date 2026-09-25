import torch

from gradiend.model.layers import LargeLinear


def test_forward_casts_mismatched_input_dtype_like_it_already_casts_device():
    """AGIEND's decoder is a LargeLinear built at the default float32 dtype.

    Steering a bfloat16 base model (e.g. llama-3.1-8b, correctly loaded in
    bf16) feeds bf16 activations into it. forward() already reconciles a
    device mismatch between input and weight; it must do the same for dtype,
    or torch's matmul raises "mat1 and mat2 must have the same dtype".
    """
    layer = LargeLinear(8, 4, dtype=torch.float32, device="cpu")
    bf16_input = torch.randn(2, 8, dtype=torch.bfloat16)

    output = layer(bf16_input)

    assert output.dtype == torch.float32
    assert output.shape == (2, 4)


def test_forward_is_a_noop_dtype_cast_when_input_already_matches():
    layer = LargeLinear(8, 4, dtype=torch.float32, device="cpu")
    input_ = torch.randn(2, 8, dtype=torch.float32)

    output = layer(input_)

    assert output.dtype == torch.float32
    assert output.shape == (2, 4)
