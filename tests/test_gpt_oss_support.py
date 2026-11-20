"""gpt-oss: layers-only topology on the fused-expert MoE and MXFP4 dequantize-on-load kwargs."""
from types import SimpleNamespace

import pytest

transformers = pytest.importorskip("transformers")


def _tiny_gpt_oss():
    from transformers import GptOssConfig, GptOssForCausalLM

    cfg = GptOssConfig(
        vocab_size=64, hidden_size=32, intermediate_size=32, num_hidden_layers=2, num_attention_heads=4,
        num_key_value_heads=2, head_dim=8, num_local_experts=4, num_experts_per_tok=2, layer_types=["full_attention"] * 2,
    )
    return GptOssForCausalLM(cfg)


def test_gpt_oss_topology_resolves_layers_only_scope():
    from gradiend.model_topology import infer_model_topology

    topo = infer_model_topology(_tiny_gpt_oss())
    assert topo is not None
    assert topo.layers == ("model.layers.0", "model.layers.1")
    assert topo.word_embedding == "model.embed_tokens"
    assert "lm_head" in topo.prediction_heads


def test_dequantize_kwargs_only_for_mxfp4(monkeypatch):
    import transformers as tf
    from gradiend.model import utils

    def fake(quant):
        return lambda *a, **k: SimpleNamespace(quantization_config=quant)

    monkeypatch.setattr(tf.AutoConfig, "from_pretrained", fake({"quant_method": "mxfp4"}))
    kw = utils.dequantize_load_kwargs("openai/gpt-oss-20b")
    assert kw["quantization_config"].dequantize is True
    monkeypatch.setattr(tf.AutoConfig, "from_pretrained", fake(None))
    assert utils.dequantize_load_kwargs("gpt2") == {}
    monkeypatch.setattr(tf.AutoConfig, "from_pretrained", fake({"quant_method": "bitsandbytes"}))
    assert utils.dequantize_load_kwargs("x") == {}


def test_mxfp4_load_gets_a_torch_accelerator_shim_on_old_torch(monkeypatch):
    """torch 2.5 (cluster image) has no torch.accelerator, which transformers' MXFP4 quantizer calls first."""
    import torch
    import transformers as tf
    from gradiend.model import utils

    monkeypatch.delattr(torch, "accelerator", raising=False)
    monkeypatch.setattr(tf.AutoConfig, "from_pretrained",
                        lambda *a, **k: SimpleNamespace(quantization_config={"quant_method": "mxfp4"}))
    utils.dequantize_load_kwargs("openai/gpt-oss-20b")
    dev = torch.accelerator.current_accelerator()
    assert dev is None or dev.type == "cuda"
    monkeypatch.delattr(torch, "accelerator", raising=False)
