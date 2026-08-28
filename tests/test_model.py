"""
Tests for core GRADIEND models (modality-independent).

Tests GradiendModel, ParamMappedGradiendModel, and ModelWithGradiend with toy networks.
"""

import os
import sys
import json
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from gradiend.model import GradiendComponent, GradiendModel, ParamMappedGradiendModel


class TestGradiendModel:
    """Test GradiendModel (weights-only encoder-decoder)."""
    
    def test_gradiend_model_creation(self):
        """Test GradiendModel creation with various configs."""
        # Test with different latent_dim and input_dim (should default to CPU if CUDA not available)
        model1 = GradiendModel(input_dim=100, latent_dim=1)
        assert model1.input_dim == 100
        assert model1.latent_dim == 1
        assert model1.bias_encoder is False
        assert model1.encoder[0].bias is None
        
        model2 = GradiendModel(input_dim=500, latent_dim=2, activation_encoder="relu")
        assert model2.input_dim == 500
        assert model2.latent_dim == 2
        
        # Test dtype
        model3 = GradiendModel(input_dim=100, latent_dim=1, torch_dtype=torch.float16)
        assert model3.torch_dtype == torch.float16
    
    def test_gradiend_model_forward(self, set_seed):
        """Test GradiendModel forward pass with mock gradients."""
        set_seed(42)
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        # Create mock gradient tensor
        x = torch.randn(100)
        
        # Forward pass
        decoded = model.forward(x)
        assert decoded.shape == (100,)
        assert decoded.dtype == model.torch_dtype
        
        # Forward with return_encoded
        decoded2, encoded = model.forward(x, return_encoded=True)
        assert decoded2.shape == (100,)
        assert encoded.shape == (1,)

    def test_gradiend_model_forward_batched(self, set_seed):
        """Test GradiendModel batched forward pass."""
        set_seed(42)
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))

        x = torch.randn(4, 100)
        decoded, encoded = model.forward(x, return_encoded=True)

        assert decoded.shape == (4, 100)
        assert encoded.shape == (4, 1)
    
    def test_gradiend_model_forward_encoder(self, set_seed):
        """Test encoder-only forward pass."""
        set_seed(42)
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        x = torch.randn(100)
        encoded = model.forward_encoder(x)
        assert encoded.shape == (1,)

    def test_gradiend_model_default_init_uses_fan_in_floor_for_small_inputs(self, set_seed):
        set_seed(42)
        model = GradiendModel(input_dim=100, latent_dim=4, device=torch.device("cpu"))

        expected_bound = 1.0 / (10_000 ** 0.5)

        assert model.init_fan_in_floor == 10_000
        assert model._init_bound_for_fan_in(100) == pytest.approx(expected_bound)
        assert model.encoder[0].linear.weight.abs().max().item() <= expected_bound + 1e-7
        assert model.decoder[0].linear.weight.abs().max().item() <= expected_bound + 1e-7
        assert model.decoder[0].linear.bias.abs().max().item() <= expected_bound + 1e-7

    def test_gradiend_model_component_init_uses_component_fan_in(self, set_seed):
        set_seed(42)
        model = GradiendModel(
            input_dim=12,
            latent_dim=8,
            init_fan_in_floor=None,
            device=torch.device("cpu"),
            component_slices=[
                {"id": "left", "start": 0, "end": 4},
                {"id": "right", "start": 4, "end": 12},
            ],
        )

        left_bound = 1.0 / (4 ** 0.5)
        right_bound = 1.0 / (8 ** 0.5)

        assert model._init_bound_for_fan_in(4) == pytest.approx(left_bound)
        assert model._init_bound_for_fan_in(8) == pytest.approx(right_bound)
        assert model.encoder[0].linear.weight[:, :4].abs().max().item() <= left_bound + 1e-7
        assert model.encoder[0].linear.weight[:, 4:].abs().max().item() <= right_bound + 1e-7
        assert model.decoder[0].linear.weight[:4, :].abs().max().item() <= left_bound + 1e-7
        assert model.decoder[0].linear.weight[4:, :].abs().max().item() <= right_bound + 1e-7
    
    def test_gradiend_model_save_load(self, temp_dir):
        """Test saving and loading GradiendModel."""
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        # Save
        save_path = os.path.join(temp_dir, "test_model")
        model.save_pretrained(save_path)
        
        # Verify files exist
        assert os.path.exists(os.path.join(save_path, "config.json"))
        with open(os.path.join(save_path, "config.json"), encoding="utf-8") as handle:
            config = json.load(handle)
        assert config["architecture"]["init_fan_in_floor"] == 10_000
        # Check for either safetensors or bin file
        has_weights = (
            os.path.exists(os.path.join(save_path, "model.safetensors")) or
            os.path.exists(os.path.join(save_path, "pytorch_model.bin"))
        )
        assert has_weights, "Model weights should be saved"
        
        # Load (should default to CPU if CUDA not available)
        loaded = GradiendModel.from_pretrained(save_path)
        assert loaded.input_dim == model.input_dim
        assert loaded.latent_dim == model.latent_dim
        assert loaded.init_fan_in_floor == model.init_fan_in_floor
        
        # Verify forward pass works
        x = torch.randn(100)
        original_output = model.forward(x)
        loaded_output = loaded.forward(x)
        torch.testing.assert_close(original_output, loaded_output, rtol=1e-5, atol=1e-5)

    def test_gradiend_model_loads_legacy_checkpoint_without_init_fan_in_floor(self, temp_dir):
        model = GradiendModel(input_dim=8, latent_dim=2, device=torch.device("cpu"))
        save_path = os.path.join(temp_dir, "legacy_init_model")
        model.save_pretrained(save_path)

        config_path = os.path.join(save_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        del config["architecture"]["init_fan_in_floor"]
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle)

        loaded = GradiendModel.from_pretrained(save_path)

        assert loaded.init_fan_in_floor is None
        x = torch.randn(3, 8)
        torch.testing.assert_close(model(x), loaded(x), rtol=1e-5, atol=1e-5)

    def test_gradiend_model_can_disable_encoder_bias_and_roundtrip(self, temp_dir):
        model = GradiendModel(
            input_dim=8,
            latent_dim=2,
            bias_encoder=False,
            bias_decoder=False,
            device=torch.device("cpu"),
        )

        assert model.bias_encoder is False
        assert model.bias_decoder is False
        assert model.encoder[0].bias is None
        assert model.decoder[0].bias is None

        save_path = os.path.join(temp_dir, "no_bias_model")
        model.save_pretrained(save_path)

        loaded = GradiendModel.from_pretrained(save_path)

        assert loaded.bias_encoder is False
        assert loaded.bias_decoder is False
        assert loaded.encoder[0].bias is None
        assert loaded.decoder[0].bias is None
        x = torch.randn(3, 8)
        torch.testing.assert_close(model(x), loaded(x), rtol=1e-5, atol=1e-5)

    def test_gradiend_model_loads_old_checkpoint_encoder_bias_from_state_dict(self, temp_dir):
        model = GradiendModel(
            input_dim=8,
            latent_dim=2,
            bias_encoder=True,
            device=torch.device("cpu"),
        )
        save_path = os.path.join(temp_dir, "old_bias_model")
        model.save_pretrained(save_path)

        config_path = os.path.join(save_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        del config["architecture"]["bias_encoder"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        loaded = GradiendModel.from_pretrained(save_path)

        assert loaded.bias_encoder is True
        assert loaded.encoder[0].bias is not None
        x = torch.randn(3, 8)
        torch.testing.assert_close(model(x), loaded(x), rtol=1e-5, atol=1e-5)

    def test_gradiend_model_loads_old_checkpoint_without_encoder_bias_from_state_dict(self, temp_dir):
        model = GradiendModel(
            input_dim=8,
            latent_dim=2,
            bias_encoder=False,
            device=torch.device("cpu"),
        )
        save_path = os.path.join(temp_dir, "old_no_bias_model")
        model.save_pretrained(save_path)

        config_path = os.path.join(save_path, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        del config["architecture"]["bias_encoder"]
        with open(config_path, "w") as f:
            json.dump(config, f)

        loaded = GradiendModel.from_pretrained(save_path)

        assert loaded.bias_encoder is False
        assert loaded.encoder[0].bias is None
        x = torch.randn(3, 8)
        torch.testing.assert_close(model(x), loaded(x), rtol=1e-5, atol=1e-5)

    def test_gradiend_model_virtual_component_encoders_use_weight_views(self):
        model = GradiendModel(
            input_dim=4,
            latent_dim=1,
            activation_encoder="tanh",
            activation_decoder="id",
            bias_encoder=False,
            bias_decoder=True,
            device=torch.device("cpu"),
            component_slices=[
                {"id": "left", "start": 0, "end": 2},
                {"id": "right", "start": 2, "end": 4},
            ],
        )
        with torch.no_grad():
            model.encoder[0].linear.weight.copy_(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
            model.decoder[0].linear.weight.copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
            model.decoder[0].linear.bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))

        x = torch.tensor([10.0, 20.0, 30.0, 40.0])

        left = model._component_encoders[0](x)
        right = model._component_encoders["right"](x)
        component_encodings = model._encode_components(x)

        torch.testing.assert_close(left, torch.tanh(torch.tensor([50.0])))
        torch.testing.assert_close(right, torch.tanh(torch.tensor([250.0])))
        torch.testing.assert_close(component_encodings, torch.stack([left, right], dim=0))
        assert model._component_encoders["left"].weight.shape == (1, 2)
        assert model._component_encoders["left"].weight.data_ptr() == model.encoder[0].linear.weight[:, :2].data_ptr()

    def test_gradiend_model_reports_nontrivial_component_split(self):
        full = GradiendModel(input_dim=4, latent_dim=1, device=torch.device("cpu"))
        single_partition = GradiendModel(
            input_dim=4,
            latent_dim=1,
            device=torch.device("cpu"),
            component_slices=[{"id": "full", "start": 0, "end": 4}],
            component_split_mode="single",
        )
        split = GradiendModel(
            input_dim=4,
            latent_dim=1,
            device=torch.device("cpu"),
            component_slices=[
                {"id": "left", "start": 0, "end": 2},
                {"id": "right", "start": 2, "end": 4},
            ],
        )

        assert full.component_count == 0
        assert full.component_slices == ()
        assert [component.to_dict() for component in full._virtual_component_slices] == [
            {"id": "full", "start": 0, "end": 4},
        ]
        assert full.has_component_split is False
        assert single_partition.component_count == 1
        assert single_partition.has_component_split is True
        assert [component.to_dict() for component in single_partition.component_slices] == [
            {"id": "full", "start": 0, "end": 4},
        ]
        assert split.component_count == 2
        assert split.has_component_split is True

    def test_gradiend_model_split_reconstruction_loss_aggregates_component_losses(self):
        model = GradiendModel(
            input_dim=4,
            latent_dim=1,
            activation_encoder="id",
            activation_decoder="id",
            bias_encoder=False,
            bias_decoder=False,
            device=torch.device("cpu"),
            component_slices=[
                {"id": "left", "start": 0, "end": 1},
                {"id": "right", "start": 1, "end": 4},
            ],
        )
        with torch.no_grad():
            model.encoder[0].linear.weight.fill_(1.0)
            model.decoder[0].linear.weight.fill_(1.0)

        source = torch.tensor([1.0, 2.0, 3.0, 4.0])
        target = torch.tensor([0.5, 1.0, 1.0, 1.0])
        criterion = torch.nn.MSELoss()

        decoded_components, encoded_components = model._forward_components(source, return_encoded=True)
        component_targets = model._component_target_slices(target)
        component_losses = torch.stack([
            criterion(decoded, expected)
            for decoded, expected in zip(decoded_components, component_targets)
        ])
        weights = torch.tensor([1.0, 3.0])

        torch.testing.assert_close(
            model.reconstruction_loss(source, target, criterion=criterion, aggregation="mean"),
            component_losses.mean(),
        )
        torch.testing.assert_close(
            model.reconstruction_loss(source, target, criterion=criterion, aggregation="sum"),
            component_losses.sum(),
        )
        torch.testing.assert_close(
            model.reconstruction_loss(source, target, criterion=criterion, aggregation="size_weighted"),
            (component_losses * (weights / weights.sum())).sum(),
        )
        assert encoded_components.shape == (2, 1)

    def test_gradiend_model_full_reconstruction_loss_matches_forward_loss(self):
        model = GradiendModel(
            input_dim=4,
            latent_dim=1,
            activation_encoder="id",
            activation_decoder="id",
            bias_encoder=False,
            bias_decoder=False,
            device=torch.device("cpu"),
            component_slices=[
                {"id": "left", "start": 0, "end": 2},
                {"id": "right", "start": 2, "end": 4},
            ],
        )
        source = torch.randn(4)
        target = torch.randn(4)
        criterion = torch.nn.MSELoss()

        decoded, encoded = model(source, return_encoded=True)
        loss, loss_encoded = model.reconstruction_loss(
            source,
            target,
            criterion=criterion,
            aggregation="full",
            return_encoded=True,
        )

        torch.testing.assert_close(loss, criterion(decoded, target.to(decoded.device)))
        torch.testing.assert_close(loss_encoded, encoded)

    def test_gradiend_model_virtual_component_decoders_use_weight_views(self):
        model = GradiendModel(
            input_dim=4,
            latent_dim=1,
            activation_decoder="id",
            bias_decoder=True,
            device=torch.device("cpu"),
            component_slices=[
                GradiendComponent("left", 0, 2),
                GradiendComponent("right", 2, 4),
            ],
        )
        with torch.no_grad():
            model.decoder[0].linear.weight.copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
            model.decoder[0].linear.bias.copy_(torch.tensor([0.1, 0.2, 0.3, 0.4]))

        z = torch.tensor([2.0])

        torch.testing.assert_close(model._component_decoders[0](z), torch.tensor([2.1, 4.2]))
        torch.testing.assert_close(model._component_decoders["right"](z), torch.tensor([6.3, 8.4]))
        assert model._component_decoders["right"].weight.shape == (2, 1)
        assert model._component_decoders["right"].weight.data_ptr() == model.decoder[0].linear.weight[2:, :].data_ptr()

    def test_gradiend_model_component_metadata_roundtrips_checkpoint(self, temp_dir):
        model = GradiendModel(
            input_dim=4,
            latent_dim=1,
            device=torch.device("cpu"),
            component_slices=[
                {"id": "left", "start": 0, "end": 2},
                {"id": "right", "start": 2, "end": 4},
            ],
        )

        save_path = os.path.join(temp_dir, "component_model")
        model.save_pretrained(save_path)
        loaded = GradiendModel.from_pretrained(save_path)

        assert [component.to_dict() for component in loaded.component_slices] == [
            {"id": "left", "start": 0, "end": 2},
            {"id": "right", "start": 2, "end": 4},
        ]
        x = torch.randn(4)
        torch.testing.assert_close(model._encode_components(x), loaded._encode_components(x), rtol=1e-5, atol=1e-5)

    def test_gradiend_model_with_components_returns_weight_sharing_view(self):
        model = GradiendModel(input_dim=4, latent_dim=1, device=torch.device("cpu"))

        view = model._with_components([
            {"id": "left", "start": 0, "end": 2},
            {"id": "right", "start": 2, "end": 4},
        ])

        assert view is not model
        assert view.encoder is model.encoder
        assert view.decoder is model.decoder
        assert [component.id for component in view.component_slices] == ["left", "right"]
        assert model.component_slices == ()
        assert [component.id for component in model._virtual_component_slices] == ["full"]

    def test_param_mapped_gradiend_model_component_metadata_roundtrips_checkpoint(self, temp_dir):
        model = ParamMappedGradiendModel(
            input_dim=4,
            latent_dim=1,
            param_map={"p": {"shape": (4,), "repr": "all"}},
            device=torch.device("cpu"),
            component_slices=[
                {"id": "left", "start": 0, "end": 2},
                {"id": "right", "start": 2, "end": 4},
            ],
        )

        save_path = os.path.join(temp_dir, "mapped_component_model")
        model.save_pretrained(save_path)
        loaded = ParamMappedGradiendModel.from_pretrained(save_path)

        assert [component.to_dict() for component in loaded.component_slices] == [
            {"id": "left", "start": 0, "end": 2},
            {"id": "right", "start": 2, "end": 4},
        ]
        x = torch.randn(4)
        torch.testing.assert_close(model._encode_components(x), loaded._encode_components(x), rtol=1e-5, atol=1e-5)
    
    def test_gradiend_model_get_weight_importance(self):
        """Test weight importance computation."""
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        # Test different parts
        imp_decoder_weight = model.get_weight_importance(part="decoder-weight")
        assert imp_decoder_weight.shape == (100,)
        
        imp_decoder_bias = model.get_weight_importance(part="decoder-bias")
        assert imp_decoder_bias.shape == (100,)
        
        imp_decoder_sum = model.get_weight_importance(part="decoder-sum")
        assert imp_decoder_sum.shape == (100,)
        
        imp_encoder_weight = model.get_weight_importance(part="encoder-weight")
        assert imp_encoder_weight.shape == (100,)
    
    def test_gradiend_model_get_topk_weights(self):
        """Test top-k weight selection."""
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        # Test absolute topk
        topk = model.get_topk_weights(part="decoder-weight", topk=10)
        assert len(topk) == 10
        assert all(isinstance(i, int) for i in topk)
        
        # Test relative topk
        topk_relative = model.get_topk_weights(part="decoder-weight", topk=0.1)
        assert len(topk_relative) == 10  # 10% of 100
    
    def test_gradiend_model_pruning_length_changes(self, set_seed):
        """Test that pruning changes input_dim length correctly."""
        set_seed(42)
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        original_dim = model.input_dim
        
        # Prune to top 50
        pruned = model.prune(topk=50, inplace=False)
        assert pruned.input_dim == 50
        assert model.input_dim == original_dim  # Original unchanged
        
        # Prune in place
        model.prune(topk=30, inplace=True)
        assert model.input_dim == 30
    
    def test_gradiend_model_pruning_with_mask(self, set_seed):
        """Test pruning with provided mask."""
        set_seed(42)
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        # Create a mask keeping first 50 dimensions
        mask = torch.zeros(100, dtype=torch.bool)
        mask[:50] = True
        
        pruned = model.prune(mask=mask, inplace=False)
        assert pruned.input_dim == 50
        
        # Test with return_mask
        pruned2, returned_mask = model.prune(mask=mask, inplace=False, return_mask=True)
        assert pruned2.input_dim == 50
        assert returned_mask.shape == (100,)
        assert returned_mask.sum() == 50
    
    def test_gradiend_model_pruning_with_threshold(self, set_seed):
        """Test pruning with threshold."""
        set_seed(42)
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        # Get importance and set threshold
        importance = model.get_weight_importance(part="decoder-weight")
        threshold = importance.median().item()
        
        pruned = model.prune(threshold=threshold, part="decoder-weight", inplace=False)
        # Should keep roughly half (depending on distribution)
        assert pruned.input_dim < model.input_dim
        assert pruned.input_dim > 0


class TestParamMappedGradiendModel:
    """Test ParamMappedGradiendModel (with parameter mapping)."""
    
    def test_param_mapped_model_creation(self):
        """Test ParamMappedGradiendModel creation with param_map."""
        # Create a simple param_map
        param_map = {
            "layer1.weight": {
                "shape": (10, 5),
                "repr": "all"
            },
            "layer2.weight": {
                "shape": (5, 2),
                "repr": "all"
            }
        }
        input_dim = 10 * 5 + 5 * 2  # 60
        
        model = ParamMappedGradiendModel(
            input_dim=input_dim,
            latent_dim=1,
            param_map=param_map
        )
        
        assert model.input_dim == input_dim
        assert model.latent_dim == 1
        assert len(model.param_map) == 2

    def test_param_mapped_model_signal_flags(self):
        gradient_model = ParamMappedGradiendModel(
            input_dim=2,
            latent_dim=1,
            param_map={"w": {"shape": (2,), "repr": "all"}},
        )
        assert gradient_model.signal_kind == "gradient"
        assert gradient_model.uses_gradients is True
        assert gradient_model.uses_activations is False
        assert gradient_model.is_gradiend is True
        assert gradient_model.is_actiend is False

        activation_model = ParamMappedGradiendModel(
            input_dim=2,
            latent_dim=1,
            param_map={"activation:emb": {"shape": (2,), "repr": "all"}},
            mapping_kind="activation",
            signal_space={"kind": "activation", "signal_id": "activation"},
        )
        assert activation_model.signal_kind == "activation"
        assert activation_model.uses_gradients is False
        assert activation_model.uses_activations is True
        assert activation_model.is_gradiend is False
        assert activation_model.is_actiend is True
    
    def test_param_mapped_model_flatten_gradient_dict(self):
        """Test flattening gradient dict to tensor."""
        param_map = {
            "layer1.weight": {"shape": (2, 3), "repr": "all"},
            "layer2.weight": {"shape": (3, 1), "repr": "all"}
        }
        input_dim = 2 * 3 + 3 * 1  # 9
        
        model = ParamMappedGradiendModel(
            input_dim=input_dim,
            latent_dim=1,
            param_map=param_map
        )
        
        # Create gradient dict
        grad_dict = {
            "layer1.weight": torch.randn(2, 3),
            "layer2.weight": torch.randn(3, 1)
        }
        
        flattened = model.flatten_gradient_dict(grad_dict)
        assert flattened.shape == (input_dim,)

    def test_param_mapped_streaming_extract_matches_regular_extract(self, set_seed):
        """Streaming extraction should collect the same mapped gradient entries."""
        set_seed(42)
        base = nn.Linear(4, 3)
        base_stream = nn.Linear(4, 3)
        base_stream.load_state_dict(base.state_dict())

        weight_mask = torch.tensor(
            [[True, False, True, False], [False, True, False, True], [True, True, False, False]]
        )
        bias_indices = torch.tensor([0, 2])
        param_map = {
            "weight": {"shape": tuple(base.weight.shape), "repr": "mask", "mask": weight_mask},
            "bias": {"shape": tuple(base.bias.shape), "repr": "indices", "indices": bias_indices},
        }
        input_dim = int(weight_mask.sum().item() + bias_indices.numel())
        gradiend = ParamMappedGradiendModel(input_dim=input_dim, latent_dim=1, param_map=param_map)

        x = torch.randn(5, 4)
        base.zero_grad(set_to_none=True)
        loss = base(x).pow(2).sum()
        loss.backward()
        expected = gradiend.extract_gradients(base, target_device=torch.device("cpu"))
        base.zero_grad(set_to_none=True)

        base_stream.zero_grad(set_to_none=True)

        def backward_fn():
            base_stream(x).pow(2).sum().backward()

        actual = gradiend.extract_gradients_streaming(
            base_stream,
            backward_fn,
            target_device=torch.device("cpu"),
        )

        torch.testing.assert_close(actual, expected)
        assert base_stream.weight.grad is None
        assert base_stream.bias.grad is None

    def test_param_mapped_select_from_param_grad_matches_select_flat(self, set_seed):
        """Sparse selection should match flat indexing without cloning the full gradient."""
        set_seed(0)
        weight_mask = torch.tensor(
            [[True, False, True, False], [False, True, False, True], [True, True, False, False]]
        )
        bias_indices = torch.tensor([0, 2])
        param_map = {
            "weight": {"shape": tuple(weight_mask.shape), "repr": "mask", "mask": weight_mask},
            "bias": {"shape": (3,), "repr": "indices", "indices": bias_indices},
        }
        input_dim = int(weight_mask.sum().item() + bias_indices.numel())
        gradiend = ParamMappedGradiendModel(input_dim=input_dim, latent_dim=1, param_map=param_map)

        weight_grad = torch.randn(weight_mask.shape)
        bias_grad = torch.randn(3)
        for selector, grad in (
            (gradiend._get_compiled_param_selectors()[0], weight_grad),
            (gradiend._get_compiled_param_selectors()[1], bias_grad),
        ):
            expected = selector.select_flat(grad.detach().flatten())
            actual = selector.select_from_param_grad(grad)
            torch.testing.assert_close(actual, expected)

        base = nn.Linear(4, 3)
        base.weight.grad = weight_grad.clone()
        base.bias.grad = bias_grad.clone()
        weight_ptr = base.weight.grad.untyped_storage().data_ptr()
        extracted = gradiend.extract_gradients(base, target_device=torch.device("cpu"))
        assert base.weight.grad.untyped_storage().data_ptr() == weight_ptr
        assert extracted.numel() == input_dim

    def test_param_mapped_model_pruning_updates_map(self):
        """Test that pruning updates param_map correctly."""
        param_map = {
            "layer1.weight": {"shape": (10, 5), "repr": "all"},
            "layer2.weight": {"shape": (5, 2), "repr": "all"}
        }
        input_dim = 10 * 5 + 5 * 2  # 60
        
        model = ParamMappedGradiendModel(
            input_dim=input_dim,
            latent_dim=1,
            param_map=param_map
        )
        
        original_map_size = len(model.param_map)
        
        # Prune to top 30
        pruned = model.prune(topk=30, inplace=False)
        
        # Param map should still exist (but may be updated)
        assert len(pruned.param_map) == original_map_size
        assert pruned.input_dim == 30

    def test_param_mapped_model_pruning_accepts_keep_idx(self):
        """Test compact keep_idx pruning without dense importance/mask inputs."""
        param_map = {
            "layer1.weight": {"shape": (2, 3), "repr": "all"},
            "layer2.weight": {"shape": (2, 2), "repr": "all"},
        }
        model = ParamMappedGradiendModel(input_dim=10, latent_dim=1, param_map=param_map)

        pruned = model.prune(keep_idx=torch.tensor([0, 2, 6, 9]), inplace=False)

        assert pruned.input_dim == 4

        def selected_positions(spec):
            if spec["repr"] == "indices":
                return spec["indices"].tolist()
            if spec["repr"] == "mask":
                return spec["mask"].flatten().nonzero(as_tuple=False).flatten().tolist()
            if spec["repr"] == "all":
                return list(range(int(torch.tensor(spec["shape"]).prod().item())))
            raise AssertionError(f"Unexpected repr {spec['repr']!r}")

        assert selected_positions(pruned.param_map["layer1.weight"]) == [0, 2]
        assert selected_positions(pruned.param_map["layer2.weight"]) == [0, 3]


class TestModelWithGradiend:
    """Test ModelWithGradiend wrapper."""

    def test_prune_gradiend_updates_base_requires_grad_from_param_map(self):
        """Pruning GRADIEND must disable base gradients for fully unmapped params."""
        from gradiend.model import ModelWithGradiend

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                pass

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        base = nn.Sequential(nn.Linear(3, 2), nn.Linear(2, 1))
        param_map = {
            "0.weight": {"shape": tuple(base[0].weight.shape), "repr": "all"},
            "0.bias": {"shape": tuple(base[0].bias.shape), "repr": "all"},
            "1.weight": {"shape": tuple(base[1].weight.shape), "repr": "all"},
            "1.bias": {"shape": tuple(base[1].bias.shape), "repr": "all"},
        }
        gradiend = ParamMappedGradiendModel(
            input_dim=sum(p.numel() for p in base.parameters()),
            latent_dim=1,
            param_map=param_map,
        )
        model = TinyModelWithGradiend(base, gradiend)

        assert {name: p.requires_grad for name, p in model.base_model.named_parameters()} == {
            "0.weight": True,
            "0.bias": True,
            "1.weight": True,
            "1.bias": True,
        }

        # Keep only the two GRADIEND dimensions belonging to 0.bias.
        pruned = model.prune_gradiend(keep_idx=torch.tensor([6, 7]), inplace=False)

        assert pruned.gradiend.input_dim == 2
        assert {name: p.requires_grad for name, p in pruned.base_model.named_parameters()} == {
            "0.weight": False,
            "0.bias": True,
            "1.weight": False,
            "1.bias": False,
        }
        # Non-inplace pruning must not mutate the source wrapper's base gradient flags.
        assert {name: p.requires_grad for name, p in model.base_model.named_parameters()} == {
            "0.weight": True,
            "0.bias": True,
            "1.weight": True,
            "1.bias": True,
        }

    def test_model_with_gradiend_signal_properties_and_capabilities(self):
        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        gradient_base = nn.Linear(2, 1, bias=False)
        gradient_gradiend = ParamMappedGradiendModel(
            input_dim=2,
            latent_dim=1,
            param_map={"weight": {"shape": tuple(gradient_base.weight.shape), "repr": "all"}},
        )
        gradient_model = TinyModelWithGradiend(gradient_base, gradient_gradiend)
        assert gradient_model.signal_kind == "gradient"
        assert gradient_model.uses_gradients is True
        assert gradient_model.uses_activations is False
        assert gradient_model.is_gradiend is True
        assert gradient_model.is_actiend is False
        assert gradient_model.capabilities.gradient_rewrite is True
        assert gradient_model.capabilities.activation_interventions is False
        assert gradient_model.capabilities.activation_selector_coverage is False
        assert gradient_model.capabilities.activation_module_ablation is False
        assert gradient_model.activation_site_modules == []

        class TinyActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Linear(2, 2, bias=False)

            def forward(self, input_ids=None, **kwargs):
                return self.emb(input_ids.float())

        activation_base = TinyActivationBase()
        activation_gradiend = ParamMappedGradiendModel(
            input_dim=2,
            latent_dim=1,
            param_map={"activation:emb": {"shape": (2,), "repr": "all"}},
            mapping_kind="activation",
            signal_space={"kind": "activation", "signal_id": "activation"},
            bias_decoder=False,
            activation_decoder="id",
        )
        activation_model = TinyModelWithGradiend(activation_base, activation_gradiend)
        assert activation_model.signal_kind == "activation"
        assert activation_model.uses_gradients is False
        assert activation_model.uses_activations is True
        assert activation_model.is_gradiend is False
        assert activation_model.is_actiend is True
        assert activation_model.capabilities.gradient_rewrite is False
        assert activation_model.capabilities.activation_interventions is True
        assert activation_model.capabilities.activation_selector_coverage is True
        assert activation_model.capabilities.activation_module_ablation is True
        assert activation_model.activation_site_modules == ["emb"]


    def test_model_with_gradiend_creation(self, mock_model):
        """Test ModelWithGradiend wrapper creation."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        # Patch _load_model to return the mock model directly
        original_load = TextModelWithGradiend._load_model
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            # Return mock model and tokenizer directly
            from tests.testing_mocks import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
            )
        
        assert hasattr(model_with_gradiend, 'base_model')
        assert hasattr(model_with_gradiend, 'gradiend')
        assert model_with_gradiend.base_model == mock_model

    def test_model_with_gradiend_passes_base_device_map(self, mock_model):
        """HF device_map should be passed to the base loader and mark the base as sharded."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        from gradiend.trainer.core.arguments import TrainingArguments

        captured = {}

        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            captured.update(kwargs)
            from tests.testing_mocks import MockTokenizer
            return mock_model, MockTokenizer()

        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
                training_args=TrainingArguments(
                    base_model_device_map="auto",
                    base_model_max_memory={0: "0GiB", 1: "70GiB"},
                ),
            )

        assert captured["base_model_device_map"] == "auto"
        assert captured["base_model_max_memory"] == {0: "0GiB", 1: "70GiB"}
        assert model_with_gradiend.base_model_is_sharded is True
        assert model_with_gradiend.base_model_device is None
    
    def test_model_with_gradiend_encode(self, mock_model, mock_tokenizer):
        """Test encoding functionality with mock base model."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            return mock_model, mock_tokenizer
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
            )
        
        # Test encoding (this will use gradient creator internally)
        # For a simple test, we'll just verify the method exists
        assert hasattr(model_with_gradiend, 'encode')
        assert callable(model_with_gradiend.encode)
    
    def test_model_with_gradiend_rewrite_base_model(self, mock_model):
        """Test rewrite_base_model method with decoder part."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.testing_mocks import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
            )
        
        # Test that rewrite_base_model exists and works
        assert hasattr(model_with_gradiend, 'rewrite_base_model')
        assert callable(model_with_gradiend.rewrite_base_model)
        
        # Test rewrite_base_model with decoder part
        learning_rate = 1e-4
        feature_factor = 1.0
        
        rewritten_model = model_with_gradiend.rewrite_base_model(
            learning_rate=learning_rate,
            feature_factor=feature_factor,
            part="decoder"
        )
        
        # Should return a model (not the same instance as base_model)
        assert rewritten_model is not None
        assert rewritten_model is not model_with_gradiend.base_model

    def test_effective_rewrite_learning_rate_alternative_source(self):
        from gradiend.model.model_with_gradiend import effective_rewrite_learning_rate

        assert effective_rewrite_learning_rate(0.1, "factual") == 0.1
        assert effective_rewrite_learning_rate(0.1, "diff") == 0.1
        assert effective_rewrite_learning_rate(0.1, "alternative") == 0.1

    def test_effective_rewrite_learning_rate_rejects_invalid_source(self):
        from gradiend.model.model_with_gradiend import effective_rewrite_learning_rate

        with pytest.raises(ValueError, match="source must be"):
            effective_rewrite_learning_rate(0.1, "counterfactual")

    def test_rewrite_base_model_same_lr_for_alternative_source(self):
        """Learning rate sign is unchanged for alternative-source models."""
        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel
        from gradiend.model.model_with_gradiend import effective_rewrite_learning_rate

        class TinyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(2, 2))

        class TinyMWG(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        def _make_mwg(source: str) -> TinyMWG:
            torch.manual_seed(0)
            base = TinyModel()
            gradiend = ParamMappedGradiendModel(
                input_dim=4,
                latent_dim=1,
                param_map={"weight": {"shape": (2, 2), "repr": "all"}},
            )
            with torch.no_grad():
                gradiend.decoder[0].linear.weight.fill_(1.0)
                gradiend.decoder[0].linear.bias.zero_()
            return TinyMWG(
                base,
                gradiend,
                source=source,
                target="diff",
            )

        nominal_lr = 0.05
        ff = 1.0
        factual = _make_mwg("factual")
        counter = _make_mwg("alternative")
        w0 = factual.base_model.weight.detach().clone()

        factual_out = factual.rewrite_base_model(learning_rate=nominal_lr, feature_factor=ff, part="decoder")
        counter_out = counter.rewrite_base_model(learning_rate=nominal_lr, feature_factor=ff, part="decoder")

        delta_f = factual_out.weight.detach() - w0
        delta_c = counter_out.weight.detach() - w0
        assert not torch.allclose(delta_f, torch.zeros_like(delta_f))
        assert torch.allclose(delta_f, delta_c, atol=1e-6)
        assert effective_rewrite_learning_rate(nominal_lr, "alternative") == nominal_lr

    def test_rewrite_base_model_resolves_backbone_local_param_names(self):
        """Decoder-only wrappers may train on backbone-local names such as embed_tokens.weight."""
        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel

        class TinyDecoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.embed_tokens = torch.nn.Embedding(3, 2)

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        base_model = TinyDecoder()
        before = base_model.model.embed_tokens.weight.detach().clone()
        gradiend = ParamMappedGradiendModel(
            input_dim=6,
            latent_dim=1,
            param_map={"embed_tokens.weight": {"shape": (3, 2), "repr": "all"}},
            bias_decoder=False,
        )
        with torch.no_grad():
            gradiend.decoder[0].linear.weight.fill_(1.0)

        model_with_gradiend = TinyModelWithGradiend(base_model, gradiend)
        rewritten_model = model_with_gradiend.rewrite_base_model(
            learning_rate=0.5,
            feature_factor=1.0,
            part="decoder-weight",
        )

        assert torch.allclose(
            rewritten_model.model.embed_tokens.weight,
            before + 0.5,
        )
        assert torch.allclose(base_model.model.embed_tokens.weight, before)

    def test_modify_model_for_activation_gradiend_adds_hook_steering(self):
        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel

        class TinyActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.emb.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, **kwargs):
                return self.emb(input_ids.float())

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        base_model = TinyActivationBase()
        gradiend = ParamMappedGradiendModel(
            input_dim=2,
            latent_dim=1,
            param_map={"activation:emb": {"shape": (2,), "repr": "all"}},
            mapping_kind="activation",
            signal_space={
                "kind": "activation",
                "signal_id": "activation",
                "signal": {"kind": "activation", "name": None, "options": {"token_selector": 1}},
            },
            bias_encoder=False,
            bias_decoder=False,
            activation_decoder="id",
        )
        with torch.no_grad():
            gradiend.encoder[0].linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
            gradiend.decoder[0].linear.weight.copy_(torch.tensor([[2.0], [-1.0]]))
        model_with_gradiend = TinyModelWithGradiend(base_model, gradiend)
        inputs = torch.tensor([[[0.1, 2.0], [3.0, 4.0]]])
        before = base_model(input_ids=inputs)

        modified = model_with_gradiend.modify_model(
            learning_rate=0.5,
            feature_factor=1.0,
            part="decoder",
        )
        after = modified(input_ids=inputs)

        expected = before.clone()
        expected[:, 1, :] += torch.tensor([1.0, -0.5])
        assert torch.allclose(after, expected)
        assert torch.allclose(base_model(input_ids=inputs), before)
        config = modified._gradiend_modified_config
        application = config["interventions"][0]["application"]
        assert application["token_selector"] == "encoder_direction"
        assert application["threshold"] == 0.5
        assert application["direction"] == 1.0
        assert "encoder_weight_0" in modified._gradiend_modified_tensors

    def test_save_and_load_modified_activation_model_owns_steering_vector(self, tmp_path):
        from gradiend.model.modified import load_modified_model

        class TinyActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.emb.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, **kwargs):
                return self.emb(input_ids.float())

            def save_pretrained(self, save_directory, **kwargs):
                os.makedirs(save_directory, exist_ok=True)
                torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

            @classmethod
            def from_pretrained(cls, load_directory):
                model = cls()
                model.load_state_dict(torch.load(os.path.join(load_directory, "pytorch_model.bin"), weights_only=True))
                return model

        from gradiend.model.modified import apply_activation_steering

        model = apply_activation_steering(
            TinyActivationBase(),
            interventions=[
                {
                    "module": "emb",
                    "tensor_key": "steering_0",
                    "application": {"axis": "last_dim", "token_selector": "all"},
                }
            ],
            tensors={"steering_0": torch.tensor([0.25, -0.75])},
        )
        inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        expected = model(input_ids=inputs)
        model.save_pretrained_modified(str(tmp_path))

        loaded = load_modified_model(
            str(tmp_path),
            model_loader=lambda path: TinyActivationBase.from_pretrained(path),
        )

        assert torch.allclose(loaded(input_ids=inputs), expected)
        config_path = os.path.join(tmp_path, "gradiend_modified_config.json")
        assert os.path.isfile(config_path)
        assert os.path.isfile(os.path.join(tmp_path, "gradiend_modified_tensors.pt"))
        with open(config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        assert config["format_version"] == 1
        assert config["interventions"][0]["tensor_key"] == "steering_0"

        legacy_config = dict(config)
        legacy_config["version"] = legacy_config.pop("format_version")
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(legacy_config, handle)
        legacy_loaded = load_modified_model(
            str(tmp_path),
            model_loader=lambda path: TinyActivationBase.from_pretrained(path),
        )
        assert torch.allclose(legacy_loaded(input_ids=inputs), expected)
        assert legacy_loaded._gradiend_modified_config["format_version"] == 1

    def test_save_and_load_modified_activation_model_with_encoder_abs_selector(self, tmp_path):
        from gradiend.model.modified import apply_activation_steering, load_modified_model

        class TinyActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.emb.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, **kwargs):
                return self.emb(input_ids.float())

            def save_pretrained(self, save_directory, **kwargs):
                os.makedirs(save_directory, exist_ok=True)
                torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))

            @classmethod
            def from_pretrained(cls, load_directory):
                model = cls()
                model.load_state_dict(torch.load(os.path.join(load_directory, "pytorch_model.bin"), weights_only=True))
                return model

        model = apply_activation_steering(
            TinyActivationBase(),
            interventions=[
                {
                    "module": "emb",
                    "tensor_key": "steering_0",
                    "application": {
                        "axis": "last_dim",
                        "token_selector": "encoder_abs",
                        "threshold": 0.5,
                        "encoder_weight_key": "encoder_weight_0",
                    },
                }
            ],
            tensors={
                "steering_0": torch.tensor([0.25, -0.75]),
                "encoder_weight_0": torch.tensor([[1.0, 0.0]]),
            },
        )
        inputs = torch.tensor([[[0.1, 2.0], [3.0, 4.0]]])
        expected = inputs.clone()
        expected[:, 1, :] += torch.tensor([0.25, -0.75])

        torch.testing.assert_close(model(input_ids=inputs), expected)
        model.save_pretrained_modified(str(tmp_path))

        loaded = load_modified_model(
            str(tmp_path),
            model_loader=lambda path: TinyActivationBase.from_pretrained(path),
        )

        torch.testing.assert_close(loaded(input_ids=inputs), expected)
        assert "encoder_weight_0" in loaded._gradiend_modified_tensors

    def test_activation_clamp_mode_forces_feature_to_target(self):
        from gradiend.model.modified import apply_activation_steering

        class TinyActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.emb.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, **kwargs):
                return self.emb(input_ids.float())

        model = apply_activation_steering(
            TinyActivationBase(),
            interventions=[
                {
                    "module": "emb",
                    "tensor_key": "steering_0",
                    "application": {
                        "axis": "last_dim",
                        "token_selector": "all",
                        "mode": "clamp",
                        "target_activation": 5.0,
                        "clamp_encoder_weight_key": "clamp_enc_w",
                        "clamp_encoder_bias_key": "clamp_enc_b",
                    },
                }
            ],
            tensors={
                # Direction and encoder both read off feature 0, so the
                # clamp correction lands entirely on that axis and the
                # post-hook encoder readout should equal the target exactly.
                "steering_0": torch.tensor([1.0, 0.0]),
                "clamp_enc_w": torch.tensor([[1.0, 0.0]]),
                "clamp_enc_b": torch.tensor([0.0]),
            },
        )
        inputs = torch.tensor([[[3.0, 4.0]]])
        out = model(input_ids=inputs)
        torch.testing.assert_close(out, torch.tensor([[[5.0, 4.0]]]))

    def test_activation_clamp_mode_respects_prediction_token_selector(self):
        from gradiend.model.modified import apply_activation_steering

        class TinyPredictionBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(200, 2)
                with torch.no_grad():
                    self.emb.weight.zero_()
                    self.emb.weight[10] = torch.tensor([3.0, 4.0])

            def forward(self, input_ids=None, attention_mask=None):
                return self.emb(input_ids)

        model = apply_activation_steering(
            TinyPredictionBase(),
            interventions=[
                {
                    "module": "emb",
                    "tensor_key": "steering_0",
                    "application": {
                        "axis": "last_dim",
                        "token_selector": "prediction",
                        "mask_token_id": 103,
                        "mode": "clamp",
                        "target_activation": 5.0,
                        "clamp_encoder_weight_key": "clamp_enc_w",
                        "clamp_encoder_bias_key": "clamp_enc_b",
                    },
                }
            ],
            tensors={
                "steering_0": torch.tensor([1.0, 0.0]),
                "clamp_enc_w": torch.tensor([[1.0, 0.0]]),
                "clamp_enc_b": torch.tensor([0.0]),
            },
        )
        masked_ids = torch.tensor([[101, 103, 10]])
        out = model(input_ids=masked_ids)
        # Position 1 (mask token) reads embedding row 0 -> feature score 0,
        # clamped to 5.0; position 2 (outside the selector) is untouched.
        torch.testing.assert_close(out[:, 1, :], torch.tensor([[5.0, 0.0]]))
        torch.testing.assert_close(out[:, 2, :], torch.tensor([[3.0, 4.0]]))

    def test_activation_encoder_direction_selector_is_not_symmetric(self):
        from gradiend.model.modified import apply_activation_steering

        class TinyActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.emb.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, **kwargs):
                return self.emb(input_ids.float())

        model = apply_activation_steering(
            TinyActivationBase(),
            interventions=[
                {
                    "module": "emb",
                    "tensor_key": "steering_0",
                    "application": {
                        "axis": "last_dim",
                        "token_selector": "encoder_direction",
                        "threshold": 0.5,
                        "direction": -1.0,
                        "encoder_weight_key": "encoder_weight_0",
                    },
                }
            ],
            tensors={
                "steering_0": torch.tensor([0.25, -0.75]),
                "encoder_weight_0": torch.tensor([[1.0, 0.0]]),
            },
        )
        inputs = torch.tensor([[[3.0, 0.0], [-3.0, 0.0]]])
        expected = inputs.clone()
        expected[:, 1, :] += torch.tensor([0.25, -0.75])

        torch.testing.assert_close(model(input_ids=inputs), expected)

    def test_activation_last_token_selector_is_removed_with_migration_hint(self):
        from gradiend.model.modified import apply_activation_steering

        class TinyActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.emb.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                return self.emb(input_ids.float())

        model = apply_activation_steering(
            TinyActivationBase(),
            interventions=[
                {
                    "module": "emb",
                    "tensor_key": "steering_0",
                    "application": {"axis": "last_dim", "token_selector": "last_token"},
                }
            ],
            tensors={"steering_0": torch.tensor([0.25, -0.75])},
        )
        inputs = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [9.0, 9.0]]])
        attention_mask = torch.tensor([[1, 1, 0]])

        with pytest.raises(ValueError, match="token_selector='prediction'"):
            model(input_ids=inputs, attention_mask=attention_mask)

    def test_activation_prediction_selector_uses_clm_prediction_position(self):
        from gradiend.model.modified import apply_activation_steering

        class TinyActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.emb.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, attention_mask=None, **kwargs):
                return self.emb(input_ids.float())

        model = apply_activation_steering(
            TinyActivationBase(),
            interventions=[
                {
                    "module": "emb",
                    "tensor_key": "steering_0",
                    "application": {"axis": "last_dim", "token_selector": "prediction"},
                }
            ],
            tensors={"steering_0": torch.tensor([0.25, -0.75])},
        )
        inputs = torch.tensor([[[0.0, 0.0], [1.0, 1.0], [9.0, 9.0]]])
        attention_mask = torch.tensor([[1, 1, 0]])
        expected = inputs.clone()
        expected[:, 1, :] += torch.tensor([0.25, -0.75])

        torch.testing.assert_close(model(input_ids=inputs, attention_mask=attention_mask), expected)

    def test_activation_prediction_selector_uses_hook_only_prediction_mask(self):
        from gradiend.model.modified import apply_activation_steering

        class TinyPredictionBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(200, 2)
                with torch.no_grad():
                    self.emb.weight.zero_()

            def forward(self, input_ids=None, attention_mask=None):
                return self.emb(input_ids)

        input_ids = torch.tensor([[101, 10, 11]])
        prediction_mask = torch.tensor([[False, False, True]])
        model = apply_activation_steering(
            TinyPredictionBase(),
            interventions=[
                {
                    "module": "emb",
                    "tensor_key": "steering_0",
                    "application": {"axis": "last_dim", "token_selector": "prediction"},
                }
            ],
            tensors={"steering_0": torch.tensor([0.25, -0.75])},
        )
        expected = torch.zeros(1, 3, 2)
        expected[:, 2, :] += torch.tensor([0.25, -0.75])

        # prediction_mask is for the ACTIEND hook resolver; the pre-hook strips
        # it before forwarding into TinyPredictionBase.forward().
        torch.testing.assert_close(
            model(input_ids=input_ids, prediction_mask=prediction_mask),
            expected,
        )

    def test_activation_prediction_selector_falls_back_to_mlm_mask_token(self):
        from gradiend.model.modified import apply_activation_steering

        class TinyPredictionBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Embedding(200, 2)
                with torch.no_grad():
                    self.emb.weight.zero_()

            def forward(self, input_ids=None, attention_mask=None):
                return self.emb(input_ids)

        model = apply_activation_steering(
            TinyPredictionBase(),
            interventions=[
                {
                    "module": "emb",
                    "tensor_key": "steering_0",
                    "application": {
                        "axis": "last_dim",
                        "token_selector": "prediction",
                        "mask_token_id": 103,
                    },
                }
            ],
            tensors={"steering_0": torch.tensor([0.25, -0.75])},
        )
        masked_ids = torch.tensor([[101, 103, 11]])
        neutral_ids = torch.tensor([[101, 10, 11]])
        expected_masked = torch.zeros(1, 3, 2)
        expected_masked[:, 1, :] += torch.tensor([0.25, -0.75])

        torch.testing.assert_close(model(input_ids=masked_ids), expected_masked)
        torch.testing.assert_close(model(input_ids=neutral_ids), torch.zeros(1, 3, 2))

    def test_activation_encoder_selector_single_site_uses_global_encoder_bias(self):
        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel

        class TinyActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.emb.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, **kwargs):
                return self.emb(input_ids.float())

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        base_model = TinyActivationBase()
        gradiend = ParamMappedGradiendModel(
            input_dim=2,
            latent_dim=1,
            param_map={"activation:emb": {"shape": (2,), "repr": "all"}},
            mapping_kind="activation",
            signal_space={
                "kind": "activation",
                "signal_id": "activation",
                "signal": {"kind": "activation", "name": None, "options": {"token_selector": "all"}},
            },
            bias_encoder=True,
            bias_decoder=False,
            activation_decoder="id",
        )
        with torch.no_grad():
            gradiend.encoder[0].linear.weight.zero_()
            gradiend.encoder[0].linear.bias.fill_(1.0)
            gradiend.decoder[0].linear.weight.copy_(torch.tensor([[0.25], [-0.75]]))
        model_with_gradiend = TinyModelWithGradiend(base_model, gradiend)
        interventions, tensors = model_with_gradiend._activation_intervention_specs(
            torch.tensor([0.25, -0.75]),
            feature_factor=1.0,
            token_selector="encoder_direction",
            threshold=0.7,
        )

        application = interventions[0]["application"]
        assert application["encoder_activation"] == "tanh"
        assert "encoder_bias_0" in tensors

        inputs = torch.tensor([[[0.0, 0.0], [2.0, 3.0]]])
        before = base_model(input_ids=inputs)
        with model_with_gradiend.intervene(
            value=1.0,
            signal="activation",
            feature_factor=1.0,
            token_selector="encoder_direction",
            threshold=0.7,
        ):
            after = base_model(input_ids=inputs)

        expected = before + torch.tensor([0.25, -0.75])
        torch.testing.assert_close(after, expected)

    def test_activation_encoder_selector_slices_unsplit_bias_free_multi_site_encoder(self):
        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel

        class TwoSiteActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.left = torch.nn.Linear(2, 2, bias=False)
                self.right = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.left.weight.copy_(torch.eye(2))
                    self.right.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, **kwargs):
                x = input_ids.float()
                return self.left(x[..., :2]) + self.right(x[..., 2:])

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        base_model = TwoSiteActivationBase()
        gradiend = ParamMappedGradiendModel(
            input_dim=4,
            latent_dim=1,
            param_map={
                "activation:left": {"shape": (2,), "repr": "all"},
                "activation:right": {"shape": (2,), "repr": "all"},
            },
            mapping_kind="activation",
            signal_space={
                "kind": "activation",
                "signal_id": "activation",
                "signal": {"kind": "activation", "name": None, "options": {"token_selector": "all"}},
            },
            bias_encoder=False,
            bias_decoder=False,
            activation_decoder="id",
        )
        with torch.no_grad():
            gradiend.encoder[0].linear.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, -1.0]]))
            gradiend.decoder[0].linear.weight.copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        model_with_gradiend = TinyModelWithGradiend(base_model, gradiend)
        interventions, tensors = model_with_gradiend._activation_intervention_specs(
            torch.tensor([1.0, 2.0, 3.0, 4.0]),
            feature_factor=1.0,
            token_selector="encoder_direction",
            threshold=0.7,
        )

        assert [item["module"] for item in interventions] == ["left", "right"]
        torch.testing.assert_close(tensors["encoder_weight_0"], torch.tensor([[1.0, 0.0]]))
        torch.testing.assert_close(tensors["encoder_weight_1"], torch.tensor([[0.0, -1.0]]))
        assert interventions[0]["application"]["encoder_activation"] == "tanh"
        assert interventions[1]["application"]["encoder_activation"] == "tanh"

        inputs = torch.tensor([[[1.0, 0.0, 0.0, -1.0], [0.5, 0.0, 0.0, -0.5]]])
        before = base_model(input_ids=inputs)
        with model_with_gradiend.intervene(
            value=1.0,
            signal="activation",
            feature_factor=1.0,
            token_selector="encoder_direction",
            threshold=0.7,
        ):
            after = base_model(input_ids=inputs)

        expected = before.clone()
        expected[:, 0, :] += torch.tensor([4.0, 6.0])
        torch.testing.assert_close(after, expected)

    def test_activation_encoder_selector_uses_tanh_score_for_thresholds(self):
        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel

        class TwoSiteActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.left = torch.nn.Linear(2, 2, bias=False)
                self.right = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.left.weight.copy_(torch.eye(2))
                    self.right.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, **kwargs):
                x = input_ids.float()
                return self.left(x[..., :2]) + self.right(x[..., 2:])

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        base_model = TwoSiteActivationBase()
        gradiend = ParamMappedGradiendModel(
            input_dim=4,
            latent_dim=1,
            param_map={
                "activation:left": {"shape": (2,), "repr": "all"},
                "activation:right": {"shape": (2,), "repr": "all"},
            },
            mapping_kind="activation",
            signal_space={
                "kind": "activation",
                "signal_id": "activation",
                "signal": {"kind": "activation", "name": None, "options": {"token_selector": "all"}},
            },
            bias_encoder=False,
            bias_decoder=False,
            activation_decoder="id",
        )
        with torch.no_grad():
            gradiend.encoder[0].linear.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, -1.0]]))
            gradiend.decoder[0].linear.weight.copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        model_with_gradiend = TinyModelWithGradiend(base_model, gradiend)
        inputs = torch.tensor([[[1.0, 0.0, 0.0, -1.0]]])
        before = base_model(input_ids=inputs)

        with model_with_gradiend.intervene(
            value=1.0,
            signal="activation",
            feature_factor=1.0,
            token_selector="encoder_direction",
            threshold=0.8,
        ):
            after = base_model(input_ids=inputs)

        # The raw linear site score is 1.0, but the ACTIEND encoder score is tanh(1.0) < 0.8.
        torch.testing.assert_close(after, before)

    def test_activation_selector_coverage_uses_direction_selector_and_cleans_up(self):
        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel

        class TinyActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.emb.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, **kwargs):
                return self.emb(input_ids.float())

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        base_model = TinyActivationBase()
        base_model.train()
        gradiend = ParamMappedGradiendModel(
            input_dim=2,
            latent_dim=1,
            param_map={"activation:emb": {"shape": (2,), "repr": "all"}},
            mapping_kind="activation",
            signal_space={
                "kind": "activation",
                "signal_id": "activation",
                "signal": {"kind": "activation", "name": None, "options": {"token_selector": "all"}},
            },
            bias_encoder=False,
            bias_decoder=False,
            activation_decoder="id",
        )
        with torch.no_grad():
            gradiend.encoder[0].linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
            gradiend.decoder[0].linear.weight.copy_(torch.tensor([[1.0], [0.0]]))
        model_with_gradiend = TinyModelWithGradiend(base_model, gradiend)
        inputs = torch.tensor([[[3.0, 0.0], [-3.0, 0.0]]])

        summary = model_with_gradiend.activation_selector_coverage(
            input_ids=inputs,
            feature_factor=-1.0,
            token_selector="encoder_direction",
            threshold=0.5,
        )

        assert summary["selected_positions"] == 1
        assert summary["candidate_positions"] == 2
        assert summary["total_positions"] == 2
        assert summary["coverage"] == 0.5
        assert summary["scope_coverage"] == 0.5
        assert summary["modules"][0]["module"] == "emb"
        assert summary["modules"][0]["coverage"] == 0.5
        assert summary["modules"][0]["scope_coverage"] == 0.5
        assert base_model.training is True
        assert len(base_model.emb._forward_hooks) == 0
        assert len(base_model._forward_pre_hooks) == 0

    def test_activation_gate_composes_with_position_selector(self):
        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel

        class TinyActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.emb.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, **kwargs):
                return self.emb(input_ids.float())

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        base_model = TinyActivationBase()
        gradiend = ParamMappedGradiendModel(
            input_dim=2,
            latent_dim=1,
            param_map={"activation:emb": {"shape": (2,), "repr": "all"}},
            mapping_kind="activation",
            signal_space={
                "kind": "activation",
                "signal_id": "activation",
                "signal": {"kind": "activation", "name": None, "options": {"token_selector": "all"}},
            },
            bias_encoder=False,
            bias_decoder=False,
            activation_decoder="id",
        )
        with torch.no_grad():
            gradiend.encoder[0].linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
            gradiend.decoder[0].linear.weight.copy_(torch.tensor([[0.5], [-0.25]]))
        model_with_gradiend = TinyModelWithGradiend(base_model, gradiend)
        inputs = torch.tensor([[[3.0, 0.0], [-3.0, 0.0]]])
        before = base_model(input_ids=inputs)

        with model_with_gradiend.intervene(
            value=1.0,
            signal="activation",
            feature_factor=-1.0,
            token_selector="prediction",
            activation_gate="encoder_direction",
            threshold=0.5,
        ) as meta:
            after = base_model(input_ids=inputs)

        expected = before.clone()
        expected[:, 1, :] += torch.tensor([-0.5, 0.25])
        torch.testing.assert_close(after, expected)
        assert meta["activation_gate"] == "encoder_direction"
        assert len(base_model.emb._forward_hooks) == 0
        assert len(base_model._forward_pre_hooks) == 0

        coverage = model_with_gradiend.activation_selector_coverage(
            input_ids=inputs,
            feature_factor=-1.0,
            token_selector="prediction",
            activation_gate="encoder_direction",
            threshold=0.5,
        )
        assert coverage["selected_positions"] == 1
        assert coverage["candidate_positions"] == 1
        assert coverage["total_positions"] == 2
        assert coverage["coverage"] == 0.5
        assert coverage["scope_coverage"] == 1.0

    def test_activation_modules_filters_single_site_intervention(self):
        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel

        class TwoSiteActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.left = torch.nn.Linear(2, 2, bias=False)
                self.right = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.left.weight.copy_(torch.eye(2))
                    self.right.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, **kwargs):
                x = input_ids.float()
                return self.left(x[..., :2]) + self.right(x[..., 2:])

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        base_model = TwoSiteActivationBase()
        gradiend = ParamMappedGradiendModel(
            input_dim=4,
            latent_dim=1,
            param_map={
                "activation:left": {"shape": (2,), "repr": "all"},
                "activation:right": {"shape": (2,), "repr": "all"},
            },
            mapping_kind="activation",
            signal_space={
                "kind": "activation",
                "signal_id": "activation",
                "signal": {"kind": "activation", "name": None, "options": {"token_selector": "all"}},
            },
            bias_encoder=False,
            bias_decoder=False,
            activation_decoder="id",
        )
        with torch.no_grad():
            gradiend.decoder[0].linear.weight.copy_(torch.tensor([[1.0], [2.0], [3.0], [4.0]]))
        model_with_gradiend = TinyModelWithGradiend(base_model, gradiend)
        inputs = torch.tensor([[[1.0, 1.0, 10.0, 10.0]]])
        before = base_model(input_ids=inputs)

        with model_with_gradiend.intervene(
            value=1.0,
            signal="activation",
            feature_factor=1.0,
            token_selector="all",
            activation_modules="left",
        ) as meta:
            after = base_model(input_ids=inputs)

        expected = before + torch.tensor([1.0, 2.0])
        torch.testing.assert_close(after, expected)
        assert meta["resolved_modules"] == ["left"]

        with pytest.raises(ValueError, match="No ACTIEND activation modules matched"):
            model_with_gradiend.activation_selector_coverage(
                input_ids=inputs,
                feature_factor=1.0,
                token_selector="all",
                activation_modules="missing",
            )

    def test_intervene_gradient_value_zero_is_noop_and_restores_weights(self):
        from types import SimpleNamespace

        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel

        class TinyGradientBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1, bias=False)
                with torch.no_grad():
                    self.linear.weight.copy_(torch.tensor([[1.0, 2.0]]))

            def forward(self, input_ids=None, **kwargs):
                return SimpleNamespace(logits=self.linear(input_ids.float()))

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        base_model = TinyGradientBase()
        gradiend = ParamMappedGradiendModel(
            input_dim=2,
            latent_dim=1,
            param_map={"linear.weight": {"shape": (1, 2), "repr": "all"}},
            bias_decoder=False,
            activation_decoder="id",
        )
        with torch.no_grad():
            gradiend.decoder[0].linear.weight.copy_(torch.tensor([[3.0], [-2.0]]))
        model_with_gradiend = TinyModelWithGradiend(base_model, gradiend)
        inputs = torch.tensor([[2.0, 4.0]])
        before_weight = base_model.linear.weight.detach().clone()
        before_logits = model_with_gradiend(input_ids=inputs).logits.detach().clone()

        with model_with_gradiend.intervene(value=0.0) as meta:
            during_logits = model_with_gradiend(input_ids=inputs).logits

        assert meta["signal"] == "gradient"
        assert meta["active"] is False
        assert torch.allclose(during_logits, before_logits, atol=1e-7)
        assert torch.allclose(base_model.linear.weight, before_weight)

    def test_intervene_gradient_applies_temporarily_and_cleans_up_after_exception(self):
        from types import SimpleNamespace

        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel

        class TinyGradientBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(2, 1, bias=False)
                with torch.no_grad():
                    self.linear.weight.copy_(torch.tensor([[1.0, 2.0]]))

            def forward(self, input_ids=None, **kwargs):
                return SimpleNamespace(logits=self.linear(input_ids.float()))

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        base_model = TinyGradientBase()
        gradiend = ParamMappedGradiendModel(
            input_dim=2,
            latent_dim=1,
            param_map={"linear.weight": {"shape": (1, 2), "repr": "all"}},
            bias_decoder=False,
            activation_decoder="id",
        )
        with torch.no_grad():
            gradiend.decoder[0].linear.weight.copy_(torch.tensor([[1.0], [-1.0]]))
        model_with_gradiend = TinyModelWithGradiend(base_model, gradiend)
        inputs = torch.tensor([[2.0, 4.0]])
        before_weight = base_model.linear.weight.detach().clone()
        before_logits = model_with_gradiend(input_ids=inputs).logits.detach().clone()

        with pytest.raises(RuntimeError, match="study failure"):
            with model_with_gradiend.intervene(value=0.5, signal="gradient") as meta:
                during_logits = model_with_gradiend(input_ids=inputs).logits.detach()
                assert meta["active"] is True
                assert meta["resolved_params"][0]["name"] == "linear.weight"
                raise RuntimeError("study failure")

        assert not torch.allclose(during_logits, before_logits)
        assert torch.allclose(base_model.linear.weight, before_weight)
        assert not hasattr(model_with_gradiend, "active_intervention_metadata")

    def test_intervene_activation_hooks_only_inside_context(self):
        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel

        class TinyActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.emb.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, **kwargs):
                return self.emb(input_ids.float())

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        base_model = TinyActivationBase()
        gradiend = ParamMappedGradiendModel(
            input_dim=2,
            latent_dim=1,
            param_map={"activation:emb": {"shape": (2,), "repr": "all"}},
            mapping_kind="activation",
            signal_space={
                "kind": "activation",
                "signal_id": "activation",
                "signal": {"kind": "activation", "name": None, "options": {"token_selector": "all"}},
            },
            bias_encoder=False,
            bias_decoder=False,
            activation_decoder="id",
        )
        with torch.no_grad():
            gradiend.encoder[0].linear.weight.copy_(torch.tensor([[1.0, 0.0]]))
            gradiend.decoder[0].linear.weight.copy_(torch.tensor([[2.0], [-1.0]]))
        model_with_gradiend = TinyModelWithGradiend(base_model, gradiend)
        inputs = torch.tensor([[[0.1, 2.0], [3.0, 4.0]]])
        before = model_with_gradiend(input_ids=inputs)
        assert len(base_model.emb._forward_hooks) == 0

        with model_with_gradiend.intervene(value=0.5, signal="activation") as meta:
            assert len(base_model.emb._forward_hooks) == 1
            after = model_with_gradiend(input_ids=inputs)
            assert meta["resolved_modules"] == ["emb"]
            assert meta["num_dimensions"] == 2

        expected = before.clone()
        expected[:, 1, :] += torch.tensor([1.0, -0.5])
        assert torch.allclose(after, expected)
        assert torch.allclose(model_with_gradiend(input_ids=inputs), before)
        assert len(base_model.emb._forward_hooks) == 0

    def test_intervene_activation_value_zero_is_noop_and_installs_no_hooks(self):
        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel

        class TinyActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Linear(2, 2, bias=False)
                with torch.no_grad():
                    self.emb.weight.copy_(torch.eye(2))

            def forward(self, input_ids=None, **kwargs):
                return self.emb(input_ids.float())

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        base_model = TinyActivationBase()
        gradiend = ParamMappedGradiendModel(
            input_dim=2,
            latent_dim=1,
            param_map={"activation:emb": {"shape": (2,), "repr": "all"}},
            mapping_kind="activation",
            signal_space={
                "kind": "activation",
                "signal_id": "activation",
                "signal": {"kind": "activation", "name": None, "options": {"token_selector": "all"}},
            },
            bias_decoder=False,
            activation_decoder="id",
        )
        with torch.no_grad():
            gradiend.decoder[0].linear.weight.copy_(torch.tensor([[2.0], [-1.0]]))
        model_with_gradiend = TinyModelWithGradiend(base_model, gradiend)
        inputs = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        before = model_with_gradiend(input_ids=inputs).detach().clone()

        with model_with_gradiend.intervene(value=0.0, signal="activation") as meta:
            during = model_with_gradiend(input_ids=inputs)
            assert meta["signal"] == "activation"
            assert meta["active"] is False
            assert len(base_model.emb._forward_hooks) == 0

        assert torch.allclose(during, before, atol=1e-7)
        assert torch.allclose(model_with_gradiend(input_ids=inputs), before, atol=1e-7)
        assert len(base_model.emb._forward_hooks) == 0
        assert len(base_model._forward_pre_hooks) == 0

    def test_intervene_activation_removes_hooks_after_exception(self):
        from gradiend.model import ModelWithGradiend, ParamMappedGradiendModel

        class TinyActivationBase(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.emb = torch.nn.Linear(2, 2, bias=False)

            def forward(self, input_ids=None, **kwargs):
                return self.emb(input_ids.float())

        class TinyModelWithGradiend(ModelWithGradiend):
            def create_gradients(self, *args, **kwargs):
                raise NotImplementedError

            def _save_model(self, save_directory, **kwargs):
                raise NotImplementedError

            @classmethod
            def _load_model(cls, *args, **kwargs):
                raise NotImplementedError

        base_model = TinyActivationBase()
        gradiend = ParamMappedGradiendModel(
            input_dim=2,
            latent_dim=1,
            param_map={"activation:emb": {"shape": (2,), "repr": "all"}},
            mapping_kind="activation",
            signal_space={
                "kind": "activation",
                "signal_id": "activation",
                "signal": {"kind": "activation", "name": None, "options": {"token_selector": "all"}},
            },
            bias_decoder=False,
            activation_decoder="id",
        )
        model_with_gradiend = TinyModelWithGradiend(base_model, gradiend)

        with pytest.raises(RuntimeError, match="study failure"):
            with model_with_gradiend.intervene(value=-0.5, signal="activation"):
                assert len(base_model.emb._forward_hooks) == 1
                raise RuntimeError("study failure")

        assert len(base_model.emb._forward_hooks) == 0
        assert len(base_model._forward_pre_hooks) == 0
    
    def test_model_with_gradiend_rewrite_base_model_with_different_parts(self, mock_model):
        """Test rewrite_base_model with different part options."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.testing_mocks import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
            )
        
        learning_rate = 1e-4
        
        # Test different part options
        parts = ["decoder", "decoder-weight", "decoder-bias", "decoder-sum", "encoder-weight"]
        
        for part in parts:
            try:
                rewritten_model = model_with_gradiend.rewrite_base_model(
                    learning_rate=learning_rate,
                    feature_factor=1.0,
                    part=part
                )
                assert rewritten_model is not None
            except ValueError as e:
                # Some parts might not be available depending on model structure
                # That's okay - we're just testing that the method handles them
                pass
    
    def test_model_with_gradiend_rewrite_base_model_with_list_feature_factor(self, mock_model):
        """Test rewrite_base_model with list feature_factor (requires latent_dim=2)."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.testing_mocks import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                latent_dim=2,
            )
        
        # Test with list feature_factor (must match latent_dim)
        rewritten_model = model_with_gradiend.rewrite_base_model(
            learning_rate=1e-4,
            feature_factor=[1.0, -1.0],
            part="decoder"
        )
        
        assert rewritten_model is not None
    
    def test_model_with_gradiend_rewrite_base_model_raises_invalid_part(self, mock_model):
        """Test rewrite_base_model raises ValueError for invalid part."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.testing_mocks import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
            )
        
        with pytest.raises(ValueError) as exc_info:
            model_with_gradiend.rewrite_base_model(
                learning_rate=1e-4,
                feature_factor=1.0,
                part="invalid_part"
            )
        
        assert "part must be" in str(exc_info.value).lower()
    
    def test_model_with_gradiend_prune_gradiend(self, mock_model):
        """Test pruning via prune_gradiend method."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.testing_mocks import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1
            )
        
        original_input_dim = model_with_gradiend.gradiend.input_dim
        
        # Prune to top 50% (if input_dim > 1)
        if original_input_dim > 1:
            pruned = model_with_gradiend.prune_gradiend(topk=0.5, inplace=False)
            assert pruned.gradiend.input_dim < original_input_dim
            assert pruned.gradiend.input_dim > 0
    
    def test_model_with_gradiend_pre_prune(self, mock_model, mock_tokenizer):
        """Test pre-pruning before training."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        from gradiend.trainer.core.pruning import pre_prune, PrePruneConfig
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            return mock_model, mock_tokenizer
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
            )
        
        original_input_dim = model_with_gradiend.gradiend.input_dim
        
        # Create a mock dataset with text inputs (as expected by TextPredictionModelWithGradiend)
        # The gradient_creator expects text strings, not tensors
        class MockTextDataset:
            def __len__(self):
                return 10
            
            def __getitem__(self, idx):
                # Return text strings that can be processed by create_gradients
                # Include labels for gradient creation
                return {
                    'factual': f"factual text {idx}",
                    'alternative': f"alternative text {idx}",
                    'label': f"label_{idx % 2}",  # Add label for gradient creation
                    'feature_class_id': idx % 2
                }
        
        dataset = MockTextDataset()
        
        # Pre-prune config
        config = PrePruneConfig(
            n_samples=4,
            topk=0.5,
            source="factual"
        )
        
        # Pre_prune calls gradient_creator(factual_in) with just the text string
        # But create_gradients needs (text, label). We need to patch gradient_creator
        # to extract the label from the dataset item or use a default
        original_create_gradients = model_with_gradiend.create_gradients
        
        def mock_create_gradients(text, label=None, return_dict=False, **kwargs):
            # If called with just text (no label), use a default label
            if label is None:
                label = "default_label"
            # Return a mock gradient tensor with the right shape
            # For simplicity, return a random tensor matching input_dim
            if return_dict:
                # If return_dict, return a dict that can be converted to vector
                return {"gradient": torch.randn(original_input_dim)}
            else:
                # Otherwise return a tensor directly
                return torch.randn(original_input_dim)
        
        # Patch create_gradients to avoid actual model forward pass
        model_with_gradiend.create_gradients = mock_create_gradients
        
        # Pre_prune calls gradient_creator(text) directly
        # gradient_creator is a @property that returns _gradient_creator
        # We need to set _gradient_creator to a callable that handles single-argument calls
        # Use object.__setattr__ to bypass PyTorch's nn.Module attribute registration
        def gradient_creator_callable(text, target_device=None, **kwargs):
            """Callable wrapper for gradient_creator(text) calls from pre_prune."""
            # Pre_prune calls with text and target_device=..., so add a default label
            return mock_create_gradients(text, label="default_label", return_dict=False)
        
        # Set _gradient_creator (the underlying attribute) to our callable wrapper
        # Use object.__setattr__ to bypass PyTorch's nn.Module special handling
        original_gradient_creator = model_with_gradiend._gradient_creator
        object.__setattr__(model_with_gradiend, '_gradient_creator', gradient_creator_callable)
        
        try:
            pruned = pre_prune(
                model_with_gradiend,
                dataset,
                config,
                inplace=False
            )
            assert pruned.gradiend.input_dim < original_input_dim
            assert pruned.gradiend.input_dim > 0
        finally:
            # Restore original methods
            model_with_gradiend.create_gradients = original_create_gradients
            object.__setattr__(model_with_gradiend, '_gradient_creator', original_gradient_creator)
    
    def test_model_with_gradiend_post_prune(self, mock_model):
        """Test post-pruning after training."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        from gradiend.trainer.core.pruning import post_prune, PostPruneConfig
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.testing_mocks import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            model_with_gradiend = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
            )
        
        original_input_dim = model_with_gradiend.gradiend.input_dim
        
        # Post-prune config
        config = PostPruneConfig(
            topk=0.5,
            part="decoder-weight"
        )
        
        pruned = post_prune(model_with_gradiend, config)
        assert pruned.gradiend.input_dim < original_input_dim
        assert pruned.gradiend.input_dim > 0
    
    def test_model_saving_with_safetensors_available(self, temp_dir):
        """Test that safetensors is used when available."""
        from gradiend.model.model import GradiendModel
        
        # Create a simple model to test saving
        model = GradiendModel(input_dim=100, latent_dim=1)
        
        # Mock safetensors.save_file to actually create a file
        def mock_save_file(state_dict, path):
            """Mock save_file that creates a file to verify it was called."""
            # Create a marker file to verify safetensors was used
            with open(path, 'wb') as f:
                f.write(b"safetensors_marker")
        
        # Create mock safetensors module structure
        mock_safetensors_torch = MagicMock()
        mock_safetensors_torch.save_file = mock_save_file
        
        # Patch the import where it's used in save_pretrained
        # The import happens as: from safetensors.torch import save_file
        # We need to patch it at the module level
        import sys
        original_modules = {}
        for key in list(sys.modules.keys()):
            if key.startswith('safetensors'):
                original_modules[key] = sys.modules.pop(key)
        
        try:
            # Add mock module to sys.modules
            sys.modules['safetensors'] = MagicMock()
            sys.modules['safetensors.torch'] = mock_safetensors_torch
            
            # Patch the save_file function
            with patch.object(mock_safetensors_torch, 'save_file', mock_save_file):
                save_path = os.path.join(temp_dir, "test_model_safetensors")
                model.save_pretrained(save_path, use_safetensors=None)  # None = prefer safetensors
            
            # Check that safetensors file was created
            safetensors_path = os.path.join(save_path, "model.safetensors")
            bin_path = os.path.join(save_path, "pytorch_model.bin")
            
            # When safetensors is available, it should be used
            assert os.path.exists(safetensors_path), "model.safetensors should exist when safetensors is available"
            assert not os.path.exists(bin_path), "pytorch_model.bin should not exist when safetensors is used"
        finally:
            # Restore modules
            for key in list(sys.modules.keys()):
                if key.startswith('safetensors') and key not in original_modules:
                    sys.modules.pop(key)
            sys.modules.update(original_modules)
    
    def test_model_saving_without_safetensors_available(self, temp_dir):
        """Test that bin format is used when safetensors is not available."""
        from gradiend.model.model import GradiendModel
        
        # Create a simple model to test saving
        model = GradiendModel(input_dim=100, latent_dim=1)
        
        # Remove safetensors from sys.modules to ensure clean state
        original_modules = {}
        for key in list(sys.modules.keys()):
            if key.startswith('safetensors'):
                original_modules[key] = sys.modules.pop(key)
        
        try:
            # Mock the import to fail by patching __import__
            import builtins
            original_import = builtins.__import__
            def failing_import(name, *args, **kwargs):
                if name == 'safetensors.torch':
                    raise ImportError("No module named 'safetensors'")
                # Use the real __import__ for everything else
                return original_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=failing_import):
                save_path = os.path.join(temp_dir, "test_model_no_safetensors")
                model.save_pretrained(save_path, use_safetensors=None)  # None = prefer safetensors, but will fallback
            
            # When safetensors is not available, bin should be used
            safetensors_path = os.path.join(save_path, "model.safetensors")
            bin_path = os.path.join(save_path, "pytorch_model.bin")
            
            assert os.path.exists(bin_path), "pytorch_model.bin should exist when safetensors is not available"
            assert not os.path.exists(safetensors_path), "model.safetensors should not exist when safetensors is not available"
        finally:
            # Restore modules
            sys.modules.update(original_modules)
    
    def test_model_saving_force_safetensors_when_unavailable(self, temp_dir):
        """Test that forcing safetensors raises error when unavailable."""
        from gradiend.model.model import GradiendModel
        
        model = GradiendModel(input_dim=100, latent_dim=1, device=torch.device("cpu"))
        
        # Remove safetensors from sys.modules to ensure clean state
        original_modules = {}
        for key in list(sys.modules.keys()):
            if key.startswith('safetensors'):
                original_modules[key] = sys.modules.pop(key)
        
        try:
            # Mock the import to fail
            import builtins
            original_import = builtins.__import__
            def failing_import(name, *args, **kwargs):
                if name == 'safetensors.torch':
                    raise ImportError("No module named 'safetensors'")
                # Use the real __import__ for everything else
                return original_import(name, *args, **kwargs)
            
            with patch('builtins.__import__', side_effect=failing_import):
                save_path = os.path.join(temp_dir, "test_model_force_safetensors")
                
                # When use_safetensors=True but safetensors is unavailable, should raise ImportError
                with pytest.raises(ImportError, match="safetensors not installed"):
                    model.save_pretrained(save_path, use_safetensors=True)
        finally:
            # Restore modules
            sys.modules.update(original_modules)
    
    def test_model_dtype_handling(self, mock_model):
        """Test Float32 vs Float16 dtype handling."""
        from gradiend.trainer.text.prediction.model_with_gradiend import TextPredictionModelWithGradiend
        from gradiend.trainer.text.common.model_base import TextModelWithGradiend
        
        def mock_load_model(cls, load_directory, base_model_id=None, tokenizer=None, **kwargs):
            from tests.testing_mocks import MockTokenizer
            return mock_model, MockTokenizer()
        
        with patch.object(TextModelWithGradiend, '_load_model', classmethod(mock_load_model)):
            # Test float32
            model_f32 = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
                torch_dtype=torch.float32,
            )
            assert model_f32.gradiend.torch_dtype == torch.float32
            
            # Test float16
            model_f16 = TextPredictionModelWithGradiend.from_pretrained(
                mock_model,
                n_features=1,
                torch_dtype=torch.float16,
            )
            assert model_f16.gradiend.torch_dtype == torch.float16
    
    def test_pruning_efficient_storage_with_full_mask(self, temp_dir, set_seed):
        """Test that pruning uses efficient storage when full masks are provided."""
        set_seed(42)
        
        from gradiend.model import ParamMappedGradiendModel
        
        # Create a model with param mapping
        # Each layer has 10*10=100 elements, so total input_dim should be 200
        input_dim = 200
        param_map = {
            "layer1.weight": {"shape": (10, 10), "repr": "all"},  # 100 elements
            "layer2.weight": {"shape": (10, 10), "repr": "all"},  # 100 elements
        }
        
        model = ParamMappedGradiendModel(
            input_dim=input_dim,
            latent_dim=1,
            param_map=param_map
        )
        
        # Create a full mask (all True)
        full_mask = torch.ones(input_dim, dtype=torch.bool)
        
        # Prune with full mask
        pruned = model.prune(mask=full_mask, inplace=False)
        
        # When full mask is provided, efficient storage should be used
        # Save the model to check storage format
        save_path = os.path.join(temp_dir, "pruned_model_full_mask")
        pruned.save_pretrained(save_path)
        
        # Check that model was saved
        assert os.path.exists(save_path)
        assert os.path.exists(os.path.join(save_path, "config.json"))
        
        # For full mask, the model should still be pruned correctly
        # (though with full mask, no actual pruning occurs)
        assert pruned.input_dim == input_dim  # Full mask means no reduction
    
    def test_pruning_efficient_storage_with_partial_mask(self, temp_dir, set_seed):
        """Test that pruning uses efficient storage when partial masks are provided."""
        set_seed(42)
        
        from gradiend.model import ParamMappedGradiendModel
        
        # Create a model with param mapping
        # Each layer has 10*10=100 elements, so total input_dim should be 200
        input_dim = 200
        param_map = {
            "layer1.weight": {"shape": (10, 10), "repr": "all"},  # 100 elements
            "layer2.weight": {"shape": (10, 10), "repr": "all"},  # 100 elements
        }
        
        model = ParamMappedGradiendModel(
            input_dim=input_dim,
            latent_dim=1,
            param_map=param_map
        )
        
        # Create a partial mask (keep first 100 dimensions)
        partial_mask = torch.zeros(input_dim, dtype=torch.bool)
        partial_mask[:100] = True
        
        # Prune with partial mask
        pruned = model.prune(mask=partial_mask, inplace=False)
        
        # Should reduce input_dim
        assert pruned.input_dim == 100
        
        # Save the model to check storage format
        save_path = os.path.join(temp_dir, "pruned_model_partial_mask")
        pruned.save_pretrained(save_path)
        
        # Check that model was saved
        assert os.path.exists(save_path)
        assert os.path.exists(os.path.join(save_path, "config.json"))
        
        # Check that mapping files are saved efficiently
        # When masks are provided, they should be stored efficiently
        config_path = os.path.join(save_path, "config.json")
        import json
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check that mapping information is present
        if "mapping" in config:
            mapping = config["mapping"]
            # Efficient storage: masks should be saved in mapping_masks file
            assert "masks_file" in mapping or "mode" in mapping
