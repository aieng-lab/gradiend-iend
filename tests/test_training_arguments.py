"""
Tests for TrainingArguments (modality-independent).

Tests parameter validation, serialization, and override behavior.
"""

import json
import os

import pytest
import torch

from gradiend.trainer.core.signals import Signal, SignalScope, SignalSet
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.gradiend_split import GradiendSplit


class TestTrainingArguments:
    """Test TrainingArguments validation and serialization."""
    
    def test_training_arguments_validation(self):
        """Test that invalid values raise errors."""
        # Invalid source
        with pytest.raises(ValueError, match="source must be one of"):
            TrainingArguments(source="invalid")
        
        # Invalid target
        with pytest.raises(ValueError, match="target must be one of"):
            TrainingArguments(target="invalid")
        
        # Note: eval_strategy, save_strategy, and optim are not currently validated
        # They are just strings that are used as-is. If validation is added later,
        # these tests should be updated.

    def test_include_other_classes_defaults_false(self):
        args = TrainingArguments()
        assert args.include_other_classes is False

    def test_encoder_bias_defaults_true(self):
        args = TrainingArguments()
        assert args.bias_encoder is True

    def test_mask_placeholder_is_serialized_and_validated(self):
        args = TrainingArguments(mask_placeholder="[PRONOUN]")

        assert args.mask_placeholder == "[PRONOUN]"
        assert TrainingArguments.from_dict(args.to_dict()).mask_placeholder == "[PRONOUN]"
        with pytest.raises(TypeError, match="mask_placeholder"):
            TrainingArguments(mask_placeholder=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="mask_placeholder"):
            TrainingArguments(mask_placeholder="")

    def test_training_arguments_default_signal_is_gradient(self):
        args = TrainingArguments()

        assert args.signal == Signal.gradient()
        assert isinstance(args.signals, SignalSet)
        assert args.signals.ids == ("gradient",)

    def test_training_arguments_accepts_single_signal(self):
        signal = Signal.activation(name="activation", token_selector="mask")

        args = TrainingArguments(signal=signal)

        assert args.signal == signal
        assert args.signals.ids == ("activation",)
        assert args.signals.single == signal

    def test_training_arguments_accepts_signal_string(self):
        args = TrainingArguments(signal="gradient")

        assert args.signal == Signal.gradient()
        assert args.signals.ids == ("gradient",)

    def test_training_arguments_accepts_multi_signal_set(self):
        signals = SignalSet(
            Signal.gradient(name="gradient"),
            Signal.activation(name="activation", token_selector="mask"),
        )

        args = TrainingArguments(signals=signals)

        assert args.signal is None
        assert args.signals.ids == ("gradient", "activation")

    def test_training_arguments_rejects_conflicting_signal_and_signals(self):
        with pytest.raises(ValueError, match="Pass either signal"):
            TrainingArguments(
                signal=Signal.gradient(name="gradient"),
                signals=SignalSet(Signal.activation(name="activation")),
            )
    
    def test_training_arguments_from_dict(self):
        """Test creation from dict."""
        config_dict = {
            "learning_rate": 1e-4,
            "max_steps": 100,
            "train_batch_size": 16,
            "source": "factual",
            "target": "diff",
            "bias_encoder": False,
            "bias_decoder": False,
            "init_fan_in_floor": 2048,
        }

        args = TrainingArguments.from_dict(config_dict)
        assert args.learning_rate == 1e-4
        assert args.max_steps == 100
        assert args.train_batch_size == 16
        assert args.source == "factual"
        assert args.target == "diff"
        assert args.bias_encoder is False
        assert args.bias_decoder is False
        assert args.init_fan_in_floor == 2048
    
    def test_training_arguments_serialization(self, temp_dir):
        """Test JSON serialization/deserialization."""
        args = TrainingArguments(
            learning_rate=1e-4,
            max_steps=100,
            train_batch_size=16,
            source="factual",
            target="diff",
            experiment_dir=temp_dir,
            bias_encoder=False,
            bias_decoder=False,
        )
        
        # Test to_dict
        config_dict = args.to_dict()
        assert isinstance(config_dict, dict)
        assert config_dict["learning_rate"] == 1e-4
        assert config_dict["max_steps"] == 100
        assert config_dict["bias_encoder"] is False
        assert config_dict["bias_decoder"] is False
        assert config_dict["init_fan_in_floor"] == 10_000
        
        # Test JSON serialization
        json_path = os.path.join(temp_dir, "config.json")
        with open(json_path, 'w') as f:
            json.dump(config_dict, f)
        
        # Test loading from JSON
        with open(json_path, 'r') as f:
            loaded_dict = json.load(f)
        
        loaded_args = TrainingArguments.from_dict(loaded_dict)
        assert loaded_args.learning_rate == args.learning_rate
        assert loaded_args.max_steps == args.max_steps
        assert loaded_args.train_batch_size == args.train_batch_size
        assert loaded_args.bias_encoder is False
        assert loaded_args.bias_decoder is False
        assert loaded_args.init_fan_in_floor == args.init_fan_in_floor

    def test_training_arguments_validates_init_fan_in_floor(self):
        assert TrainingArguments(init_fan_in_floor=None).init_fan_in_floor is None

        with pytest.raises(TypeError, match="init_fan_in_floor"):
            TrainingArguments(init_fan_in_floor=1.5)
        with pytest.raises(TypeError, match="init_fan_in_floor"):
            TrainingArguments(init_fan_in_floor=True)
        with pytest.raises(ValueError, match="init_fan_in_floor"):
            TrainingArguments(init_fan_in_floor=0)

    def test_training_arguments_signal_serialization(self, temp_dir):
        """Signal config should be JSON-serializable and round-trip."""
        args = TrainingArguments(
            signals=SignalSet(
                Signal.gradient(name="gradient"),
                Signal.activation(name="activation", token_selector="mask"),
            ),
            signal_scope=SignalScope.from_values(activation_sites=["emb", "encoder.layer.*"]),
        )

        config_dict = args.to_dict()
        assert config_dict["signal"] is None
        assert config_dict["signals"] == [
            {"kind": "gradient", "name": "gradient", "options": {}},
            {"kind": "activation", "name": "activation", "options": {"token_selector": "mask"}},
        ]
        assert config_dict["signal_scope"] == {
            "params": None,
            "activation_sites": ["emb", "encoder.layer.*"],
        }

        json_path = os.path.join(temp_dir, "signal_config.json")
        with open(json_path, "w") as f:
            json.dump(config_dict, f)
        with open(json_path, "r") as f:
            loaded_dict = json.load(f)

        loaded_args = TrainingArguments.from_dict(loaded_dict)
        assert loaded_args.signal is None
        assert loaded_args.signals.ids == ("gradient", "activation")
        assert loaded_args.signals["activation"].options == {"token_selector": "mask"}
        assert loaded_args.signal_scope == SignalScope.from_values(
            activation_sites=["emb", "encoder.layer.*"],
        )

    def test_training_arguments_gradiend_split_serialization(self):
        args = TrainingArguments(
            gradiend_split=GradiendSplit.by_tensor(),
            gradiend_split_loss="size_weighted",
        )

        assert args.gradiend_split == GradiendSplit.by_tensor()
        assert args.gradiend_split_loss == "size_weighted"
        config_dict = args.to_dict()
        assert config_dict["gradiend_split"] == {"mode": "tensors"}
        assert config_dict["gradiend_split_loss"] == "size_weighted"

        loaded_args = TrainingArguments.from_dict(config_dict)
        assert loaded_args.gradiend_split == GradiendSplit.by_tensor()
        assert loaded_args.gradiend_split_loss == "size_weighted"

    def test_training_arguments_validates_gradiend_split_loss(self):
        with pytest.raises(ValueError, match="gradiend_split_loss"):
            TrainingArguments(gradiend_split_loss="median")

    def test_training_arguments_params_is_deprecated_signal_scope_alias(self):
        with pytest.warns(DeprecationWarning, match="params is deprecated"):
            args = TrainingArguments(params=["encoder.*"])

        assert args.signal_scope == SignalScope.from_values(params=["encoder.*"])

    def test_gradient_signal_accepts_layers_scope_but_rejects_raw_activation_sites(self):
        """SignalScope's semantic shortcuts (.layers()/.layer()/.embeddings()/
        .word_embedding(), i.e. activation_selector) are scope metadata
        independent of signal kind -- for a gradient signal they now resolve
        into weight-parameter wildcards via gradient_params_from_selector()
        (gradiend/signal_space.py), using the same ModelTopology the
        activation-signal path already used. This was originally a silent
        no-op instead (a GRADIEND run configured with SignalScope.layers()
        trained against the package's plain default scope, embeddings
        included, with no error anywhere -- caught 2026-08-20 via
        gradiend-sae); construction-time resolvability can't be checked here
        since no model is loaded yet, but it must not be rejected up front.

        Raw activation_sites=[...] (an explicit module-path include-list, not
        one of the semantic shortcuts) is a different case: there is no
        generic, safe way to turn an arbitrary caller-supplied module pattern
        into the right parameter-name wildcard, so it is still rejected.
        """
        # No longer raises -- resolved later, at model-construction time.
        TrainingArguments(signal=Signal.gradient(), signal_scope=SignalScope.layers())
        TrainingArguments(signal=Signal.gradient(), signal_scope=SignalScope.layer(2))
        TrainingArguments(signal=Signal.gradient(), signal_scope=SignalScope.embeddings())

        with pytest.raises(ValueError, match="activation_sites"):
            TrainingArguments(
                signal=Signal.gradient(),
                signal_scope=SignalScope.from_values(activation_sites=["encoder.layer.*"]),
            )

        # Sanity: the pre-existing legitimate combinations still construct fine.
        TrainingArguments(
            signal=Signal.gradient(), signal_scope=SignalScope.from_values(params=["encoder.*"])
        )
        TrainingArguments(signal=Signal.activation(), signal_scope=SignalScope.layers())

    def test_training_arguments_override(self):
        """Test parameter override behavior."""
        args = TrainingArguments(
            learning_rate=1e-5,
            max_steps=50,
            train_batch_size=32
        )
        
        # Override via kwargs
        args.learning_rate = 1e-4
        assert args.learning_rate == 1e-4
        
        # Override via setattr
        setattr(args, 'max_steps', 100)
        assert args.max_steps == 100
    
    def test_training_arguments_post_init(self):
        """Test __post_init__ validation."""
        # Test that post_init sets defaults correctly
        args = TrainingArguments()
        
        # Check that defaults are set (source default is "alternative" per arguments.py)
        assert args.source == "alternative"
        assert args.target == "diff"
        assert args.train_batch_size == 32
        assert args.base_gradient_batch_size == 32
        assert args.gradiend_batch_size == 1
        assert args.precompute_gradient_batches is None
        assert args.precompute_gradient_buffer_size == 1
        assert args.gradient_timing_steps == 0
        assert args.runtime_monitor is False
        assert args.runtime_monitor_interval == 5.0
        assert args.runtime_monitor_system_stats is True
        assert args.learning_rate == 1e-5
        assert args.convergent_score_threshold == 0.5
        assert args.convergent_mean_by_class_threshold == 0.5
        assert args.prefer_convergent_checkpoint is False
        assert args.base_model_device_map is None
        assert args.base_model_max_memory is None

        args_device_map = TrainingArguments(base_model_device_map="auto")
        assert args_device_map.base_model_device_map == "auto"

        args_no_device_map = TrainingArguments(base_model_device_map=False)
        assert args_no_device_map.base_model_device_map is False

        args_max_memory = TrainingArguments(
            base_model_device_map="auto",
            base_model_max_memory={0: "0GiB", 1: "70GiB"},
        )
        assert args_max_memory.base_model_max_memory == {0: "0GiB", 1: "70GiB"}

        args_split_batch = TrainingArguments(
            train_batch_size=8,
            base_gradient_batch_size=1,
            gradiend_batch_size=4,
        )
        assert args_split_batch.train_batch_size == 8
        assert args_split_batch.base_gradient_batch_size == 1
        assert args_split_batch.gradiend_batch_size == 4

        args_train_batch_alias = TrainingArguments(train_batch_size=8)
        assert args_train_batch_alias.train_batch_size == 8
        assert args_train_batch_alias.base_gradient_batch_size == 8
        assert args_train_batch_alias.gradiend_batch_size == 1
    
    def test_training_arguments_dtype_handling(self):
        """Test torch_dtype parameter handling."""
        # Test None (defaults to float32)
        args1 = TrainingArguments()
        assert args1.torch_dtype is None or args1.torch_dtype == torch.float32
        
        # Test explicit float32
        args2 = TrainingArguments(torch_dtype=torch.float32)
        assert args2.torch_dtype == torch.float32
        
        # Test float16
        args3 = TrainingArguments(torch_dtype=torch.float16)
        assert args3.torch_dtype == torch.float16
    
    def test_training_arguments_pre_prune_config(self):
        """Test PrePruneConfig integration."""
        from gradiend.trainer.core.pruning import PrePruneConfig
        
        pre_prune_config = PrePruneConfig(
            n_samples=100,
            topk=0.5,
            source="factual"
        )
        
        args = TrainingArguments(pre_prune_config=pre_prune_config)
        assert args.pre_prune_config is not None
        assert args.pre_prune_config.n_samples == 100
        assert args.pre_prune_config.topk == 0.5

    def test_training_arguments_convergence_plot_flags(self):
        args = TrainingArguments(
            reuse_pre_prune=True,
            fail_on_non_convergence=True,
            highlight_non_convergence=False,
        )
        assert args.reuse_pre_prune is True
        assert args.fail_on_non_convergence is True
        assert args.highlight_non_convergence is False
    
    def test_training_arguments_post_prune_config(self):
        """Test PostPruneConfig integration."""
        from gradiend.trainer.core.pruning import PostPruneConfig
        
        post_prune_config = PostPruneConfig(
            topk=0.5,
            part="decoder-weight"
        )
        
        args = TrainingArguments(post_prune_config=post_prune_config)
        assert args.post_prune_config is not None
        assert args.post_prune_config.topk == 0.5
        assert args.post_prune_config.part == "decoder-weight"
    
    def test_training_arguments_seed_handling(self):
        """Test seed parameter handling."""
        # Default seed is 0 for reproducibility
        args1 = TrainingArguments()
        assert args1.seed == 0
        assert args1.seed_stability_topk == 1000
        assert args1.seed_stability_part == "decoder-weight"

        # Explicit None for non-deterministic runs
        args_none = TrainingArguments(seed=None)
        assert args_none.seed is None

        # Test explicit seed
        args2 = TrainingArguments(seed=42)
        assert args2.seed == 42

        # Test max_seeds
        args3 = TrainingArguments(max_seeds=5)
        assert args3.max_seeds == 5

        args4 = TrainingArguments(seed_stability_topk=128, seed_stability_part="encoder-weight")
        assert args4.seed_stability_topk == 128
        assert args4.seed_stability_part == "encoder-weight"

        args5 = TrainingArguments(analyze_seed_stability=True)
        assert args5.analyze_seed_stability is True
        assert args5.saved_seed_runs == "all_convergent"

        args6 = TrainingArguments(split_resplit_per_seed=True, split_resplit_strategy="balanced_cycle")
        assert args6.split_resplit_strategy == "balanced_cycle"
        with pytest.raises(ValueError, match="split_resplit_strategy"):
            TrainingArguments(split_resplit_strategy="weighted")
    
    def test_training_arguments_output_paths(self):
        """Test output directory handling."""
        args = TrainingArguments(
            experiment_dir="/path/to/experiment",
            output_dir="/path/to/output"
        )
        
        assert args.experiment_dir == "/path/to/experiment"
        assert args.output_dir == "/path/to/output"
        
        # Test use_cache
        args.use_cache = True
        assert args.use_cache is True
