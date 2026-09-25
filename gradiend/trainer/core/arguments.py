"""
Training arguments for GRADIEND Trainer (HF-like API).
"""

import dataclasses
import warnings
from dataclasses import dataclass, field
from typing import Literal, Optional, Callable, Union, Any, List, Dict

import torch
import torch.nn as nn

from gradiend.trainer.core.cache_policy import normalize_use_cache
from gradiend.trainer.core.metric_names import normalize_metric_name
from gradiend.trainer.core.config import validate_source_target, validate_source_target_combination
from gradiend.trainer.core.pruning import PostPruneConfig, PrePruneConfig, _validate_topk
from gradiend.trainer.core.signals import (
    Signal,
    SignalScope,
    SignalSet,
    coerce_signal,
    coerce_signal_scope,
    coerce_signal_set,
    normalize_signal_arguments,
)
from gradiend.gradiend_split import GradiendSplit, coerce_gradiend_split


def dtype_to_name(dtype: torch.dtype) -> str:
    """Serialize a torch dtype as its bare name ("bfloat16", not "torch.bfloat16").

    ``str(torch.bfloat16)`` returns ``"torch.bfloat16"``, which no longer
    resolves through ``getattr(torch, name)``.  Writing the prefixed form and
    reading it back silently produced ``float32``, so a configured
    ``torch_dtype=bfloat16`` was lost on every ``to_dict()``/``from_dict()``
    round trip -- and :class:`TextPredictionTrainer` round-trips its arguments
    on construction, so no trainer ever saw the requested dtype.
    """
    if not isinstance(dtype, torch.dtype):
        raise TypeError(f"dtype_to_name expects a torch.dtype, got {type(dtype).__name__}")
    return str(dtype).split(".")[-1]


def dtype_from_name(name: Any) -> torch.dtype:
    """Resolve a serialized dtype name back to the torch dtype.

    Accepts both the bare name and the legacy ``"torch."``-prefixed form so
    configs and caches written before :func:`dtype_to_name` still load.  An
    unknown name raises instead of falling back to ``float32``: a silent
    fallback is what hid the round-trip loss described above.
    """
    if isinstance(name, torch.dtype):
        return name
    if not isinstance(name, str):
        raise TypeError(f"dtype_from_name expects a str or torch.dtype, got {type(name).__name__}")
    resolved = getattr(torch, name.split(".")[-1], None)
    if not isinstance(resolved, torch.dtype):
        raise ValueError(f"Unknown torch dtype name {name!r}")
    return resolved


@dataclass
class TrainingArguments:
    """
    Arguments for GRADIEND training (HF Trainer–style, single training_args class).

    Pass to Trainer at construction: Trainer(model=..., training_args=TrainingArguments(...)).
    Used directly by the core training loop.
    """

    # ----- Output -----
    experiment_dir: Optional[str] = None
    """Root directory for this experiment. When set, default paths use subpaths under it (model, encoded_values, etc.). One experiment dir holds one model. Trainer.run_id (when set) is used as subdir under this."""

    output_dir: Optional[str] = None
    """Directory to save the trained model. If None and experiment_dir is set, uses experiment_dir/model (or experiment_dir/run_id/model when Trainer.run_id is set). Otherwise must be set explicitly."""

    use_cache: Union[bool, Literal["always", "only_convergent"]] = False
    """Training checkpoint reuse policy.

    - ``False``: always retrain even when a saved model exists.
    - ``True``: reuse when a saved model exists and matches the training cache fingerprint.
    - ``"always"``: reuse any saved model at the output path (skip fingerprint matching).
    - ``"only_convergent"``: same fingerprint check as ``True``, but only when the saved run
      meets ``min_convergent_seeds`` (per-seed convergence for individual seed dirs;
      aggregate count for the selected model).
    """

    add_identity_for_other_classes: bool = False
    """If True, add identity (factual==alternative) examples for classes not in the target classes used for training."""

    add_neutral_identity_transitions: Optional[bool] = None
    """Whether to add neutral identity transitions from ``TextPredictionConfig.neutral_data``.

    These rows have factual==alternative and label 0. For target='diff' they
    train the decoded GRADIEND/ACTIEND update toward the zero vector on neutral
    examples without adding a separate neutral-specific loss.

    - ``None`` (default): enabled automatically when ``neutral_data`` is configured,
      otherwise disabled.
    - ``True``: always enabled; training fails if no ``neutral_data`` is configured.
    - ``False``: never used, even if ``neutral_data`` is configured.
    """

    # ----- GRADIEND interpretation -----
    source: str = "alternative"
    """Source for GRADIEND input: 'factual', 'alternative', 'diff', or 'both'.

    ``both`` alternates factual/alternative poles across balance-group *visits*
    (orthogonal to ``feature_class_id`` / neutral balance cycling) and requires
    ``target='diff'``.
    """

    target: str = "diff"
    """Target for GRADIEND output: 'factual', 'alternative', or 'diff'."""

    # ----- Training loop -----
    train_batch_size: int = 32
    """Alias/default for base_gradient_batch_size. Does not affect gradiend_batch_size."""

    base_gradient_batch_size: Optional[int] = None
    """Number of raw training examples merged into one base-model loss/backward call, producing one base-gradient vector."""

    gradiend_batch_size: Optional[int] = None
    """Number of base-gradient vectors stacked into one GRADIEND optimizer step. Defaults to 1."""

    precompute_gradient_batches: Optional[bool] = None
    """Whether to precompute the next gradient row asynchronously.
    None (default): auto-enable only when the base model is sharded and multiple CUDA devices
    are available. False: never precompute. True: always precompute (thread-safe via
    ModelWithGradiend.exclusive_base_gradient_access during base forward/backward)."""

    precompute_gradient_buffer_size: int = 1
    """Number of already-computed gradient rows to keep in the asynchronous precompute queue."""

    gradient_timing_steps: int = 0
    """If > 0, log timing for gradient-row creation every N rows."""

    runtime_monitor: bool = False
    """If True, write persistent JSONL runtime diagnostics under the training output directory."""

    runtime_monitor_interval: float = 5.0
    """Seconds between runtime monitor heartbeat samples."""

    runtime_monitor_system_stats: bool = True
    """If True, runtime monitor heartbeats include CPU/GPU memory stats."""

    train_max_size: Optional[int] = None
    """If set, cap training samples per feature_class_id (downsampling). 
    
    Note: Balancing is handled automatically by the dataset scheduler via oversampling (cycling through 
    balance groups). This parameter primarily reduces total dataset size for memory/performance. 
    None = use all data."""

    learning_rate: float = 1e-5
    """Peak learning rate."""

    learning_rate_decoder: Optional[Union[float, Literal["auto", "default"]]] = "default"
    """Optional separate learning rate for the decoder.

    - ``"default"`` (the default) resolves per signal kind at construction:
      ``"auto"`` for pure activation signals (ACTIEND) and ``None`` otherwise.
    - ``None``: encoder and decoder share ``learning_rate`` in one optimizer group
      (the classic GRADIEND behavior).
    - a float: the decoder gets its own parameter group with this rate.
    - ``"auto"``: the decoder starts at ``learning_rate``; at the first
      ``eval_steps`` boundary it is raised to a model-local reachability floor.
      This needs training to run *past* the first ``eval_steps`` boundary, an
      Adam/AdamW optimizer, an identity decoder and an ``MSELoss`` criterion, and
      it sets the decoder weight decay to 0. The estimate reuses Adam's decoder
      first moments and stores only latent-sized statistics, so it adds no
      decoder-sized memory and no extra pass.

    Why a separate decoder rate exists: Adam-family updates are scale-free, so the
    decoder can move at most about ``lr * steps * sqrt(output_dim)``. When the
    decoder optimum is far larger than the encoder's (ACTIEND reconstructs raw
    activation differences), an encoder-tuned rate cannot reach it. Raising only
    the decoder rate removes that limit without changing encoder optimization.

    Incompatible with ``supervised_encoder=True`` (no decoder parameter is trained).
    Note that ``"default"`` is resolved once in ``__post_init__``; when copying
    arguments with ``dataclasses.replace(args, signal=...)`` pass
    ``learning_rate_decoder="default"`` again to re-resolve it for the new signal.
    """

    num_train_epochs: int = 3
    """Number of training epochs."""

    max_steps: int = -1
    """If > 0, total number of steps; overrides num_train_epochs. -1 = use epochs."""

    weight_decay: float = 1e-2
    """Weight decay for the optimizer."""

    adam_epsilon: float = 1e-8
    """Epsilon for Adam/AdamW."""

    optim: str = "adamw"
    """Optimizer: 'adamw', 'adam', or 'sgd'.

    'sgd' exists to test whether a decoder-reachability shortfall is specific to
    Adam-family updates. Adam's per-coordinate step is scale-free (bounded by
    approximately ``learning_rate`` regardless of gradient magnitude), so total
    displacement over training is bounded by ``lr * steps * sqrt(dim)``. Plain
    SGD has no such bound -- its displacement scales with the gradients
    themselves -- so the two optimizers make different predictions about whether
    a large-norm decoder optimum is reachable within a fixed budget.
    """

    sgd_momentum: float = 0.0
    """Momentum for ``optim='sgd'``. Defaults to 0.0, i.e. plain gradient
    descent, which is the case the reachability analysis actually covers."""

    criterion: Optional[Union[nn.Module, Any]] = field(default=None, repr=False)
    """Loss function; None = MSELoss()."""

    # ----- Evaluation -----
    eval_strategy: str = "steps"
    """When to run evaluation: 'steps' (every eval_steps) or 'no'."""

    eval_steps: int = 250
    """Run evaluation every eval_steps (when eval_strategy == 'steps')."""

    encoder_eval_train_max_size: Optional[int] = None
    """Max samples for encoder evaluation during training (fast estimate; per-feature_class when available). None = use encoder_eval_max_size."""

    encoder_eval_max_size: Optional[int] = None
    """Max samples for encoder evaluation outside training (e.g. analysis, manual evaluate_encoder). None = use all available."""

    encoder_eval_balance: bool = True
    """If True, balance encoder evaluation data per feature_class_id. If False, use natural class distribution."""

    include_other_classes: bool = False
    """If True, encoder evaluation includes all class transitions in the split, not just the trained target pair. Applies when ``all_classes`` has more than two entries. Affects encoder metrics, encoder plots, and suite cross-encoding plots built from encoder evaluation. Set on [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments] or pass ``include_other_classes=True`` to ``evaluate_encoder()``."""

    seed_selection_eval_max_size: Optional[int] = None
    """Max samples for encoder evaluation when selecting the best seed. None = use encoder_eval_max_size."""

    decoder_eval_max_size_training_like: Optional[int] = None
    """Max samples for decoder training-like evaluation data. None = use default behavior."""

    decoder_eval_max_size_neutral: Optional[int] = None
    """Max samples for decoder neutral evaluation data (also LMS text cap). None = use default behavior."""

    decoder_eval_lrs: Optional[List[float]] = None
    """Learning rates for decoder grid search. None = DecoderEvaluator defaults (1/2/5 grid from 100 to 1e-3)."""

    decoder_eval_feature_factors: Optional[List[float]] = None
    """Feature factors for decoder grid search. None = derive from trainer target classes."""

    eval_batch_size: int = 32
    """Batch size for evaluation."""

    max_length: int = 128
    """Max token length for text training / ACTIEND filled templates.

    Longer inputs are truncated to keep ``[MASK]`` inside this window (left context
    dropped so the mask sits near the end). Lower values are cheaper; raise for
    long-context tasks if needed.
    """

    do_eval: bool = True
    """Whether to run evaluation during training."""

    evaluate_fn: Optional[Callable] = field(default=None, repr=False)
    """Custom evaluation function; None = default (encoder correlation on eval data)."""

    # ----- Checkpointing / saving -----
    save_strategy: str = "best"
    """'best' (default): keep only best checkpoint by correlation. 'steps': also save periodic checkpoints every save_steps. 'no': no checkpointing."""

    save_steps: int = 5000
    """Save checkpoint every save_steps when save_strategy == 'steps'."""

    save_only_best: bool = True
    """If True, keep only the best checkpoint (by evaluation correlation)."""

    delete_models: bool = False
    """If True, delete intermediate model files at end (e.g. .bin). This can be used to save disk space if you only care about metrics and not the model itself. Does not delete the whole model directory, which may contain other files (e.g. config, pre/post-prune results)."""

    # ----- Model / GRADIEND -----
    trust_remote_code: bool = False
    """If True, pass trust_remote_code=True when loading models/tokenizers from Hugging Face (e.g. for EuroBERT)."""

    dataset_trust_remote_code: Optional[bool] = None
    """Optional trust_remote_code value for HuggingFace datasets.load_dataset. None means do not pass the keyword."""

    model_use_cache: bool = False
    """When False (default), pass use_cache=False to decoder model forward during training (KV cache disabled).
    Use True only for inference/generation. Decoder-only MLM head training respects this via train_decoder_only_mlm_head."""

    label_token_protocol: str = "canonical"
    """Which token id a decoder-only *training* item labels the prediction with.

    ``"canonical"`` (default for new runs): the leading-space variant of the label
    (``▁she`` / ``Ġshe``), i.e. ``tokenizer(" " + label)`` -- the rule
    :meth:`TextPredictionModelWithGradiend.create_inputs` always used, so training
    items and single-row scoring agree. The label is written onto the last prefix
    token (the template's trailing space) and predicted from the token before it,
    so the natural next token is the leading-space variant.

    ``"legacy"``: the pre-2026-09-25 dataset rule, ``vocab[label]`` i.e. the
    no-space variant (``she``), which the model almost never predicts after a
    word boundary and which disagrees with ``create_inputs``.

    Checkpoints/argument dicts that do not carry this key deserialize as
    ``"legacy"`` (see :meth:`from_dict`), so already trained artifacts are never
    silently re-interpreted."""

    prediction_objective: str = "auto"
    """Prediction objective for text-gradient training and decoder probability scoring.
    Supported: ``auto``, ``mlm_mask_token``, ``clm_next_token``, ``clm_mlm_head``,
    ``clm_sequence_cloze``, ``seq2seq_decoder`` (experimental), ``seq2seq_decoder_sequence_cloze`` (experimental),
    ``seq2seq_encoder_mlm``.
    ``auto``: seq2seq models → ``seq2seq_encoder_mlm``; decoder-only → ``clm_next_token`` (or cached
    ``clm_mlm_head`` when a saved head exists); else ``mlm_mask_token``."""

    mask_placeholder: str = "[MASK]"
    """Dataset-level prediction placeholder used inside masked text templates.

    This is independent of ``tokenizer.mask_token``. Prefer setting
    ``TextPredictionConfig.mask_placeholder`` with the data schema (e.g.
    ``mask_placeholder=\"[PRONOUN]\"``). A non-default value here still overrides
    when the config field is left at its default.
    """

    decoder_mlm_head_epochs: int = 5
    """Epochs used when prediction_objective="clm_mlm_head" has to train the auxiliary head."""

    decoder_mlm_head_batch_size: int = 4
    """Batch size used when prediction_objective="clm_mlm_head" trains the auxiliary head."""

    decoder_mlm_head_lr: float = 1e-4
    """Learning rate used when prediction_objective="clm_mlm_head" trains the auxiliary head."""

    decoder_mlm_head_max_size: Optional[int] = None
    """Optional per-label cap for auxiliary decoder MLM-head training data."""

    decoder_sequence_cloze_rhs_window: int = -1
    """Right-context token window for clm_sequence_cloze / seq2seq_decoder_sequence_cloze scoring and training. -1 uses the full RHS."""

    params: Optional[List[str]] = None
    """If set, only these parameter names or wildcards are included in the GRADIEND param map when building from a base model. None = include all backbone parameters (default). Enables future params selection processes."""

    signal: Optional[Union[Signal, str, Dict[str, Any]]] = None
    """What is measured from the base model (``Signal.gradient()`` by default; also ``Signal.activation(...)``
    and ``Signal.activation_gradient(...)``). *Where* it is measured is ``signal_scope``; how the resolved
    space is partitioned is ``gradiend_split``. Strings (``"gradient"``, ``"activation"``) and dicts are coerced."""

    signals: Optional[Union[SignalSet, Signal, List[Any], Dict[str, Any]]] = None
    """Optional ``SignalSet`` / sequence of signals. Reserved for multi-signal training (currently exactly one signal is
    supported). Passing both ``signal`` and ``signals`` is only allowed when they describe the same single signal."""

    signal_scope: Optional[Union[SignalScope, Dict[str, Any]]] = None
    """Optional ``SignalScope`` describing where the signal is measured (parameters for gradient signals, module
    sites for activation signals). ``None`` uses the default backbone scope."""

    gradiend_split: Optional[Union[GradiendSplit, str, Dict[str, Any]]] = None
    """Optional GradiendSplit describing component partitioning over the resolved signal space.

    ``signal`` defines what is measured and ``signal_scope`` defines where it is
    measured. ``gradiend_split`` is the separate axis that decides whether the
    flattened eligible signal space is trained as one model or partitioned into
    virtual component models. ``None``/``GradiendSplit.none()`` keeps the ordinary
    unpartitioned ``GradiendModel`` behavior; ``GradiendSplit.single()`` uses the
    partitioned code path with one full-space component; ``GradiendSplit.by_tensor()``
    creates one component per resolved signal-space tensor entry.
    """

    gradiend_split_loss: str = "mean"
    """Loss aggregation for component-split GRADIEND training: 'mean', 'sum', 'size_weighted', or 'full'. Non-split models always use the classic full-vector objective."""

    activation_encoder: Optional[str] = None
    """Encoder activation name (e.g. 'tanh', 'gelu', 'relu'). None = model default ('tanh')."""

    activation_decoder: Optional[str] = None
    """Decoder activation name (e.g. 'id', 'tanh'). None = model default ('id')."""

    bias_encoder: Optional[bool] = True
    """Whether the encoder linear layer uses a bias term. Enabled by default; None also defers to the enabled model default."""

    bias_decoder: Optional[bool] = None
    """Whether the decoder linear layer uses a bias term. None = model default (True)."""

    latent_dim: Optional[int] = None
    """GRADIEND latent dimension (number of features). None = model default (1)."""

    init_fan_in_floor: Optional[int] = 10_000
    """Optional lower bound for fresh GRADIEND/ACTIEND initialization fan-in.

    Encoder weights use ``1 / sqrt(max(init_fan_in_floor, component_fan_in))``
    and decoder rows are initialized on the matching component scale. The
    default keeps small activation-space ACTIEND components from starting with
    much larger random weights than classic parameter-space GRADIENDs; this was
    empirically helpful for ACTIEND convergence. Large GRADIEND parameter spaces
    are already above the floor, so their scale is unchanged. Set to ``None``
    for raw component fan-in initialization.
    """

    normalize_gradiend: bool = True
    """Whether to normalize GRADIEND encodings during training, i.e., first target class is encoded to +1 and second to -1. This is recommended for enhanced comparability between runs."""

    positive_class: Optional[str] = None
    """Optional canonical positive feature class used for binary cross-encoding comparisons.
    When None, comparison utilities may infer it conservatively from target classes via
    the non_/non- prefix heuristic. Ignored for normal training."""

    torch_dtype: Optional[torch.dtype] = None
    """dtype for model; None = torch.float32."""

    base_model_device_map: Optional[Union[bool, str, Dict[str, Any]]] = None
    """Hugging Face device_map for the base model. None auto-detects large models on >3 GPUs,
    False disables device_map, and strings/dicts are passed through explicitly."""

    base_model_max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None
    """Optional Hugging Face max_memory map for base-model device placement.
    When unset and base_model_device_map='auto' on multiple GPUs, GPU0 is reserved for GRADIEND automatically."""

    encoder_decoder_same_device: bool = False
    """If True, place encoder and decoder on the same GPU (cuda:0), giving the base model the rest.
    Useful for large base models with pre-pruning: encoder+decoder are small and can share GPU 0;
    base model can use cuda:1 (2 GPUs) or cuda:2 (3+ GPUs). If False (default),
    encoder and decoder are split across cuda:0 and cuda:1 when 2+ GPUs are available."""

    # ----- Multi-seed training -----
    max_seeds: int = 3
    """Maximum number of seeds to try."""

    min_convergent_seeds: Optional[int] = 1
    """Stop once this many seeds have converged. None = run max_seeds. 0 is invalid."""

    convergent_metric: Optional[str] = None
    """Metric used to decide convergence: "correlation", "roc_auc"/"auroc",
    "min_auc_n_o"/"min_auc", or "loss".

    Defaults to correlation unless supervised_decoder (then loss). Use ``min_auc_n_o`` for
    one-pole runs (``min(auc_n, auc_o)`` so neutrals alone cannot carry selection). Legacy
    ``roc_auc`` is pooled one-vs-rest (rivals ∪ neutrals as negatives)."""

    selection_metric: Optional[str] = None
    """Metric used for best-checkpoint and best-seed selection.

    ``None`` preserves the historical behavior by using ``convergent_metric``.
    Set ``"encoding_e"``/``"E"`` to select by the encoding-E validation bottleneck
    (see the evaluation guide) while leaving convergence semantics unchanged. For one-pole data this includes ``auc_rival`` and class
    exclusivity whenever rival factual rows exist."""

    convergent_score_threshold: Optional[float] = None
    """Threshold for convergence. Defaults: 0.5 (correlation), 0.9 (roc_auc / min_auc_n_o); required for loss."""

    convergent_mean_by_class_threshold: Optional[float] = None
    """Optional additional convergence criterion: minimum absolute mean encoded value per target class.

    Default: 0.5 when convergent_metric='correlation'. Set to None to disable the mean-based check and use only
    convergent_score_threshold. When set, convergence requires BOTH |correlation| >= convergent_score_threshold AND
    min(|mean|) over non-zero target classes >= convergent_mean_by_class_threshold at the best checkpoint step. For
    correlation-based convergence, the two non-zero target classes must also have opposite-sign mean encodings
    at the best checkpoint step (their product must be negative).

    For ``roc_auc`` / ``min_auc_n_o``, the mean-based opposite-sign check is off unless this threshold is set explicitly.

    This flag only defines the end-of-training convergence check unless
    ``prefer_convergent_checkpoint=True`` (see that argument)."""

    prefer_convergent_checkpoint: bool = False
    """If True, best-checkpoint selection prefers steps that meet the convergence
    criteria (score threshold, and for correlation: opposite-sign target means / optional mean threshold)
    over a higher-score step that fails them.

    If False (default), the best checkpoint is selected by the selection score alone
    (``|correlation|``, ``roc_auc``, or ``min_auc_n_o``); convergence is still evaluated afterward at that best step."""

    split_resplit_per_seed: bool = False
    """When ``split_col`` is ``\"heldout\"`` or ``None``, re-draw splits per training seed.
    For ``\"heldout\"``, vocabulary groups rotate (see ``split_resplit_strategy``).
    For ``None``, rows are reshuffled randomly. False keeps the same assignment across
    multi-seed runs (using TrainingArguments.seed)."""

    split_resplit_strategy: Literal["random", "balanced_cycle"] = "random"
    """Strategy used when split_resplit_per_seed=True.
    ``"random"`` redraws splits from each seed. ``"balanced_cycle"`` rotates
    canonical target groups through train/validation/test across seed indices
    so ratios such as 60/20/20 over five seed slots place each word in train
    three times, validation once, and test once."""

    seed: Optional[int] = 0
    """Random seed for reproducible runs (default 0). The Trainer sets PyTorch/numpy/Python RNG, CUDA determinism, and CUBLAS/OMP env vars; data pipelines use this as random_state. Also the base for multi-seed runs (seed+i). Pass seed=None for non-deterministic runs. If results still vary, call set_seed(42) at the very start of your script or set env CUBLAS_WORKSPACE_CONFIG=:4096:8 and OMP_NUM_THREADS=1 before starting Python."""

    seed_runs_dir: Optional[str] = None
    """Directory for per-seed runs. Defaults to experiment_dir/seeds when experiment_dir is set."""

    saved_seed_runs: str = "all_convergent"
    """Multi-seed retention policy: 'best_only', 'all_convergent', or 'all_tried'.
    """

    seed_stability_topk: Optional[Union[int, float]] = 1000
    """Top-k selection used for stability summaries across convergent seeds. None disables the report."""

    seed_stability_part: str = "decoder-weight"
    """Importance part used for convergent-seed top-k stability summaries."""

    analyze_seed_stability: bool = False
    """If True, require at least min_convergent_seeds convergent seeds after multi-seed training
    and forbid saved_seed_runs='best_only'. Multi-seed evaluation uses trainer.multi_seed()."""

    # ----- Advanced -----
    supervised_encoder: bool = False
    """If True, train only the GRADIEND encoder: encode(source) vs labels (MSE). Baseline mode."""

    supervised_decoder: bool = False
    """If True, train only the GRADIEND decoder: decoder(labels) vs target gradients (MSE). Baseline mode. Cannot be True together with supervised_encoder."""

    use_cached_gradients: bool = False
    """Whether to use cached gradients if available. Using cached gradients speeds up training and evaluation, but leads to exhaustive memory usage (in memory) and/or on disk (in cached files)."""

    # ----- Pre-prune -----
    pre_prune_config: Optional["PrePruneConfig"] = None
    """If set, pre-prune is run automatically before training. The pruned model is kept in memory; training then uses it. No disk save unless you save explicitly."""

    reuse_pre_prune: bool = False
    """If True, cache pre-prune keep_idx under experiment_dir/cache/pre_prune for reuse across seeds in one train() call. Cache is removed when training finishes."""

    fail_on_non_convergence: bool = False
    """If True, raise when training finishes and convergent_count < min_convergent_seeds (requires min_convergent_seeds > 0)."""

    highlight_non_convergence: bool = True
    """If True, append a non-convergence marker (†) to plot/tick labels for non-converged runs."""

    # ----- Post-prune -----
    post_prune_config: Optional["PostPruneConfig"] = None
    """If set, post-prune is run automatically after training. The pruned model is kept in memory for subsequent evaluation. No disk save unless you save explicitly."""

    # ----- Extra -----
    metadata: dict = field(default_factory=dict)

    @staticmethod
    def _coerce_signal(value: Any) -> Any:
        return coerce_signal(value)

    @staticmethod
    def _coerce_signal_set(value: Any) -> Any:
        return coerce_signal_set(value)

    def _signal_default_decoder_lr_is_auto(self) -> bool:
        """Whether the ``"default"`` decoder learning rate resolves to ``"auto"``.

        True only when every configured signal is ``activation``. Reads the kinds
        directly because it runs before ``_normalize_signal_arguments``.
        """
        kinds = set()
        if self.signal is not None:
            kinds.add(getattr(self.signal, "kind", None))
        if self.signals is not None:
            try:
                for s in self.signals:
                    kinds.add(getattr(s, "kind", None))
            except TypeError:
                pass
        kinds.discard(None)
        return kinds == {"activation"}

    def _normalize_signal_arguments(self) -> None:
        signal, signals = normalize_signal_arguments(signal=self.signal, signals=self.signals)
        self.signal = signal
        self.signals = signals
        self.signal_scope = coerce_signal_scope(self.signal_scope)
        if (
            self.signal_scope is not None
            and self.signal is not None
            and self.signal.kind == "gradient"
            and self.signal_scope.activation_sites is not None
        ):
            # Semantic shortcuts (SignalScope.layers()/.embeddings()/...) resolve to weight
            # parameters for gradient signals (see signal_space.gradient_params_from_selector).
            # A raw module-path include-list cannot be translated to parameter names
            # unambiguously, so reject it instead of silently matching nothing.
            raise ValueError(
                "signal_scope.activation_sites has no effect on a gradient signal "
                "(Signal.gradient()) -- it is only resolved for Signal.activation(). "
                "SignalScope.layers()/.layer()/.embeddings()/.word_embedding() (the "
                "activation_selector shortcuts) DO work for gradient signals; only a "
                "raw activation_sites=[...] include-list does not. For a gradient "
                "signal's weight-parameter scope, use SignalScope.from_values(params=[...]) "
                "(or .default()/.full() for the built-in mode presets) instead."
            )
        self.gradiend_split = coerce_gradiend_split(self.gradiend_split)
        if self.params is not None:
            warnings.warn(
                "TrainingArguments.params is deprecated; use "
                "TrainingArguments.signal_scope=SignalScope.from_values(params=...) instead.",
                DeprecationWarning,
                stacklevel=4,
            )
            params_tuple = tuple(self.params)
            if self.signal_scope is None:
                self.signal_scope = SignalScope.from_values(params=params_tuple)
            elif self.signal_scope.params is None:
                self.signal_scope = SignalScope.from_values(
                    params=params_tuple,
                    activation_sites=self.signal_scope.activation_sites,
                    mode=self.signal_scope.mode,
                )
            elif self.signal_scope.params != params_tuple:
                raise ValueError(
                    "TrainingArguments.params conflicts with signal_scope.params. "
                    "Use only signal_scope.params."
                )

    def __post_init__(self) -> None:
        # Type checks for key scalar parameters
        if self.experiment_dir is not None and not isinstance(self.experiment_dir, str):
            raise TypeError(f"experiment_dir must be str or None, got {type(self.experiment_dir).__name__}")
        if self.output_dir is not None and not isinstance(self.output_dir, str):
            raise TypeError(f"output_dir must be str or None, got {type(self.output_dir).__name__}")
        normalize_use_cache(self.use_cache)
        if self.label_token_protocol not in ("canonical", "legacy"):
            raise ValueError(
                f"label_token_protocol must be 'canonical' or 'legacy', got {self.label_token_protocol!r}"
            )
        if self.learning_rate_decoder == "default":
            # Signal-kind-aware default: 'auto' only for pure activation signals (see the
            # field documentation); explicit 'auto'/None/float bypass this.
            self.learning_rate_decoder = (
                "auto" if self._signal_default_decoder_lr_is_auto() else None
            )
        if self.learning_rate_decoder is not None:
            if isinstance(self.learning_rate_decoder, str):
                normalized_decoder_lr = self.learning_rate_decoder.strip().lower()
                if normalized_decoder_lr != "auto":
                    raise ValueError(
                        "learning_rate_decoder string value must be 'auto', got "
                        f"{self.learning_rate_decoder!r}"
                    )
                self.learning_rate_decoder = normalized_decoder_lr
            elif isinstance(self.learning_rate_decoder, bool) or not isinstance(
                self.learning_rate_decoder, (int, float)
            ):
                raise TypeError(
                    "learning_rate_decoder must be a number, 'auto', or None, got "
                    f"{type(self.learning_rate_decoder).__name__}"
                )
            elif not self.learning_rate_decoder > 0:
                raise ValueError(
                    f"learning_rate_decoder must be positive, got {self.learning_rate_decoder}"
                )
            if self.supervised_encoder:
                raise ValueError(
                    "learning_rate_decoder is incompatible with supervised_encoder=True: "
                    "no decoder parameter is trained, so the decoder learning rate "
                    "would have no effect."
                )
            if self.learning_rate_decoder != "auto":
                self.learning_rate_decoder = float(self.learning_rate_decoder)
        if not isinstance(self.reuse_pre_prune, bool):
            raise TypeError(f"reuse_pre_prune must be bool, got {type(self.reuse_pre_prune).__name__}")
        if not isinstance(self.fail_on_non_convergence, bool):
            raise TypeError(f"fail_on_non_convergence must be bool, got {type(self.fail_on_non_convergence).__name__}")
        if not isinstance(self.highlight_non_convergence, bool):
            raise TypeError(
                f"highlight_non_convergence must be bool, got {type(self.highlight_non_convergence).__name__}"
            )
        if not isinstance(self.analyze_seed_stability, bool):
            raise TypeError(
                f"analyze_seed_stability must be bool, got {type(self.analyze_seed_stability).__name__}"
            )
        if not isinstance(self.split_resplit_per_seed, bool):
            raise TypeError(
                f"split_resplit_per_seed must be bool, got {type(self.split_resplit_per_seed).__name__}"
            )
        if self.split_resplit_strategy not in {"random", "balanced_cycle"}:
            raise ValueError(
                "split_resplit_strategy must be 'random' or 'balanced_cycle', "
                f"got {self.split_resplit_strategy!r}"
            )
        if not isinstance(self.source, str):
            raise TypeError(f"source must be str, got {type(self.source).__name__}")
        if not isinstance(self.target, str):
            raise TypeError(f"target must be str, got {type(self.target).__name__}")
        if not isinstance(self.train_batch_size, int):
            raise TypeError(f"train_batch_size must be int, got {type(self.train_batch_size).__name__}")
        if self.train_batch_size < 1:
            raise ValueError(f"train_batch_size must be >= 1, got {self.train_batch_size}")
        if self.base_gradient_batch_size is None:
            self.base_gradient_batch_size = self.train_batch_size
        if self.gradiend_batch_size is None:
            self.gradiend_batch_size = 1
        if not isinstance(self.base_gradient_batch_size, int):
            raise TypeError(
                f"base_gradient_batch_size must be int or None, got {type(self.base_gradient_batch_size).__name__}"
            )
        if self.base_gradient_batch_size < 1:
            raise ValueError(f"base_gradient_batch_size must be >= 1, got {self.base_gradient_batch_size}")
        if not isinstance(self.gradiend_batch_size, int):
            raise TypeError(
                f"gradiend_batch_size must be int or None, got {type(self.gradiend_batch_size).__name__}"
            )
        if self.gradiend_batch_size < 1:
            raise ValueError(f"gradiend_batch_size must be >= 1, got {self.gradiend_batch_size}")
        if self.precompute_gradient_batches is not None and not isinstance(self.precompute_gradient_batches, bool):
            raise TypeError(
                "precompute_gradient_batches must be bool or None, "
                f"got {type(self.precompute_gradient_batches).__name__}"
            )
        if not isinstance(self.precompute_gradient_buffer_size, int):
            raise TypeError(
                "precompute_gradient_buffer_size must be int, "
                f"got {type(self.precompute_gradient_buffer_size).__name__}"
            )
        if self.precompute_gradient_buffer_size < 1:
            raise ValueError(
                f"precompute_gradient_buffer_size must be >= 1, got {self.precompute_gradient_buffer_size}"
            )
        if not isinstance(self.gradient_timing_steps, int):
            raise TypeError(f"gradient_timing_steps must be int, got {type(self.gradient_timing_steps).__name__}")
        if self.gradient_timing_steps < 0:
            raise ValueError(f"gradient_timing_steps must be >= 0, got {self.gradient_timing_steps}")
        if not isinstance(self.runtime_monitor, bool):
            raise TypeError(f"runtime_monitor must be bool, got {type(self.runtime_monitor).__name__}")
        if not isinstance(self.runtime_monitor_interval, (int, float)):
            raise TypeError(
                f"runtime_monitor_interval must be int or float, got {type(self.runtime_monitor_interval).__name__}"
            )
        if float(self.runtime_monitor_interval) < 0:
            raise ValueError(f"runtime_monitor_interval must be >= 0, got {self.runtime_monitor_interval}")
        self.runtime_monitor_interval = float(self.runtime_monitor_interval)
        if not isinstance(self.runtime_monitor_system_stats, bool):
            raise TypeError(
                f"runtime_monitor_system_stats must be bool, got {type(self.runtime_monitor_system_stats).__name__}"
            )
        if self.train_max_size is not None and not isinstance(self.train_max_size, int):
            raise TypeError(f"train_max_size must be int or None, got {type(self.train_max_size).__name__}")
        if self.train_max_size is not None and self.train_max_size < 0:
            raise ValueError(f"train_max_size must be >= 0, got {self.train_max_size}")
        if not isinstance(self.learning_rate, (int, float)):
            raise TypeError(f"learning_rate must be float, got {type(self.learning_rate).__name__}")
        if not isinstance(self.num_train_epochs, int):
            raise TypeError(f"num_train_epochs must be int, got {type(self.num_train_epochs).__name__}")
        if not isinstance(self.max_steps, int):
            raise TypeError(f"max_steps must be int, got {type(self.max_steps).__name__}")
        if not isinstance(self.eval_steps, int):
            raise TypeError(f"eval_steps must be int, got {type(self.eval_steps).__name__}")
        if self.learning_rate_decoder == "auto" and self.eval_steps < 1:
            raise ValueError(
                "learning_rate_decoder='auto' requires eval_steps >= 1 because "
                "the first evaluation boundary is the calibration boundary"
            )
        if not isinstance(self.eval_batch_size, int):
            raise TypeError(f"eval_batch_size must be int, got {type(self.eval_batch_size).__name__}")
        if self.eval_batch_size < 1:
            raise ValueError(f"eval_batch_size must be >= 1, got {self.eval_batch_size}")
        if not isinstance(self.max_length, int):
            raise TypeError(f"max_length must be int, got {type(self.max_length).__name__}")
        if self.max_length < 8:
            raise ValueError(f"max_length must be >= 8, got {self.max_length}")
        if not isinstance(self.do_eval, bool):
            raise TypeError(f"do_eval must be bool, got {type(self.do_eval).__name__}")
        if self.add_neutral_identity_transitions is not None and not isinstance(
            self.add_neutral_identity_transitions, bool
        ):
            raise TypeError(
                "add_neutral_identity_transitions must be bool or None, got "
                f"{type(self.add_neutral_identity_transitions).__name__}"
            )
        if self.seed is not None and not isinstance(self.seed, int):
            raise TypeError(f"seed must be int or None, got {type(self.seed).__name__}")
        if not isinstance(self.gradiend_split_loss, str):
            raise TypeError(f"gradiend_split_loss must be str, got {type(self.gradiend_split_loss).__name__}")
        self.gradiend_split_loss = self.gradiend_split_loss.strip().lower()
        if self.gradiend_split_loss not in {"mean", "sum", "size_weighted", "full"}:
            raise ValueError(
                "gradiend_split_loss must be 'mean', 'sum', 'size_weighted', or 'full', "
                f"got {self.gradiend_split_loss!r}"
            )
        if not isinstance(self.mask_placeholder, str):
            raise TypeError(
                f"mask_placeholder must be a non-empty str, got {type(self.mask_placeholder).__name__}"
            )
        if not self.mask_placeholder:
            raise ValueError("mask_placeholder must be a non-empty str")

        self._normalize_signal_arguments()

        validate_source_target("source", self.source)
        validate_source_target("target", self.target)
        validate_source_target_combination(self.source, self.target)
        if self.torch_dtype is None:
            self.torch_dtype = torch.float32
        if self.init_fan_in_floor is not None:
            if isinstance(self.init_fan_in_floor, bool) or not isinstance(self.init_fan_in_floor, int):
                raise TypeError(
                    "init_fan_in_floor must be a positive int or None, "
                    f"got {type(self.init_fan_in_floor).__name__}"
                )
            if self.init_fan_in_floor < 1:
                raise ValueError(
                    f"init_fan_in_floor must be >= 1 or None, got {self.init_fan_in_floor}"
                )
        if self.base_model_device_map is not None and self.base_model_device_map is not False and not isinstance(self.base_model_device_map, (str, dict)):
            raise TypeError(
                "base_model_device_map must be None, False, a string such as 'auto', or a device-map dict; "
                f"got {type(self.base_model_device_map).__name__}"
            )
        if self.base_model_max_memory is not None and not isinstance(self.base_model_max_memory, dict):
            raise TypeError(
                "base_model_max_memory must be None or a max-memory dict; "
                f"got {type(self.base_model_max_memory).__name__}"
            )
        if self.criterion is None:
            self.criterion = nn.MSELoss()
        if self.supervised_encoder and self.supervised_decoder:
            raise ValueError(
                "Cannot set both supervised_encoder and supervised_decoder. "
                "Run two separate train() calls: train(supervised_encoder=True) then train(supervised_decoder=True)."
            )
        if self.supervised_encoder and self.target is not None:
            self.target = None  # encoder baseline doesn't use target
        if self.max_seeds is None:
            self.max_seeds = 3
        if not isinstance(self.max_seeds, int) or self.max_seeds < 1:
            raise ValueError(f"max_seeds must be a positive int, got {self.max_seeds!r}")
        if self.min_convergent_seeds == 0:
            raise ValueError("min_convergent_seeds=0 is not allowed. Use None to run max_seeds.")
        if self.min_convergent_seeds is not None:
            if not isinstance(self.min_convergent_seeds, int) or self.min_convergent_seeds < 0:
                raise ValueError("min_convergent_seeds must be a positive int or None.")
            if self.min_convergent_seeds > self.max_seeds:
                raise ValueError("min_convergent_seeds cannot exceed max_seeds.")
        if not isinstance(self.saved_seed_runs, str):
            raise TypeError(f"saved_seed_runs must be str, got {type(self.saved_seed_runs).__name__}")
        self.saved_seed_runs = str(self.saved_seed_runs).strip().lower()
        supported_saved_seed_runs = {"best_only", "all_convergent", "all_tried"}
        if self.saved_seed_runs not in supported_saved_seed_runs:
            raise ValueError(
                f"saved_seed_runs must be one of {sorted(supported_saved_seed_runs)}, got {self.saved_seed_runs!r}"
            )
        if self.analyze_seed_stability and self.saved_seed_runs == "best_only":
            raise ValueError(
                "analyze_seed_stability=True requires convergent seed checkpoints on disk; "
                "saved_seed_runs='best_only' deletes non-selected seed runs."
            )
        if self.seed_stability_topk is not None:
            _validate_topk(self.seed_stability_topk, "seed_stability_topk")
        if not isinstance(self.seed_stability_part, str) or not self.seed_stability_part.strip():
            raise ValueError("seed_stability_part must be a non-empty string.")

        metric = normalize_metric_name(
            self.convergent_metric or ("loss" if self.supervised_decoder else "correlation")
        )
        if self.convergent_metric is not None:
            self.convergent_metric = metric
        if metric not in ("correlation", "loss", "roc_auc", "min_auc_n_o"):
            raise ValueError(
                "convergent_metric must be 'correlation', 'roc_auc'/'auroc', "
                "'min_auc_n_o'/'min_auc', or 'loss', "
                f"got {metric!r}"
            )
        if metric == "correlation" and self.convergent_score_threshold is None:
            self.convergent_score_threshold = 0.5
        if metric == "correlation" and self.convergent_mean_by_class_threshold is None:
            self.convergent_mean_by_class_threshold = 0.5
        if metric in {"roc_auc", "min_auc_n_o"} and self.convergent_score_threshold is None:
            self.convergent_score_threshold = 0.9
        # roc_auc / min_auc_n_o: do not auto-enable bipolar mean threshold
        # (identity/neutral at ≤0 is fine).
        if metric == "loss" and self.convergent_score_threshold is None:
            raise ValueError("convergent_score_threshold is required when convergent_metric='loss'.")

        if self.selection_metric is not None:
            selection = normalize_metric_name(self.selection_metric)
            if selection not in {"correlation", "loss", "roc_auc", "min_auc_n_o", "encoding_e"}:
                raise ValueError(
                    "selection_metric must be 'correlation', 'roc_auc', 'min_auc_n_o', "
                    f"'encoding_e'/'E', or 'loss', got {selection!r}"
                )
            self.selection_metric = selection

    def to_dict(self) -> dict:
        """Dict for serialization (excludes callables and nn.Module). Canonical keys only."""
        fields = getattr(type(self), "__dataclass_fields__", {})
        result = {}
        for k in fields:
            if k in ("evaluate_fn", "criterion"):
                continue
            v = getattr(self, k, None)
            if callable(v):
                continue
            if isinstance(v, torch.dtype):
                # ``str(torch.bfloat16)`` is "torch.bfloat16"; ``from_dict``
                # resolves the name via ``getattr(torch, name)``, which cannot
                # see through that prefix.  Serialize the bare name so the
                # round-trip is lossless (see ``dtype_from_name``).
                result[k] = dtype_to_name(v)
            elif isinstance(v, nn.Module):
                result[k] = str(v) if v is not None else None
            elif k == "pre_prune_config" and v is not None:
                cfg = dataclasses.asdict(v)
                cfg["dataset"] = None  # do not serialize dataset reference
                result[k] = cfg
            elif k == "post_prune_config" and v is not None:
                cfg = dataclasses.asdict(v)
                cfg["mask"] = None  # do not serialize tensor
                result[k] = cfg
            elif k == "signal":
                result[k] = v.to_dict() if v is not None and hasattr(v, "to_dict") else v
            elif k == "signals":
                result[k] = list(v.to_list()) if v is not None and hasattr(v, "to_list") else v
            elif k == "signal_scope":
                result[k] = v.to_dict() if v is not None and hasattr(v, "to_dict") else v
            elif k == "gradiend_split":
                result[k] = v.to_dict() if v is not None and hasattr(v, "to_dict") else v
            else:
                result[k] = v
        return result

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingArguments":
        """Create from dict (e.g. loaded from JSON). Canonical keys only."""
        d = dict(d)
        # A serialized dict without the key was written before the protocol existed:
        # it must keep the old label convention, not silently pick up the new default.
        d.setdefault("label_token_protocol", "legacy")
        if "signal" in d and isinstance(d.get("signal"), dict):
            d["signal"] = Signal.from_dict(d["signal"])
        if "signals" in d and isinstance(d.get("signals"), list):
            d["signals"] = SignalSet.from_list(d["signals"])
        if "signal_scope" in d and isinstance(d.get("signal_scope"), dict):
            d["signal_scope"] = SignalScope.from_dict(d["signal_scope"])
        if "gradiend_split" in d and isinstance(d.get("gradiend_split"), dict):
            d["gradiend_split"] = GradiendSplit.from_dict(d["gradiend_split"])
        if "torch_dtype" in d and isinstance(d.get("torch_dtype"), str):
            d["torch_dtype"] = dtype_from_name(d["torch_dtype"])
        if "pre_prune_config" in d and isinstance(d.get("pre_prune_config"), dict):
            d["pre_prune_config"] = PrePruneConfig(**d["pre_prune_config"])
        if "post_prune_config" in d and isinstance(d.get("post_prune_config"), dict):
            d["post_prune_config"] = PostPruneConfig(**d["post_prune_config"])
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def get(self, key: str, default: Any = None) -> Any:
        """Dict-like get method."""
        return getattr(self, key, default)

    def __str__(self) -> str:
        parts = [
            f"experiment_dir={self.experiment_dir!r}",
            f"output_dir={self.output_dir!r}",
            f"source={self.source!r}",
            f"target={self.target!r}",
            f"learning_rate={self.learning_rate}",
            f"num_train_epochs={self.num_train_epochs}",
            f"max_steps={self.max_steps}",
            f"seed={self.seed}",
            f"max_seeds={self.max_seeds}",
        ]
        return f"TrainingArguments({', '.join(parts)})"
