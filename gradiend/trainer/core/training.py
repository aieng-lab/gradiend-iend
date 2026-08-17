"""
Core training loop for GRADIEND models.
"""

import gc
import os
import shutil
import time
import dataclasses
from typing import Any, List, Optional, Sequence

import torch
from gradiend.util.tqdm_utils import gradiend_tqdm
from torch.utils.data import DataLoader

from gradiend.util.logging import get_logger
from gradiend.trainer.core.arguments import TrainingArguments
from gradiend.trainer.core.signals import require_single_signal
from gradiend.trainer.core.callbacks import (
    TrainingCallback,
    EvaluationCallback,
    NormalizationCallback,
    CheckpointCallback,
    LoggingCallback,
    get_default_callbacks,
)
from gradiend.trainer.core.component_seed import (
    apply_saved_component_best_states,
    component_run_from_training_stats,
    finalize_component_best_states,
)
from gradiend.trainer.core.stats import (
    write_training_stats,
    load_training_stats,
    _best_checkpoint_step_is_after_initial,
    _best_step_min_target_class_abs_mean,
    _best_step_target_class_mean_product,
)
from gradiend.util.component_logging import (
    best_step_component_payload,
    format_component_convergence_fragment,
    format_component_seed_summary_fragment,
    strip_component_prefix,
)

logger = get_logger(__name__)


def format_training_start_message(model_with_gradiend: Any, training_args: TrainingArguments) -> str:
    """Describe the GRADIEND/ACTIEND training space without hiding split structure."""
    gradiend = getattr(model_with_gradiend, "gradiend", None)
    if gradiend is None:
        return "Training GRADIEND model."
    describe = getattr(gradiend, "describe_training_space", None)
    if callable(describe):
        return describe(split_loss=getattr(training_args, "gradiend_split_loss", None))
    return (
        f"Training GRADIEND over {len(model_with_gradiend):,} "
        "gradient entries."
    )


def _best_step_component_summary(training_stats: dict, best_score_checkpoint: dict) -> Optional[dict]:
    best_step = best_score_checkpoint.get("global_step") if isinstance(best_score_checkpoint, dict) else None
    if best_step is None or not isinstance(training_stats, dict):
        return None
    payload = best_step_component_payload(training_stats, best_score_checkpoint)
    if isinstance(payload, dict) and isinstance(payload.get("summary"), dict):
        return payload["summary"]
    hist = training_stats.get("component_summary")
    if not isinstance(hist, dict):
        return None
    return hist.get(best_step) or hist.get(str(best_step))


def format_non_convergence_error(
    *,
    actual: int,
    min_required: int,
    training_args: Optional[TrainingArguments] = None,
    run_id: Optional[str] = None,
    model: Any = None,
    pair: Any = None,
    output_dir: Optional[str] = None,
    seed_report: Optional[Sequence[dict]] = None,
    components: Optional[dict] = None,
    component_summary: Optional[dict] = None,
    component_seed_summary: Optional[dict] = None,
    convergence_metric: Optional[str] = None,
    threshold: Optional[float] = None,
) -> str:
    """Build an actionable fail_on_non_convergence error message."""
    args = training_args
    out = output_dir or (getattr(args, "output_dir", None) if args is not None else None)
    metric = convergence_metric or (getattr(args, "convergent_metric", None) if args is not None else None)
    if metric is None and args is not None:
        metric = "loss" if getattr(args, "supervised_decoder", False) else "correlation"
    thr = threshold if threshold is not None else (
        getattr(args, "convergent_score_threshold", None) if args is not None else None
    )

    name = run_id or (str(out).rstrip("/\\").split("/")[-1].split("\\")[-1] if out else None)
    lines = [
        f"fail_on_non_convergence=True for GRADIEND {name!r}: "
        f"only {actual} seed(s) converged, but {min_required} are required."
    ]
    if run_id:
        lines.append(f"  run_id: {run_id}")
    if model is not None:
        model_name = getattr(model, "name_or_path", model)
        lines.append(f"  model: {model_name}")
    if pair is not None:
        lines.append(f"  target pair: {pair}")
    if out:
        lines.append(f"  output_dir: {out}")
    if metric is not None:
        metric_line = f"  convergence metric: {metric}"
        if thr is not None:
            metric_line += f" (threshold={thr})"
        lines.append(metric_line)
    mean_thr = getattr(args, "convergent_mean_by_class_threshold", None) if args is not None else None
    if mean_thr is not None:
        lines.append(f"  mean-by-class threshold: {mean_thr}")
    component_fragment = format_component_convergence_fragment(
        components,
        summary=component_summary,
        show_blockers=True,
    )
    if component_fragment:
        lines.append(f"  component convergence: {strip_component_prefix(component_fragment)}")
    component_seed_fragment = format_component_seed_summary_fragment(
        component_seed_summary,
        show_blockers=True,
    )
    if component_seed_fragment:
        lines.append(f"  component convergence: {strip_component_prefix(component_seed_fragment)}")
    if seed_report:
        lines.append("  seed results:")
        for run in seed_report:
            seed = run.get("seed", "?")
            converged = bool(run.get("converged"))
            value = run.get("convergence_metric_value", run.get("selection_score", run.get("training_score")))
            step = run.get("best_checkpoint_global_step")
            seed_out = run.get("output_dir")
            detail = f"    - seed {seed}: converged={converged}"
            if value is not None:
                detail += f", value={value}"
            if step is not None:
                detail += f", best_step={step}"
            if seed_out:
                detail += f", output={seed_out}"
            lines.append(detail)
    lines.append(
        "  Try increasing max_seeds, lowering min_convergent_seeds, relaxing "
        "convergent_score_threshold, or inspecting the per-seed training.json files."
    )
    return "\n".join(lines)


def train(
    model_with_gradiend,
    data: DataLoader,
    training_args: Optional[TrainingArguments] = None,
    callbacks: Optional[List[TrainingCallback]] = None,
    runtime_monitor=None,
    **kwargs
) -> str:
    """
    Train a GRADIEND model.

    Uses a list of callbacks (default: evaluation, normalization, checkpoint, logging).
    Pass callbacks=[] to add extra callbacks (e.g. EarlyStoppingCallback, TensorBoardCallback).

    Args:
        model_with_gradiend: ModelWithGradiend instance to train
        data: DataLoader providing training batches
        training_args: TrainingArguments (single config class for all training parameters)
        callbacks: Optional list of extra callbacks (appended to default callbacks)
        **kwargs: Additional parameters (will override config values)

    Returns:
        Path to saved model
    """
    if training_args is None:
        training_args = TrainingArguments.from_dict(kwargs)
    else:
        # Create a copy to avoid modifying the original object
        if kwargs:
            # Build dict of fields to update
            updates = {}
            for key, value in kwargs.items():
                if hasattr(training_args, key) and key in getattr(TrainingArguments, "__dataclass_fields__", {}):
                    updates[key] = value
                else:
                    raise ValueError(f"Invalid training argument: {key} with value {value}. All train() kwargs must be a field in TrainingArguments.")
            if updates:
                if "signal" in updates and "signals" not in updates:
                    updates["signals"] = None
                if "signals" in updates and "signal" not in updates:
                    updates["signal"] = None
                training_args = dataclasses.replace(training_args, **updates)

    training_args.__post_init__()
    require_single_signal(
        signal=training_args.signal,
        signals=training_args.signals,
        context="train()",
    )

    # Re-apply seed so training loop (forward/backward, dropout, etc.) starts from a known RNG state
    if getattr(training_args, "seed", None) is not None:
        from gradiend.trainer.trainer import _apply_seed
        _apply_seed(int(training_args.seed))

    # Log training start
    if model_with_gradiend.gradiend is not None:
        logger.info(format_training_start_message(model_with_gradiend, training_args))
        logger.info(f'Output: {training_args.output_dir or ""}')
    if runtime_monitor is not None:
        runtime_monitor.mark(
            "training:start",
            output_dir=training_args.output_dir,
            max_steps=training_args.max_steps,
            num_train_epochs=training_args.num_train_epochs,
            dataloader_len=len(data),
        )

    if len(data) == 0:
        raise ValueError('Dataloader is empty! Please provide a dataloader with training data.')

    # Ensure model is on correct dtype
    if (
        not getattr(model_with_gradiend, "base_model_is_sharded", False)
        and getattr(model_with_gradiend.base_model, "dtype", None) != training_args.torch_dtype
    ):
        model_with_gradiend = model_with_gradiend.to(torch_dtype=training_args.torch_dtype)
    elif getattr(model_with_gradiend, "base_model_is_sharded", False):
        model_with_gradiend = model_with_gradiend.to(torch_dtype=training_args.torch_dtype)

    # Initialize training state
    last_losses = []
    losses = []
    global_step = 0
    total_training_time_start = time.time()

    training_stats = {
        'global_step': 0,
        'correlation': -1.0,
        'scores': {},
        'mean_by_class': {},  # step -> dict of label -> mean encoded value
        'mean_by_feature_class': {},  # step -> dict of feature_class (e.g. masc_nom) -> mean encoded value
        'encoder_norms': {},
        'decoder_norms': {},
    }

    time_stats = {
        'data_preparation': 0.0,
        'model_with_gradiend': 0.0,
        'eval': 0.0,
        'total': 0.0,
    }

    # Build callback list: defaults + optional extra. Deduplicate by default type (keep first).
    default_types = (EvaluationCallback, NormalizationCallback, CheckpointCallback, LoggingCallback)
    raw_callbacks = get_default_callbacks(training_args) + (callbacks or [])
    seen_types = set()
    all_callbacks = []
    for c in raw_callbacks:
        ctype = type(c)
        if ctype in default_types:
            if ctype in seen_types:
                logger.warning(
                    f"Duplicate {ctype.__name__} skipped; only the first instance is used. "
                    "Prefer get_default_callbacks() plus extra callbacks (e.g. EarlyStoppingCallback)."
                )
                continue
            seen_types.add(ctype)
        all_callbacks.append(c)
    # Keep enough recent losses for LoggingCallback's report-window average.
    max_losses = 100
    for _cb in all_callbacks:
        n_report = getattr(_cb, "n_loss_report", None)
        if isinstance(n_report, int) and n_report > max_losses:
            max_losses = n_report
    control: dict = {"should_stop": False}
    config_dict = training_args.to_dict()
    last_epoch = 0

    for cb in all_callbacks:
        cb.on_train_begin(config=config_dict, training_stats=training_stats)
    
    # Ensure encoder/decoder built (lazy init: build with full input_dim if not yet built)
    if hasattr(model_with_gradiend.gradiend, "_ensure_built"):
        model_with_gradiend.gradiend._ensure_built()

    # Setup optimizer (encoder-only, decoder-only, or full GRADIEND)
    if training_args.supervised_encoder:
        train_params = list(model_with_gradiend.gradiend.encoder.parameters())
        logger.info("Supervised encoder: optimizing encoder parameters only.")
    elif training_args.supervised_decoder:
        train_params = list(model_with_gradiend.gradiend.decoder.parameters())
        logger.info("Supervised decoder: optimizing decoder parameters only.")
    else:
        train_params = list(model_with_gradiend.gradiend.parameters())
    if training_args.optim.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            train_params,
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            eps=training_args.adam_epsilon
        )
    else:
        optimizer = torch.optim.Adam(
            train_params,
            lr=training_args.learning_rate,
            weight_decay=training_args.weight_decay,
            eps=training_args.adam_epsilon
        )

    # Initial evaluation before training starts (at step 0)
    if training_args.do_eval and training_args.evaluate_fn is not None:
        logger.info('Running initial evaluation before training...')
        if runtime_monitor is not None:
            runtime_monitor.mark("training:initial_eval:start")
        eval_start = time.time()
        eval_result = None
        _corr0 = training_stats.get('correlation', -1.0)
        step_kwargs_0 = dict(
            step=0,
            loss=None,
            model=model_with_gradiend,
            config=config_dict,
            training_stats=training_stats,
            eval_result=eval_result,
            control=control,
            last_iteration=False,
            last_losses=last_losses,
            losses=losses,
            epoch=0,
            best_score_checkpoint={
                'correlation': _corr0,
                'global_step': 0,
                'epoch': 0,
                'loss': None,
            },
        )
        for cb in all_callbacks:
            out = cb.on_step_end(**step_kwargs_0)
            if out is not None:
                eval_result = out
                step_kwargs_0['eval_result'] = eval_result
        time_stats['eval'] += time.time() - eval_start
        if runtime_monitor is not None:
            runtime_monitor.mark("training:initial_eval:done")
    
    # Training loop
    max_iter = training_args.max_steps if training_args.max_steps > 0 else None
    for epoch in range(training_args.num_train_epochs):
        last_epoch = epoch
        if max_iter is not None and global_step >= max_iter:
            logger.info(f'Max steps {max_iter} reached, stopping training.')
            break
        
        for cb in all_callbacks:
            cb.on_epoch_begin(epoch=epoch, config=config_dict, training_stats=training_stats)
        if runtime_monitor is not None:
            runtime_monitor.mark("training:epoch:start", epoch=epoch + 1)

        # Set tqdm total to remaining steps when max_steps will cause early stop this epoch
        if max_iter is not None:
            steps_remaining = max(0, max_iter - global_step)
            epoch_total = min(len(data), steps_remaining) if steps_remaining > 0 else 0
        else:
            epoch_total = len(data)
        dataloader_iterator = gradiend_tqdm(
            data,
            desc=f"Epoch {epoch + 1}/{training_args.num_train_epochs}",
            total=epoch_total,
            leave=True,
        )

        data_prep_start = time.time()
        for i, batch in enumerate(dataloader_iterator):
            if runtime_monitor is not None:
                runtime_monitor.mark("training:step:start", step=global_step + 1, epoch=epoch + 1)
            # Data preparation
            source_tensor = batch['source']
            target_tensor = batch['target']

            if training_args.supervised_encoder or training_args.supervised_decoder:
                labels = torch.tensor(batch['label'], dtype=torch.float32)
                if labels.dim() == 0:
                    labels = labels.unsqueeze(0)
                if labels.dim() == 1:
                    labels = labels.unsqueeze(1)  # (batch, 1)
                label_dim = labels.shape[1]
                gradiend_n_features = model_with_gradiend.gradiend.latent_dim
                if label_dim != gradiend_n_features:
                    raise ValueError(
                        f'Label dimension {label_dim} does not match GRADIEND latent dimension {gradiend_n_features}! '
                        'Supervised encoder/decoder training requires equal feature and label dimensions.'
                    )

            time_stats['data_preparation'] += time.time() - data_prep_start

            step_begin_kwargs = dict(
                step=global_step + 1,
                config=training_args.to_dict(),
                model=model_with_gradiend,
                epoch=epoch,
                training_stats=training_stats,
            )
            for cb in all_callbacks:
                cb.on_step_begin(**step_begin_kwargs)

            # Forward pass
            gradiend_start = time.time()

            if training_args.supervised_decoder:
                # Supervised decoder: decoder(labels) vs target gradients; do not use source
                if target_tensor is None:
                    raise ValueError(
                        'Supervised decoder training requires target gradients. '
                        'Ensure target is set (e.g. "diff") and dataset provides target.'
                    )
                dev_dec = model_with_gradiend.gradiend.device_decoder
                labels = labels.to(device=dev_dec, dtype=model_with_gradiend.gradiend.torch_dtype)
                decoder_output = model_with_gradiend.gradiend.decoder(labels)
                if target_tensor.device != decoder_output.device:
                    target_tensor = target_tensor.to(decoder_output.device)
                loss = training_args.criterion(decoder_output, target_tensor)
                del target_tensor
            elif training_args.supervised_encoder:
                # Supervised encoder: encode(source) vs labels
                if source_tensor.device != model_with_gradiend.gradiend.device_encoder:
                    source_tensor = source_tensor.to(model_with_gradiend.gradiend.device_encoder)
                encoded_value = model_with_gradiend.gradiend.encoder(source_tensor)
                del source_tensor
                if labels.device != encoded_value.device:
                    labels = labels.to(encoded_value.device)
                labels = labels.to(dtype=encoded_value.dtype)
                if training_args.gradiend_batch_size > 1:
                    labels = labels.mean(dim=0, keepdim=True)
                else:
                    labels = labels.unsqueeze(0) if labels.dim() == 1 else labels
                loss = training_args.criterion(encoded_value, labels)
            else:
                # Standard GRADIEND: encode and decode
                if source_tensor.device != model_with_gradiend.gradiend.device_encoder:
                    source_tensor = source_tensor.to(model_with_gradiend.gradiend.device_encoder)
                loss, encoded_value = model_with_gradiend.gradiend.reconstruction_loss(
                    source_tensor,
                    target_tensor,
                    criterion=training_args.criterion,
                    aggregation=getattr(training_args, "gradiend_split_loss", "mean"),
                    return_encoded=True,
                )
                del source_tensor
                del target_tensor
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if runtime_monitor is not None:
                runtime_monitor.mark("training:step:done", step=global_step + 1, epoch=epoch + 1)
            
            time_stats['model_with_gradiend'] += time.time() - gradiend_start
            
            # Update loss tracking
            loss_value = loss.item()
            last_losses.append(loss_value)
            if len(last_losses) > max_losses:
                last_losses.pop(0)
            losses.append(loss_value)
            
            global_step += 1
            
            # Update training stats
            training_stats['global_step'] = global_step
            training_stats['encoder_norms'][global_step] = model_with_gradiend.gradiend.encoder_norm
            training_stats['decoder_norms'][global_step] = model_with_gradiend.gradiend.decoder_norm
            
            # Call callbacks in order; eval_result flows from EvaluationCallback to later callbacks
            config_dict = training_args.to_dict()
            last_iteration = (
                (max_iter is not None and global_step >= max_iter) or
                (i == len(data) - 1 and epoch == training_args.num_train_epochs - 1)
            )
            _corr = training_stats.get('correlation', -1.0)
            step_kwargs = dict(
                step=global_step,
                loss=loss_value,
                model=model_with_gradiend,
                config=config_dict,
                training_stats=training_stats,
                eval_result=None,
                control=control,
                last_iteration=last_iteration,
                epoch=epoch,
                losses=losses,
                last_losses=last_losses,
                best_score_checkpoint={
                    'correlation': _corr,
                    'global_step': global_step,
                    'epoch': epoch,
                    'loss': loss_value,
                },
            )

            step_start = time.time()
            for cb in all_callbacks:
                out = cb.on_step_end(**step_kwargs)
                if out is not None:
                    eval_result = out
                    step_kwargs['eval_result'] = eval_result
            time_stats['eval'] += time.time() - step_start
            if max_iter is not None and global_step >= max_iter:
                break
            
            # Restart timer
            data_prep_start = time.time()
        
        # End of epoch
        time_stats['total'] = time.time() - total_training_time_start

        # Epoch-end callbacks (CheckpointCallback saves final model)
        _corr = training_stats.get('correlation', -1.0)
        training_info = {
            'losses': losses,
            'best_score_checkpoint': {
                'correlation': _corr,
                'global_step': global_step,
                'epoch': epoch,
            },
            'training_stats': training_stats,
            'training_args': config_dict,
            'time': time_stats,
        }
        epoch_end_kwargs = dict(
            epoch=epoch,
            model=model_with_gradiend,
            config=config_dict,
            time_stats=time_stats,
            **training_info,
        )
        for cb in all_callbacks:
            cb.on_epoch_end(**epoch_end_kwargs)

        if control.get("should_stop"):
            break

        if training_args.num_train_epochs > 1 and not (training_args.save_only_best or training_args.delete_models):
            output_epoch = f'{training_args.output_dir or ""}_epoch_{epoch + 1}'
            model_with_gradiend.save_pretrained(output_epoch, training=training_info)
    
    # Safety net: ensure the final training step was evaluated (e.g. eval failed mid-step
    # or returned no results, so last_eval_step was not committed).
    final_step = training_stats.get('global_step', 0)
    scores = training_stats.get('scores', {})
    if (
        training_args.do_eval
        and training_args.evaluate_fn is not None
        and final_step > 0
        and final_step not in scores
    ):
        logger.info('Running final evaluation at step %s ...', final_step)
        _corr_final = training_stats.get('correlation', -1.0)
        final_loss = losses[-1] if losses else 0.0
        step_kwargs_final = dict(
            step=final_step,
            loss=final_loss,
            model=model_with_gradiend,
            config=config_dict,
            training_stats=training_stats,
            eval_result=None,
            control=control,
            last_iteration=True,
            epoch=last_epoch,
            losses=losses,
            last_losses=last_losses,
            best_score_checkpoint={
                'correlation': _corr_final,
                'global_step': final_step,
                'epoch': last_epoch,
                'loss': final_loss,
            },
        )
        eval_start = time.time()
        for cb in all_callbacks:
            out = cb.on_step_end(**step_kwargs_final)
            if out is not None:
                step_kwargs_final['eval_result'] = out
        time_stats['eval'] += time.time() - eval_start

    # Train-end callbacks
    for cb in all_callbacks:
        cb.on_train_end(config=config_dict, training_stats=training_stats)

    # Best checkpoint info (from callback; full training_stats stay in memory)
    best_score_checkpoint = {"correlation": training_stats.get("correlation", -1.0), "global_step": None, "epoch": None}
    for cb in all_callbacks:
        if isinstance(cb, CheckpointCallback):
            selection_metric = getattr(cb, "_selection_metric", None) or (
                "loss" if getattr(cb, "use_loss_for_best", False) else "correlation"
            )
            if selection_metric in {"auroc", "auc", "roc-auc"}:
                selection_metric = "roc_auc"
            if selection_metric in {"min_auc", "auc_min", "roc_auc_min", "min_auc_no", "min(auc_n,auc_o)"}:
                selection_metric = "min_auc_n_o"
            if selection_metric == "loss" or getattr(cb, "use_loss_for_best", False):
                best_score_checkpoint = {
                    "correlation": None,
                    "roc_auc": None,
                    "min_auc_n_o": None,
                    "loss": cb.best_score,
                    "global_step": cb.best_step,
                    "epoch": cb.best_epoch,
                    "selection_metric": "loss",
                }
                logger.info(f"Training completed (loss selection). Best loss: {cb.best_score:.6f}")
            elif selection_metric in {"roc_auc", "min_auc_n_o"}:
                best_score_checkpoint = {
                    "correlation": training_stats.get("correlation"),
                    "roc_auc": training_stats.get("roc_auc"),
                    "min_auc_n_o": training_stats.get("min_auc_n_o"),
                    "global_step": cb.best_step,
                    "epoch": cb.best_epoch,
                    "selection_metric": selection_metric,
                }
                best_score_checkpoint[selection_metric] = cb.best_score
                label = "AUROC" if selection_metric == "roc_auc" else "min(auc_n,auc_o)"
                if cb.best_score is None:
                    logger.info(f"Training completed. No encoder {label} score was recorded.")
                else:
                    logger.info(f"Training completed. Best {label}: {cb.best_score:.6f}")
                if cb.best_step == 0:
                    logger.info("Training did not improve on the initial evaluation; the best checkpoint is step 0.")
            else:
                best_score_checkpoint = {
                    "correlation": cb.best_score,
                    "roc_auc": training_stats.get("roc_auc"),
                    "min_auc_n_o": training_stats.get("min_auc_n_o"),
                    "global_step": cb.best_step,
                    "epoch": cb.best_epoch,
                    "selection_metric": "correlation",
                }
                if cb.best_score is None:
                    logger.info("Training completed. No encoder evaluation score was recorded.")
                else:
                    logger.info(f"Training completed. Best correlation: {cb.best_score:.6f}")
                if cb.best_step == 0:
                    logger.info("Training did not improve on the initial evaluation; the best checkpoint is step 0.")
            break
    else:
        best_corr = best_score_checkpoint.get("correlation") or training_stats.get("correlation", -1.0)
        logger.info(f"Training completed. Best correlation: {best_corr:.6f}")

    # Check convergence status and warn if non-convergent
    convergent_metric = (training_args.convergent_metric or ("loss" if training_args.supervised_decoder else "correlation")).lower()
    if convergent_metric in {"auroc", "auc", "roc-auc"}:
        convergent_metric = "roc_auc"
    if convergent_metric in {"min_auc", "auc_min", "roc_auc_min", "min_auc_no", "min(auc_n,auc_o)"}:
        convergent_metric = "min_auc_n_o"
    threshold = training_args.convergent_score_threshold
    min_convergent_seeds = training_args.min_convergent_seeds
    
    converged = True
    convergent_count = None
    min_target_class_abs_mean = None
    best_component_summary = None
    best_components = None
    if threshold is not None and min_convergent_seeds is not None and min_convergent_seeds > 0:
        # Check if this single-seed run converged
        best_step_ok = _best_checkpoint_step_is_after_initial(best_score_checkpoint)
        sign_ok = True
        target_mean_product = None
        component_run = component_run_from_training_stats(
            {
                "training_stats": training_stats,
                "best_score_checkpoint": best_score_checkpoint,
            },
            threshold=threshold,
            mean_threshold=training_args.convergent_mean_by_class_threshold,
        )
        if isinstance(component_run, dict) and isinstance(component_run.get("summary"), dict):
            best_component_summary = component_run["summary"]
            best_components = {
                "summary": best_component_summary,
                "metrics_by_component": component_run.get("metrics_by_component"),
                "convergence_by_component": component_run.get("convergence_by_component"),
            }
        else:
            best_components = best_step_component_payload(training_stats, best_score_checkpoint)
            best_component_summary = _best_step_component_summary(training_stats, best_score_checkpoint)
        if convergent_metric == "loss":
            metric_val = best_score_checkpoint.get("loss")
            if metric_val is None:
                metric_val = training_stats.get("loss")
            converged = best_step_ok and metric_val is not None and metric_val <= threshold
        elif isinstance(best_component_summary, dict) and best_component_summary.get("n_components"):
            metric_val = best_component_summary.get("correlation_mean")
            converged = bool(best_component_summary.get("converged"))
            min_target_class_abs_mean = best_component_summary.get("min_target_class_abs_mean")
        elif convergent_metric in {"roc_auc", "min_auc_n_o"}:
            metric_val = best_score_checkpoint.get(convergent_metric)
            if metric_val is None:
                metric_val = training_stats.get(convergent_metric)
            if isinstance(metric_val, dict):
                step = best_score_checkpoint.get("global_step")
                metric_val = metric_val.get(step, metric_val.get(str(step))) if step is not None else None
            # Optional bipolar mean gate only when explicitly configured.
            mean_ok = True
            if training_args.convergent_mean_by_class_threshold is not None:
                target_mean_product = _best_step_target_class_mean_product(training_stats, best_score_checkpoint)
                min_target_class_abs_mean = _best_step_min_target_class_abs_mean(
                    training_stats, best_score_checkpoint
                )
                mean_ok = (
                    isinstance(min_target_class_abs_mean, (int, float))
                    and min_target_class_abs_mean >= training_args.convergent_mean_by_class_threshold
                )
                sign_ok = isinstance(target_mean_product, (int, float)) and target_mean_product < 0
            converged = (
                best_step_ok
                and metric_val is not None
                and float(metric_val) >= threshold
                and mean_ok
                and sign_ok
            )
        else:
            metric_val = best_score_checkpoint.get("correlation")
            if metric_val is None:
                metric_val = training_stats.get("correlation")
            target_mean_product = _best_step_target_class_mean_product(training_stats, best_score_checkpoint)
            if training_args.convergent_mean_by_class_threshold is not None:
                min_target_class_abs_mean = _best_step_min_target_class_abs_mean(
                    training_stats, best_score_checkpoint
                )
            mean_ok = (
                training_args.convergent_mean_by_class_threshold is None
                or (
                    isinstance(min_target_class_abs_mean, (int, float))
                    and min_target_class_abs_mean >= training_args.convergent_mean_by_class_threshold
                )
            )
            sign_ok = isinstance(target_mean_product, (int, float)) and target_mean_product < 0
            converged = best_step_ok and metric_val is not None and abs(metric_val) >= threshold and mean_ok and sign_ok
        
        if not converged:
            if isinstance(best_component_summary, dict) and best_component_summary.get("n_components"):
                method_name = getattr(model_with_gradiend.gradiend, "method_name", "GRADIEND")
                component_fragment = strip_component_prefix(
                    format_component_convergence_fragment(
                        best_components,
                        summary=best_component_summary,
                        show_blockers=True,
                    )
                )
                criteria = []
                if isinstance(metric_val, (int, float)):
                    criteria.append(f"mean component correlation={metric_val:.4f} (threshold={threshold:.4f})")
                if training_args.convergent_mean_by_class_threshold is not None:
                    if isinstance(min_target_class_abs_mean, (int, float)):
                        criteria.append(
                            "minimum per-component target-class |mean|="
                            f"{min_target_class_abs_mean:.4f} "
                            f"(required >= {training_args.convergent_mean_by_class_threshold:.4f})"
                        )
                    else:
                        criteria.append(
                            "minimum per-component target-class |mean| was unavailable "
                            f"(required >= {training_args.convergent_mean_by_class_threshold:.4f})"
                        )
                criteria_text = "; ".join(criteria) if criteria else f"metric={convergent_metric}"
                logger.warning(
                    "Training completed but component-split %s did not converge: "
                    "%s; convergence policy requires all components "
                    "(required: %s convergent seeds). Component criteria: %s.",
                    method_name,
                    component_fragment,
                    min_convergent_seeds,
                    criteria_text,
                )
            elif not best_step_ok:
                logger.warning(
                    "Training completed but model did not converge: "
                    "best checkpoint global_step=%s; convergence requires global_step > 0 "
                    "(required: %s convergent seeds).",
                    best_score_checkpoint.get("global_step"),
                    min_convergent_seeds,
                )
            elif (
                convergent_metric != "loss"
                and metric_val is not None
                and abs(metric_val) >= threshold
                and training_args.convergent_mean_by_class_threshold is not None
                and isinstance(min_target_class_abs_mean, (int, float))
                and min_target_class_abs_mean < training_args.convergent_mean_by_class_threshold
            ):
                logger.warning(
                    "Training completed but model did not converge: "
                    "%s=%.4f met the threshold %.4f, but min |mean| among target classes=%.4f "
                    "was below the required minimum %.4f (required: %s convergent seeds).",
                    convergent_metric,
                    metric_val,
                    threshold,
                    min_target_class_abs_mean,
                    training_args.convergent_mean_by_class_threshold,
                    min_convergent_seeds,
                )
            elif convergent_metric != "loss" and metric_val is not None and abs(metric_val) >= threshold and not sign_ok:
                if isinstance(target_mean_product, (int, float)):
                    logger.warning(
                        "Training completed but model did not converge: "
                        "%s=%.4f met the threshold %.4f, but the two target-class mean encodings "
                        "had the same sign (product=%.4f; required: negative, required: %s convergent seeds).",
                        convergent_metric,
                        metric_val,
                        threshold,
                        target_mean_product,
                        min_convergent_seeds,
                    )
                else:
                    logger.warning(
                        "Training completed but model did not converge: "
                        "%s=%.4f met the threshold %.4f, but the two target-class mean encodings "
                        "were unavailable at the best checkpoint (required: %s convergent seeds).",
                        convergent_metric,
                        metric_val,
                        threshold,
                        min_convergent_seeds,
                    )
            else:
                logger.warning(
                    "Training completed but model did not converge: "
                    "%s=%.4f (threshold=%.4f, required: %s convergent seeds). ",
                    convergent_metric,
                    metric_val if metric_val is not None else float('nan'),
                    threshold,
                    min_convergent_seeds,
                )
        convergent_count = 1 if converged else 0
    
    convergence_info = None
    if threshold is not None:
        convergence_info = {
            "converged": converged,
            "convergent_count": convergent_count,
            "min_convergent_seeds": min_convergent_seeds,
            "convergence_metric": convergent_metric,
            "threshold": threshold,
            "convergent_mean_by_class_threshold": training_args.convergent_mean_by_class_threshold,
            "convergent_min_target_class_abs_mean": min_target_class_abs_mean,
        }
        if best_component_summary is not None:
            convergence_info["convergence_unit"] = "component"
            convergence_info["component_summary"] = best_component_summary
            if isinstance(best_components, dict):
                # Persist the local-best-per-component view used for the decision,
                # so load-time warnings do not fall back to the global best step.
                convergence_info["components"] = best_components

    if getattr(training_args, "fail_on_non_convergence", False) and int(getattr(training_args, "max_seeds", 1) or 1) <= 1:
        min_req = training_args.min_convergent_seeds
        if min_req is not None and min_req > 0:
            actual = int(convergent_count or 0) if convergent_count is not None else (1 if converged else 0)
            if actual < min_req:
                raise RuntimeError(
                    format_non_convergence_error(
                        actual=actual,
                        min_required=min_req,
                        training_args=training_args,
                        model=model_with_gradiend,
                        output_dir=training_args.output_dir,
                        components=best_components,
                        component_summary=best_component_summary,
                        convergence_metric=convergent_metric,
                        threshold=threshold,
                    )
                )
    
    # Handle keep_only_best
    output_dir = training_args.output_dir or ""
    if training_args.save_only_best:
        _handle_keep_only_best(output_dir)
        finalize_component_best_states(output_dir)
        # Component-split: delivered weights must be the local-best merge, not the
        # single global best-correlation step that CheckpointCallback saved.
        stitched = apply_saved_component_best_states(model_with_gradiend, output_dir)
        if stitched:
            logger.info(
                "Applied local-best component states for %s components into final model at %s",
                len(stitched),
                output_dir,
            )
            model_with_gradiend.save_pretrained(output_dir)
            if isinstance(convergence_info, dict):
                convergence_info["component_stitching"] = {
                    "applied": True,
                    "status": "intra_seed_component_best",
                    "n_components": len(stitched),
                }
        # Overwrite training.json with full run history (best checkpoint was saved with truncated stats)
        write_training_stats(
            output_dir,
            training_stats=training_stats,
            best_score_checkpoint=best_score_checkpoint,
            training_args=config_dict,
            time_stats=time_stats,
            losses=losses,
            convergence_info=convergence_info,
        )
    else:
        finalize_component_best_states(output_dir)
        stitched = apply_saved_component_best_states(model_with_gradiend, output_dir)
        if stitched:
            logger.info(
                "Applied local-best component states for %s components into final model at %s",
                len(stitched),
                output_dir,
            )
            model_with_gradiend.save_pretrained(output_dir)
            if isinstance(convergence_info, dict):
                convergence_info["component_stitching"] = {
                    "applied": True,
                    "status": "intra_seed_component_best",
                    "n_components": len(stitched),
                }
        # When save_only_best is False, CheckpointCallback saves training.json via save_pretrained.
        # We need to update it with convergence_info. Load existing training.json and rewrite with convergence_info.
        try:
            existing_stats = load_training_stats(output_dir)
            if existing_stats:
                write_training_stats(
                    output_dir,
                    training_stats=existing_stats.get("training_stats", training_stats),
                    best_score_checkpoint=existing_stats.get("best_score_checkpoint", best_score_checkpoint),
                    training_args=existing_stats.get("training_args", config_dict),
                    time_stats=existing_stats.get("time", time_stats),
                    losses=existing_stats.get("losses", losses),
                    convergence_info=convergence_info,
                )
        except Exception as e:
            logger.debug("Could not update convergence_info in training.json: %s", e)

    # Handle delete_models
    if training_args.delete_models:
        _delete_model_files(training_args.output_dir or "")
    
    # Release memory
    del model_with_gradiend
    gc.collect()
    torch.cuda.empty_cache()
    
    return training_args.output_dir or ""


def _handle_keep_only_best(output: str):
    """Handle keeping only the best model."""
    def _move_dir(src: str, dst: str) -> None:
        if not os.path.exists(src):
            return
        if os.path.exists(dst):
            shutil.rmtree(dst)
        try:
            os.rename(src, dst)
        except OSError:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            shutil.rmtree(src)
    
    best_output = f'{output}_best'
    
    # If best model exists, use it; otherwise keep the final model
    if os.path.exists(best_output):
        output_temp = f'{output}_temp'
        if os.path.exists(output):
            _move_dir(output, output_temp)
        
        _move_dir(best_output, output)
        
        # Copy PDF and PNG files from temp
        if os.path.exists(output_temp):
            for file in os.listdir(output_temp):
                if file.endswith(('.pdf', '.png')):
                    shutil.copy(os.path.join(output_temp, file), output)
            
            # Copy subdirectories
            for folder in os.listdir(output_temp):
                folder_path = os.path.join(output_temp, folder)
                if os.path.isdir(folder_path):
                    shutil.copytree(folder_path, os.path.join(output, folder), dirs_exist_ok=True)
            
            shutil.rmtree(output_temp)
    # If no best model was saved, keep the final model (already at output dir)


def _delete_model_files(output: str):
    """Delete model files from output directory (.bin and .safetensors)."""
    if os.path.exists(output):
        for file in os.listdir(output):
            path = os.path.join(output, file)
            if file.endswith(".bin") or file.endswith(".safetensors"):
                os.remove(path)
