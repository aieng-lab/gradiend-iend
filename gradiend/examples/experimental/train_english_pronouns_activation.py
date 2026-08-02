"""
English pronoun workflow using an activation signal instead of raw gradients.

This mirrors gradiend/examples/train_english_pronouns.py, but trains GRADIEND on
the filled prediction-span activation from BERT's default activation scope.

Requires: pip install gradiend[data]
"""

import os

import torch

from gradiend import (
    load_modified_model,
    Signal,
    TextPredictionConfig,
    TextPredictionTrainer,
    TrainingArguments,
)
from gradiend.examples.create_english_pronoun_data import ensure_english_pronoun_data
from gradiend.gradiend_split import GradiendSplit
from gradiend.trainer.core import SignalScope
from gradiend.trainer.text.common.loading import AutoModelForLM

DATA_DIR = "data/english_pronouns"
MODEL_NAME = "bert-base-uncased"
MODEL_NAME = "gpt2"

RUN_MODIFY_MODEL_SMOKE = True
MODIFY_TARGET_CLASS = "3SG"
MODIFIED_OUTPUT_DIR = "runs/english_pronouns_activation/modified_3sg"
MODIFY_SMOKE_MAX_SIZE = 8
RUN_TOKEN_ENCODING_EXAMPLE = True
TOKEN_ENCODING_OUTPUT = "token_encoding_highlight.html"
TOKEN_ENCODING_TEXT = "The teacher said he would arrive soon"


def _short_eval_summary(result):
    keys = ("feature_score", "lms", "accuracy", "mean_probability", "loss")
    return {key: result.get(key) for key in keys if key in result}


def _model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _prompt_logits(model, tokenizer, prompt):
    device = _model_device(model)
    was_training = bool(getattr(model, "training", False))
    model.eval()
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        logits = model(**encoded).logits.detach().cpu()
    if was_training:
        model.train()
    return logits


def _print_hook_probe(label, reference_model, candidate_model, tokenizer, prompt):
    reference_logits = _prompt_logits(reference_model, tokenizer, prompt)
    candidate_logits = _prompt_logits(candidate_model, tokenizer, prompt)
    max_delta = (candidate_logits - reference_logits).abs().max().item()
    handles = getattr(candidate_model, "_gradiend_modified_hook_handles", []) or []
    tensors = getattr(candidate_model, "_gradiend_modified_tensors", {}) or {}
    tensor_norm = sum(float(tensor.float().norm().item()) for tensor in tensors.values())
    print(
        f"  {label} hook probe: hooks={len(handles)}, "
        f"steering_norm={tensor_norm:.6g}, max_logit_delta={max_delta:.6g}"
    )


if __name__ == "__main__":
    training_path, neutral_path = ensure_english_pronoun_data(output_dir=DATA_DIR)

    config = TextPredictionConfig(
        run_id="pronoun_3sg_3pl_activation",
        data=training_path,
        target_classes=["3SG", "3PL"],
        neutral_data=neutral_path,
    )

    args = TrainingArguments(
        experiment_dir="runs/english_pronouns_activation_factual",
        train_batch_size=1,
        eval_batch_size=1,
        gradiend_batch_size=1,
        num_train_epochs=5,
        max_steps=1000,
        eval_steps=100,
        bias_encoder=False,
        learning_rate=1e-5,
        target="diff",
        source="factual",
        signal=Signal.activation(),
        add_neutral_identity_transitions=True,
        #signal=Signal.gradient(),
        signal_scope=SignalScope.layer(9),
        #signal_scope=SignalScope.from_values(activation_sites=["bert.encoder.layer.10.output"]),
        #gradiend_split=GradiendSplit.by_tensor(),
        fail_on_non_convergence=False,
        use_cache=False,
    )

    print("\n=== Training activation GRADIEND ===")
    print(f"  model: {MODEL_NAME}")
    print("  activation scope: transformer layers")
    print("  token selector: prediction span")

    trainer = TextPredictionTrainer(
        model=MODEL_NAME,
        config=config,
        args=args,
    )
    trainer.train()
    #trainer.plot_training_convergence(class_spread='ci95')

    stats = trainer.get_training_stats()
    ts = stats.get("training_stats", {}) if stats else {}
    print(f"  correlation={ts.get('correlation')}")
    trainer.evaluate_encoder(plot=True)

    if RUN_TOKEN_ENCODING_EXAMPLE:
        print("\n=== Token encoding highlight ===")
        model_with_gradiend = trainer.get_model()
        has_component_split = bool(getattr(model_with_gradiend.gradiend, "has_component_split", False))
        components = getattr(model_with_gradiend.gradiend, "component_slices", ()) if has_component_split else ()
        component = components[0].id if components else None
        html = trainer.visualizer.highlight_token_encoding(
            TOKEN_ENCODING_TEXT,
            component=component,
            show=False,
            color_center="neutral",
            color_extent=1.0,
        )
        output_path = os.path.join(args.experiment_dir, TOKEN_ENCODING_OUTPUT)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            handle.write(html)
        print(f"  text: {TOKEN_ENCODING_TEXT!r}")
        print(f"  component: {component or 'aggregate'}")
        print("  color center: neutral encoder baseline")
        print(f"  wrote: {output_path}")

    if RUN_MODIFY_MODEL_SMOKE:
        print("\n=== ACTIEND modify_model smoke check ===")
        print(f"  selecting decoder settings for {MODIFY_TARGET_CLASS} via evaluate_decoder")
        decoder_stats = trainer.evaluate_decoder(
            target_class=MODIFY_TARGET_CLASS,
            max_size=MODIFY_SMOKE_MAX_SIZE,
            #use_cache=False,
            plot=True,
            lrs=[1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6], # DO NOT REMOVE THESE MANUAL LRs!!!
        )
        chosen = decoder_stats[MODIFY_TARGET_CLASS]
        print(
            f"  chosen for {MODIFY_TARGET_CLASS}: "
            f"feature_factor={chosen['feature_factor']}, lr={chosen['learning_rate']}, "
            f"value={chosen.get('value')}"
        )

        model_with_gradiend = trainer.get_model()
        baseline = trainer.evaluate_base_model(
            model_with_gradiend.base_model,
            model_with_gradiend.tokenizer,
            use_cache=False,
            max_size_training_like=MODIFY_SMOKE_MAX_SIZE,
            max_size_neutral=MODIFY_SMOKE_MAX_SIZE,
        )
        modified = trainer.modify_model(
            decoder_results=decoder_stats,
            target_class=MODIFY_TARGET_CLASS,
        )
        _print_hook_probe(
            "modified",
            model_with_gradiend.base_model,
            modified,
            model_with_gradiend.tokenizer,
            "The teacher said he would arrive soon.",
        )
        modified_eval = trainer.evaluate_base_model(
            modified,
            model_with_gradiend.tokenizer,
            use_cache=False,
            max_size_training_like=MODIFY_SMOKE_MAX_SIZE,
            max_size_neutral=MODIFY_SMOKE_MAX_SIZE,
        )
        print(f"  baseline: {_short_eval_summary(baseline)}")
        print(f"  modified: {_short_eval_summary(modified_eval)}")

        print(f"  saving modified model to {MODIFIED_OUTPUT_DIR}")
        modified.save_pretrained_modified(MODIFIED_OUTPUT_DIR)
        reloaded = load_modified_model(
            MODIFIED_OUTPUT_DIR,
            model_loader=lambda path: AutoModelForLM.from_pretrained(path),
        )
        _print_hook_probe(
            "reloaded",
            model_with_gradiend.base_model,
            reloaded,
            model_with_gradiend.tokenizer,
            "The teacher said he would arrive soon.",
        )
        reloaded_eval = trainer.evaluate_base_model(
            reloaded,
            model_with_gradiend.tokenizer,
            use_cache=False,
            max_size_training_like=MODIFY_SMOKE_MAX_SIZE,
            max_size_neutral=MODIFY_SMOKE_MAX_SIZE,
        )
        print(f"  reloaded: {_short_eval_summary(reloaded_eval)}")

    print("\n=== Done ===")
