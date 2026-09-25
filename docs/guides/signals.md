# Signals, scopes and components

GRADIEND learns a feature from a *signal* measured on a base model. Three
independent choices define a training run:

| Axis | Question | API |
|------|----------|-----|
| **Signal** | *What* is measured? | [`Signal`][gradiend.trainer.core.signals.Signal] |
| **Scope** | *Where* is it measured? | [`SignalScope`][gradiend.trainer.core.signals.SignalScope] |
| **Split** | How is the resolved space partitioned into components? | [`GradiendSplit`][gradiend.gradiend_split.GradiendSplit] |

All three are passed through [`TrainingArguments`][gradiend.trainer.core.arguments.TrainingArguments]
(`signal`, `signal_scope`, `gradiend_split`). The defaults reproduce classic
GRADIEND: raw parameter gradients of the backbone, one unpartitioned model.

```python
from gradiend import GradiendSplit, Signal, SignalScope, TrainingArguments

args = TrainingArguments(
    signal=Signal.activation(),            # what: hidden activations (ACTIEND)
    signal_scope=SignalScope.layer(9),     # where: output of transformer layer 9
    gradiend_split=GradiendSplit.none(),   # how: one model over the whole space
)
```

A model trained on activations is called **ACTIEND**; a model trained on
parameter gradients is **GRADIEND**. Everything else (data, evaluation,
multi-seed training, decoder evaluation) works the same way.

## Signals

| Signal | Measures | Typical use |
|--------|----------|-------------|
| `Signal.gradient()` | Parameter gradient of the prediction loss (default) | GRADIEND, weight rewriting |
| `Signal.activation(token_selector=...)` | Hidden activation at the selected tokens | ACTIEND, activation steering |
| `Signal.activation_gradient(token_selector=...)` | Loss gradient with respect to the activation | Attribution-style features |

Strings (`"gradient"`, `"activation"`) and dictionaries are accepted wherever a
`Signal` is expected. Exactly one signal per training run is currently supported.

### Token selection (activation signals)

`token_selector` chooses which token positions of a site are read:

| Selector | Positions |
|----------|-----------|
| `"all"` / `"mean"` | Attention-masked mean over all tokens |
| `"cls"` | The classification token |
| an `int` | One fixed token index |
| `"mask"` | The tokenizer's mask token |
| `"prediction"` | The (filled) prediction slot of the training example |
| `"pre_prediction"` | The token *before* the prediction slot (context that cannot see the fill) |
| a callable `(activation, inputs) -> tensor` | Custom; the width must be stable |

`target_token_selector` reads a different position for the decoder target than
for the encoder input (mixed-site ACTIEND), for example
`Signal.activation(token_selector="pre_prediction", target_token_selector="prediction")`.

`scale="running_rms"` divides each site by an online root-mean-square estimate
(only `O(n_sites)` floats of state) so activation magnitudes are comparable
across sites and models.

## Scopes

`SignalScope` selects the eligible parameters (gradient signals) or modules
(activation signals):

| Constructor | Meaning |
|-------------|---------|
| `SignalScope.default()` | Backbone / text tower without prediction heads |
| `SignalScope.full()` | The full model including heads |
| `SignalScope.layers()` / `SignalScope.layers(0, 3)` | Transformer-layer outputs (all, or the listed indices) |
| `SignalScope.layer(9)` | One transformer-layer output |
| `SignalScope.embeddings()` | The combined embedding stream |
| `SignalScope.word_embedding()` | The token-embedding lookup |
| `SignalScope.from_values(params=[...])` | Explicit parameter names/wildcards (gradient signals) |
| `SignalScope.from_values(activation_sites=[...])` | Explicit module names/wildcards (activation signals) |

The semantic shortcuts (`layers`, `layer`, `embeddings`, `word_embedding`) are
resolved through a model-topology adapter and work for both signal kinds. They
support BERT-like, DistilBERT, GPT-2-like, Llama-like, OPT, GPT-NeoX and Gemma 3
models; for other architectures use an explicit `from_values(...)` list (the
error message shows a ready-to-copy example).

## Component splits

`GradiendSplit` partitions the *already resolved* input space into virtual
components. The components share one physical encoder/decoder matrix but are
trained (and reported) separately:

| Split | Components |
|-------|-----------|
| `GradiendSplit.none()` | No partition (default) |
| `GradiendSplit.single()` | One component covering the full space |
| `GradiendSplit.by_tensor()` | One component per resolved tensor (parameter tensor or activation site) |

`TrainingArguments(gradiend_split_loss=...)` selects how component losses are
aggregated (`"mean"`, `"sum"`, `"size_weighted"`, or `"full"`). Multi-seed runs
select and stitch the best seed per component; encoder evaluation and the
plots in [`plot_encoder_component_artifacts`][gradiend.visualizer.components.plot_encoder_component_artifacts]
report each component.

## Interventions and saved models

```python
with model_with_gradiend.intervene(value=0.5, signal="auto", part="decoder"):
    outputs = model_with_gradiend(**inputs)          # temporary, rolled back on exit

modified = trainer.modify_model(decoder_results=stats, target_class="3SG")
modified.save_pretrained_modified("./modified-actiend")  # ACTIEND: fixed steering hooks

from gradiend import load_modified_model
reloaded = load_modified_model("./modified-actiend", model_loader=my_loader)
```

`modify_model` is the general entry point. For gradient signals it returns a
weight-rewritten copy of the base model (identical to `rewrite_base_model`); for
activation signals it returns a copy with fixed forward hooks that add the
decoded update at the trained sites. Hook placement is controlled by
`token_selector` and the optional `activation_gate`; the encoder-driven
selectors (`"encoder_direction"`, `"encoder_abs"`, `"encoder_threshold"`,
`"encoder_range"`) fire only on tokens whose encoding matches the intended
direction. The same arguments are accepted by `evaluate_decoder`, so a model
saved with `modify_model` uses exactly the policy that was evaluated.

## Training on precomputed signal vectors

[`SignalTrainer`][gradiend.trainer.signal_trainer.SignalTrainer] trains a
`ModelWithGradiend` on signal vectors you extracted yourself. It inherits the full
training lifecycle (checkpointing, caching, multi-seed, convergence) and supports
encoder evaluation; decoder evaluation is unavailable because it needs task inputs.

```python
from gradiend import SignalTrainer, TrainingArguments, Signal

trainer = SignalTrainer(
    model_with_gradiend,        # a preconstructed ModelWithGradiend of matching signal kind
    positive_vectors,           # tensor of shape (n, input_dim)
    negative_vectors,
    args=TrainingArguments(signal=Signal.activation(), source="both", target="diff"),
)
trainer.train()
```

## Related

- [Training arguments](training-arguments.md) — all fields, including `signal`, `signal_scope`, `gradiend_split`
- [Saving & loading](saving-loading.md)
- [Cross-signal IEND plan](../design/cross-signal-iends.md) (design document)
