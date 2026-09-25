---
language:
- en
license: cc-by-3.0
task_categories:
- fill-mask
tags:
- gradiend
- nlp
- mlm
- sentiment
- nrc
pretty_name: GRADIEND English Sentiment (NRC Adjective) Data
configs:
- config_name: default
  data_files:
  - split: train
    path: default/train.parquet
  - split: validation
    path: default/validation.parquet
  - split: test
    path: default/test.parquet
- config_name: split
  data_files:
  - split: train
    path: split/train.parquet
  - split: validation
    path: split/validation.parquet
  - split: test
    path: split/test.parquet
---

# GRADIEND English Sentiment (NRC Adjective) Data

Masked tweet contexts where the masked word is a **sentiment adjective**:
top 10 adjectives per valence attested as spaCy `ADJ` in
[`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
(`sentiment`), with polarity taken from the NRC Emotion Lexicon
(Mohammad & Turney, 2013) for target selection.

Frozen training artifact for
[`gradiend.examples.train_sentiment`](https://github.com/aieng-lab/gradiend/blob/main/gradiend/examples/train_sentiment.py).

**Not a discrete emotion taxonomy** (joy/anger/…). Binary **polarity** cloze
over adjectives.

Companion neutrals:
[`aieng-lab/en-sentiment-nrc-neutral`](https://huggingface.co/datasets/aieng-lab/en-sentiment-nrc-neutral).

## Configs (subsets)

| Config | Split scheme |
|--------|----------------|
| **`default`** | `split` column as written by `generate_data` |
| **`split`** | Vocabulary-held-out by adjective lemma (60% / 20% / 20% of adjectives per class, `seed=0`) via `apply_vocabulary_held_out_split` — same as before training in `train_sentiment` |

```python
from datasets import load_dataset

# As generated
ds = load_dataset("aieng-lab/en-sentiment-nrc", "default", split="train")

# Package-paper vocabulary-held-out splits
ds = load_dataset("aieng-lab/en-sentiment-nrc", "split", split="train")
# or: load_dataset("aieng-lab/en-sentiment-nrc", split="train")  # if default is enough
```

Under config `split`, each target adjective appears in **exactly one** of
train / validation / test.

## Dataset Details

### Classes

| `label_class` | Description |
|---------------|-------------|
| `positive` | top-10 positive polarity adjectives attested in tweet_eval |
| `negative` | top-10 negative polarity adjectives attested in tweet_eval |

Ambiguous NRC polarity words (listed as both pos and neg) are dropped before
ranking. Classes are balanced (`balance="strict"`).

### Size

| Quantity | Value |
|----------|-------|
| Adjectives per class | 10 |
| Rows per adjective | **50** |
| Rows per class | 500 (`positive` / `negative`) |
| Labeled total | 1000 |

Fifty rows per adjective is intentional: `min_count_per_word=50` matches the
lexicon attestation floor, and the rarest top-10 ADJs in tweet_eval only yield
~50–60 unique maskable hits under spaCy `ADJ`. Larger `max_size_per_class`
values (e.g. 3000) do not produce a meaningfully larger *unique* set under
strict balance.

### Structure

- `masked`: context with the target adjective replaced by `[MASK]`
- `split`: `train` / `validation` / `test`
- `label_class`: `positive` or `negative`
- `label`: mask target adjective
- `feature_class_id`: equal to `label_class`

Extra unified GRADIEND columns may be present; treat the fields above as the
public API.

### Sources

| Resource | Role |
|----------|------|
| [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) `sentiment` | base texts (CC BY 3.0; Twitter ToS) |
| [NRC Emotion Lexicon](https://saifmohammad.com/WebPages/AccessResource.htm) | polarity used to *select* adjective targets (lexicon file not redistributed) |
| spaCy `en_core_web_sm` | ADJ filter |

## Dataset Creation

```text
gradiend.examples.train_sentiment.generate_data
  require_adjectives=True
  max_words_per_class=10
  min_count_per_word=50
  max_size_per_class=500   # → 50 rows / adjective after strict balance
  balance=strict
  seed=0
```

HF staging writes both configs:
`python scripts/upload_sentiment_hf_datasets.py`
(see `scripts/README_sentiment_hf_upload.md`).

## Bias, Risks, and Limitations

- Tweet / social-media domain bias.
- Polysemy (e.g. `cool`, `top`, `real`, `crazy`).
- Not a validated psychological emotion instrument.

## License

**[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)** — same as TweetEval
**sentiment**.

Provide attribution (CC BY). Tweet content remains subject to
[Twitter / X Terms of Service](https://twitter.com/tos), as required by
TweetEval. The NRC Emotion Lexicon is **not** redistributed here; only common
English adjectives appear as labels, selected using NRC polarity. For the full
lexicon, see https://saifmohammad.com/WebPages/AccessResource.htm

## Citation

```bibtex
@inproceedings{drechsel2026gradiend,
  title     = {{GRADIEND}: Feature Learning within Neural Networks Exemplified through Biases},
  author    = {Drechsel, Jonathan and Herbold, Steffen},
  booktitle = {Proceedings of the International Conference on Learning Representations},
  year      = {2026},
  url       = {https://arxiv.org/abs/2502.01406}
}

@article{mohammad2013nrc,
  title   = {Crowdsourcing a Word-Emotion Association Lexicon},
  author  = {Mohammad, Saif M. and Turney, Peter D.},
  journal = {Computational Intelligence},
  volume  = {29},
  number  = {3},
  pages   = {436--465},
  year    = {2013}
}

@inproceedings{barbieri2020tweeteval,
  title     = {{TweetEval}: Unified Benchmark and Comparative Evaluation for Tweet Classification},
  author    = {Barbieri, Francesco and Camacho-Collados, Jose and Espinosa-Anke, Luis and Neves, Leonardo},
  booktitle = {Findings of EMNLP},
  year      = {2020}
}
```

## Dataset Card Authors

jdrechsel
