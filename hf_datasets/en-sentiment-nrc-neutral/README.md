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
pretty_name: GRADIEND English Sentiment (NRC) Neutral Data
---

# GRADIEND English Sentiment (NRC) Neutral Data

Filtered tweet_eval texts with **no** NRC polarity (`positive` / `negative`)
lexicon words — not only the top-20 mask targets used by
[`aieng-lab/en-sentiment-nrc`](https://huggingface.co/datasets/aieng-lab/en-sentiment-nrc).
For neutral evaluation.

## Usage

```python
from datasets import load_dataset

neutral = load_dataset("aieng-lab/en-sentiment-nrc-neutral", split="train")
texts = neutral["text"]
```

One split: `train`.

## Dataset Details

### Description

Neutral evaluation text for GRADIEND polarity experiments. Rows come from
`cardiffnlp/tweet_eval` (`sentiment`) after excluding any sentence that
contains a word from the full NRC positive or negative association lists
(`load_nrc_sentiment_words`), in addition to the configured training
targets. The NRC lexicon file is not redistributed here.

### Structure

- `text`: neutral text

### Sources

- [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval) `sentiment`
- NRC Emotion Lexicon (full polarity lists used for exclusion at generation only)

## Dataset Creation

`gradiend.examples.train_sentiment.generate_data` → `generate_neutral_data`
with `additional_excluded_words=nrc_positive + nrc_negative`
(default `neutral_max_size=1000`).

## Bias, Risks, and Limitations

Neutral only w.r.t. NRC polarity words (and the training targets). Words with
emotion associations but no positive/negative flag are not excluded. Tweet
domain bias applies.

## License

**[CC BY 3.0](https://creativecommons.org/licenses/by/3.0/)** — same as TweetEval
**sentiment**. Attribution required. Twitter / X ToS still apply to tweet text
(as for TweetEval).

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
```

## Dataset Card Authors

jdrechsel
