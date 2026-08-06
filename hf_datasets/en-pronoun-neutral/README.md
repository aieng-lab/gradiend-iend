---
language:
- en
license: cc-by-sa-4.0
task_categories:
- fill-mask
tags:
- gradiend
- nlp
- mlm
- pronouns
pretty_name: GRADIEND English Pronoun Neutral Data
---

# GRADIEND English Pronoun Neutral Data

This dataset is a filtered English Wikipedia sample containing sentence windows
without the English pronouns used by the GRADIEND English Pronoun Data dataset.

## Usage

```python
from datasets import load_dataset

neutral = load_dataset("aieng-lab/en-pronoun-neutral", split="train")
texts = neutral["text"]
```

The dataset has one split: `train`.

## Dataset Details

### Dataset Description

This dataset is intended as neutral evaluation data for English pronoun
GRADIEND experiments. Rows are sentence windows from English Wikipedia that do
not contain any of the configured pronoun exclusion words:

`i`, `we`, `you`, `he`, `she`, `it`, `they`, `me`, `us`, `him`, `her`, `them`.

### Dataset Structure

- `text`: the neutral sentence window

### Dataset Sources

- Repository: https://github.com/aieng-lab/gradiend
- Original Data: `wikimedia/wikipedia`, config `20231101.en`

## Dataset Creation

The data is generated with `gradiend.examples.create_english_pronoun_data`.
Generation streams English Wikipedia, uses non-overlapping two-sentence
windows, keeps windows between 20 and 200 characters, excludes the configured
English pronouns, and collects up to 10,000 neutral examples.

## Bias, Risks, and Limitations

This dataset is neutral only with respect to the configured pronoun exclusion
list. It is not generally neutral with respect to other social categories,
topics, names, entities, or linguistic attributes. The examples inherit
Wikipedia's coverage and writing-style biases.

## Citation

BibTeX:

```bibtex
@inproceedings{drechsel2026gradiend,
  title     = {{GRADIEND}: Feature Learning within Neural Networks Exemplified through Biases},
  author    = {Drechsel, Jonathan and Herbold, Steffen},
  booktitle = {Proceedings of the International Conference on Learning Representations},
  year      = {2026},
  url       = {https://arxiv.org/abs/2502.01406}
}
```

## Dataset Card Authors

jdrechsel
