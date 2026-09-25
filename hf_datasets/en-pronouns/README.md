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
pretty_name: GRADIEND English Pronoun Data
---

# GRADIEND English Pronoun Data

This dataset consists of masked English Wikipedia sentence windows where the
masked word is an English pronoun.

See `de-gender-case-articles`, GENTER, GRADIEND Race Data, and GRADIEND
Religion Data for similar AI Engineering Lab datasets.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("aieng-lab/en-pronouns", split="train")

masked = ds["masked"]
label = ds["label"]
label_class = ds["label_class"]
```

`split` can be either `train`, `validation`, or `test`.

## Dataset Details

### Dataset Description

This dataset is generated from English Wikipedia and contains five pronoun
classes:

- `1SG`: `I`
- `1PL`: `we`
- `2SGPL`: `you`
- `3SG`: `he`, `she`, `it`
- `3PL`: `they`

Each row contains one masked context and the factual pronoun observed in the
source text. The dataset is intended for masked language modeling experiments
and GRADIEND feature-learning workflows over English pronoun classes.

### Dataset Structure

- `masked`: the sentence window with the pronoun replaced by `[MASK]`
- `split`: the data split (`train`, `validation`, or `test`)
- `label_class`: the pronoun class id
- `label`: the observed pronoun token
- `feature_class_id`: the GRADIEND feature class id, equal to `label_class`

### Dataset Sources

- Repository: https://github.com/aieng-lab/gradiend
- Original Data: `wikimedia/wikipedia`, config `20231101.en`

## Dataset Creation

The data is generated with `gradiend.examples.create_english_pronoun_data`.
Generation streams the full English Wikipedia split, uses non-overlapping
two-sentence windows, keeps windows between 20 and 200 characters, requires at
least five words of left context before the target pronoun, and collects up to
10,000 unique examples per pronoun class. Exact repeated prediction examples
are removed before splitting, and masked prompts with conflicting targets are
excluded. Publication fails if a masked prompt occurs in more than one split.

Rows are split into approximately 80% train, 10% validation, and 10% test per
class. Exact counts for the published revision are in `dataset_summary.json`.

## Bias, Risks, and Limitations

The examples are derived from Wikipedia and therefore reflect Wikipedia's topic
distribution, writing style, and coverage biases. The labels are pronoun tokens
observed in text; they should not be interpreted as annotations of a person,
identity, or referent outside the local language-modeling context.

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
