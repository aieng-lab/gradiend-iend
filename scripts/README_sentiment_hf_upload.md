# Publishing the NRC adjective sentiment datasets — licensing

**Not legal advice.** Practical stance for the package-paper recipe
(`train_sentiment`: top-10 NRC ADJs/valence over TweetEval `sentiment` texts).

## Short answer

**Yes — use the same license as TweetEval sentiment: [CC BY 3.0](https://creativecommons.org/licenses/by/3.0/).**

That matches the text source. The 20 mask-target adjectives are ordinary English
words selected *with help of* NRC polarity; publishing them as class labels is
not redistributing the NRC Emotion Lexicon (the word→emotion association
tables). Cite NRC as the selection method; do not ship the full lexicon file.

Also state (as TweetEval does): use must comply with Twitter / X Terms of
Service for tweet content.

| Piece | License / terms |
|-------|-----------------|
| Cloze dataset (tweet contexts + our masking/splits) | **CC BY 3.0** (same as TweetEval sentiment) |
| NRC | Cited for adjective *selection*; lexicon file not redistributed |
| Twitter / X | ToS still apply to tweet text (TweetEval requirement) |

HF front-matter: `license: cc-by-3.0`

## Why CC BY 3.0 is appropriate here

1. **Tweet text** comes from `cardiffnlp/tweet_eval` **sentiment**, which TweetEval
   documents as [CC BY 3.0 Unported](https://creativecommons.org/licenses/by/3.0/).
2. **GRADIEND work** (masking, vocab-held-out splits, neutrals, cards) can be
   released under the same CC BY 3.0 so the artifact has one clear license.
3. **NRC** forbids redistributing *the lexicon data* (the association resource).
   Our release does **not** include that resource—only ~20 common adjectives as
   `label` / `label_class`, chosen via NRC + corpus attestation. That is normal
   research practice (method citation), not a dump of EmoLex.

## Attribution checklist (CC BY)

- TweetEval / SemEval sentiment + Barbieri et al. (TweetEval paper)
- Mohammad & Turney (NRC Emotion Lexicon) — selection method
- GRADIEND (Drechsel & Herbold) — dataset construction
- Comply with Twitter / X ToS when redistributing or using tweet text

## What not to do

- Do **not** upload `vladinc/nrc` or the full NRC association tables.
- Do **not** mark the repo CC BY-SA 4.0 unless you intentionally want share-alike
  (TweetEval sentiment is BY 3.0, not BY-SA).

## Configs / splits

One labeled HF dataset with **two configs** (subsets):

| Config | Meaning |
|--------|---------|
| `default` | `split` as written by `generate_data` |
| `split` | vocabulary-held-out by adjective (60/20/20, `seed=0`) |

```python
load_dataset("aieng-lab/en-sentiment-nrc", "default")
load_dataset("aieng-lab/en-sentiment-nrc", "split")
```

Neutrals stay a separate repo (`en-sentiment-nrc-neutral`).

## Size (why 50 rows per adjective)

Labeled release size under the package recipe:

| Quantity | Value |
|----------|-------|
| Adjectives per class | 10 (top-10 NRC ADJ/valence in tweet_eval) |
| Rows per adjective | **50** |
| Rows per class | 500 |
| Labeled total | 1000 |

`balance="strict"` equalizes counts across the 10 targets in each class.
`min_count_per_word=50` is both the lexicon attestation floor and the
per-adjective row floor. Raising `max_size_per_class` toward 3k does **not**
unlock much more unique data: the rarest top-10 ADJs in tweet_eval only have
~50–60 unique maskable ADJ hits, so further growth would mostly be
replacement duplicates. We therefore set `max_size_per_class=500`
(10 × 50) and publish the even **50/adjective** artifact.

Generation is spaCy-bound (~minutes over ~32k tweet_eval texts); fine for a
one-shot build, not a tight inner loop.

## Commands

```bash
python scripts/upload_sentiment_hf_datasets.py --dry-run
python scripts/upload_sentiment_hf_datasets.py --regenerate --dry-run
python scripts/upload_sentiment_hf_datasets.py --regenerate
```
