"""
Showcase: generate NRC adjective sentiment cloze data from tweet_eval.

Builds local ``data/sentiment_tweets/{training,neutral}.csv`` via
``train_sentiment.generate_data`` (top-10 NRC ADJs/valence, strict balance,
full NRC polarity exclusion for neutrals).

Paper training and demos load the published Hugging Face datasets instead::

    python -m gradiend.examples.train_sentiment

Requires: pip install gradiend[data], spaCy ``en_core_web_sm``, and Hub access
to ``cardiffnlp/tweet_eval`` + ``vladinc/nrc``.
"""

from __future__ import annotations

from gradiend.examples.train_sentiment import (
    DEFAULT_DATA_DIR,
    DEFAULT_LEXICON_WORDS_PER_CLASS,
    DEFAULT_MAX_SIZE_PER_CLASS,
    DEFAULT_MIN_OCCURRENCES_PER_TARGET,
    DEFAULT_SEED,
    generate_data,
)


if __name__ == "__main__":
    train_path, neutral_path = generate_data(
        output_dir=DEFAULT_DATA_DIR,
        max_size_per_class=DEFAULT_MAX_SIZE_PER_CLASS,
        neutral_max_size=1000,
        lexicon_words_per_class=DEFAULT_LEXICON_WORDS_PER_CLASS,
        min_occurrences_per_target=DEFAULT_MIN_OCCURRENCES_PER_TARGET,
        seed=DEFAULT_SEED,
    )
    print(f"Wrote {train_path}")
    print(f"Wrote {neutral_path}")
