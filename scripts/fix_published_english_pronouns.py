#!/usr/bin/env python3
"""Minimal backward-compatible repair for a published pronoun training CSV.

This deliberately does not call the example generator or generate neutral
data.  It retains all safe published rows and asks the general creator for a
small, per-class candidate pool only large enough to replace unsafe rows.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gradiend.examples.create_english_pronoun_data import build_english_pronoun_data_creator
from repair_english_pronoun_dataset import repair, rows_to_replace_minimally, validate


PUBLISHED = Path("data/english_pronouns/training_published_before_repair.csv")
OUTPUT = Path("build/english_pronouns_repaired/training.csv")
# Exact repair demand is 801 rows.  These caps total 880, leaving a small
# reserve for candidates colliding with retained published prompts.
CANDIDATE_CAPS = {"1SG": 550, "1PL": 75, "2SGPL": 150, "3SG": 5, "3PL": 100}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deduplicate-only",
        action="store_true",
        help="Keep one published occurrence per prompt/target and do not generate replacements.",
    )
    args = parser.parse_args()
    published = pd.read_csv(PUBLISHED)
    replace = rows_to_replace_minimally(published)
    deficits = published.loc[replace].groupby(
        ["label_class", "split"], sort=False
    ).size()
    print(f"Published rows retained before refill: {len(published) - int(replace.sum())}")
    print(f"Rows requiring replacement: {int(deficits.sum())}")

    if args.deduplicate_only:
        repaired = published.loc[~replace].copy()
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        repaired.to_csv(OUTPUT, index=False)
        validate(repaired)
        print(f"Removed {int(replace.sum())} duplicate copies; wrote {OUTPUT}")
        return 0

    base = build_english_pronoun_data_creator(output_dir="build/unused", use_cache=False)
    base.output_dir = None  # the repair script owns all output paths
    configs = {str(config.id): config for config in base.feature_targets}
    frames = []
    for class_id, cap in CANDIDATE_CAPS.items():
        base.feature_targets = [configs[class_id]]
        candidate = base.generate_training_data(
            max_size_per_class=cap,
            format="minimal",
            balance=False,
            deduplicate=True,
            drop_ambiguous_masked=True,
            min_rows_per_class_for_split=0,
        )
        candidate["label_class"] = class_id
        candidate["feature_class_id"] = class_id
        frames.append(candidate)
        print(f"{class_id}: generated {len(candidate)}/{cap} candidates")

    candidates = pd.concat(frames, ignore_index=True)
    repaired, changed = repair(published, candidates)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(OUTPUT, index=False)
    validate(repaired)
    print(f"Retained {len(published) - changed}; replaced {changed}; wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
