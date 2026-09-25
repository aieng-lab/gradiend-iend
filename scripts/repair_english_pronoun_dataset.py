#!/usr/bin/env python3
"""Repair a published pronoun CSV with the smallest possible row change.

Keeps every safe published row and replaces only examples that are duplicated
or whose masked prompt occurs in more than one split.  Replacement examples
must be produced by the current generator; they are selected by class and then
assigned to the vacated split, so the published class/split cardinalities stay
unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY = ["masked", "label_class", "label"]


def rows_to_replace_minimally(df: pd.DataFrame) -> pd.Series:
    """Keep the first published occurrence, replace only duplicate copies.

    The published artifact has no prompt with different targets.  Therefore
    retaining one occurrence of each exact prompt/class/label both removes all
    duplicates and confines each prompt to a single existing split.
    """
    return df.duplicated(KEY, keep="first")


def validate(df: pd.DataFrame) -> None:
    required = {"masked", "split", "label_class", "label"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df[KEY].isna().any().any() or df["split"].isna().any():
        raise ValueError("Dataset has missing values in required columns")
    if df.duplicated(KEY, keep=False).any():
        raise ValueError("Dataset still has duplicate, leaking, or ambiguous masked prompts")


def repair(published: pd.DataFrame, replacement: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    bad = rows_to_replace_minimally(published)
    retained = published.loc[~bad].copy()
    deficits = published.loc[bad].groupby(["label_class", "split"], sort=False).size()

    # The replacement artifact has already passed generator-level uniqueness
    # checks.  Do not allow it to overlap any retained displayed prompt.
    pool = replacement.loc[~replacement["masked"].isin(set(retained["masked"]))].copy()
    selected = []
    for (label_class, split), count in deficits.items():
        choices = pool.loc[pool["label_class"].astype(str).eq(str(label_class))]
        if len(choices) < count:
            raise ValueError(
                f"Need {count} replacements for {label_class}/{split}, found {len(choices)}"
            )
        take = choices.iloc[:count].copy()
        take["split"] = split
        selected.append(take)
        pool = pool.drop(index=take.index)

    repaired = pd.concat([retained, *selected], ignore_index=True)
    if len(repaired) != len(published):
        raise AssertionError("Repair changed the dataset row count")
    before = published.groupby(["label_class", "split"], sort=False).size().sort_index()
    after = repaired.groupby(["label_class", "split"], sort=False).size().sort_index()
    if not before.equals(after):
        raise AssertionError("Repair changed the class/split counts")
    validate(repaired)
    return repaired, int(bad.sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--published-training", type=Path, required=True)
    parser.add_argument("--replacement-training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    published = pd.read_csv(args.published_training)
    replacement = pd.read_csv(args.replacement_training)
    validate(replacement)
    repaired, replaced = repair(published, replacement)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    repaired.to_csv(args.output, index=False)
    print(f"Retained {len(published) - replaced} published rows; replaced {replaced} rows.")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
