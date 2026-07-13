#!/usr/bin/env python3
"""Fast stale-cache migration without importing torch or the trainer stack.

Use this instead of ``--migrate-stale-cache`` on ``gender_de_pre_prune_topk_ablation.py``
when you only need to clean up bad rows/checkpoints (seconds, not ~1 min import).

Example:
  python scripts/migrate_stale_preprune_cache.py --output-dir runs/gender_de_pre_topk_ablation
  python scripts/migrate_stale_preprune_cache.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from typing import Any, Dict, List, Optional, Sequence

STALE_PRUNED_INPUT_DIM_THRESHOLD = 25_000_000


def _load_raw_rows(path: str) -> List[dict]:
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload if isinstance(payload, list) else []


def _discover_result_paths(output_dir: str) -> List[str]:
    pair_dir = os.path.join(output_dir, "pair_results")
    if os.path.isdir(pair_dir):
        paths = sorted(
            os.path.join(pair_dir, name)
            for name in os.listdir(pair_dir)
            if name.endswith(".json")
        )
        if paths:
            return paths
    combined = os.path.join(output_dir, "pre_topk_grid_results.json")
    return [combined] if os.path.isfile(combined) else []


def _checkpoint_path(output_dir: str, run_id: str) -> str:
    return os.path.join(output_dir.rstrip("/\\"), run_id.strip("/\\"), "model")


def _load_saved_input_dim(model_dir: str) -> Optional[int]:
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return None
    try:
        with open(cfg_path, encoding="utf-8") as handle:
            cfg = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    arch = cfg.get("architecture") if isinstance(cfg, dict) else None
    if not isinstance(arch, dict):
        return None
    value = arch.get("input_dim")
    return int(value) if isinstance(value, int) else None


def _is_definitely_stale_row(row: dict) -> bool:
    pre_topk = row.get("pre_topk")
    if pre_topk is None or math.isclose(float(pre_topk), 1.0):
        return False
    kept_dim = row.get("kept_dim")
    return kept_dim is not None and int(kept_dim) > STALE_PRUNED_INPUT_DIM_THRESHOLD


def _is_stale_checkpoint(model_dir: str, *, pre_topk: float) -> bool:
    if math.isclose(pre_topk, 1.0):
        return False
    saved_dim = _load_saved_input_dim(model_dir)
    if saved_dim is None:
        return False
    return saved_dim > STALE_PRUNED_INPUT_DIM_THRESHOLD


def _invalidate_checkpoint_tree(checkpoint_path: str) -> None:
    run_dir = os.path.dirname(os.path.normpath(checkpoint_path))
    for path in (checkpoint_path, run_dir):
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)


def migrate(
    *,
    output_dir: str,
    result_paths: Sequence[str],
    dry_run: bool = False,
) -> int:
    invalidated = 0
    for results_path in result_paths:
        rows = _load_raw_rows(results_path)
        kept_rows: List[dict] = []
        changed = False
        for row in rows:
            stale_row = _is_definitely_stale_row(row)
            checkpoint_path = _checkpoint_path(output_dir, str(row.get("run_id", "")))
            pre_topk = float(row.get("pre_topk") or 1.0)
            stale_ckpt = os.path.isdir(checkpoint_path) and _is_stale_checkpoint(
                checkpoint_path,
                pre_topk=pre_topk,
            )
            if not stale_row and not stale_ckpt:
                kept_rows.append(row)
                continue

            if not dry_run:
                if os.path.isdir(checkpoint_path):
                    _invalidate_checkpoint_tree(checkpoint_path)
                topk_file = row.get("topk_indices_file")
                if isinstance(topk_file, str) and os.path.isfile(topk_file):
                    try:
                        os.remove(topk_file)
                    except OSError:
                        pass
            print(
                f"{'[dry-run] ' if dry_run else ''}stale: {row.get('pair')} {row.get('run_id')} "
                f"kept_dim={row.get('kept_dim')}"
            )
            invalidated += 1
            changed = True
        if changed and not dry_run:
            with open(results_path, "w", encoding="utf-8") as handle:
                json.dump(kept_rows, handle, indent=2)
    return invalidated


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=os.path.join("runs", "gender_de_pre_topk_ablation"),
    )
    parser.add_argument("--results-path", default=None, help="Single JSON file; default: all pair_results/*.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    if args.results_path:
        paths = [args.results_path]
    else:
        paths = _discover_result_paths(args.output_dir)
    if not paths:
        print("No result JSON files found.", file=sys.stderr)
        return 1

    n = migrate(output_dir=args.output_dir, result_paths=paths, dry_run=args.dry_run)
    print(f"{'Would invalidate' if args.dry_run else 'Invalidated'} {n} stale cell(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
