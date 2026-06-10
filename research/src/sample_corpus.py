"""Build an analysis-ready sample from the full collected corpus.

Two reasons we sample rather than feed the entire 100k+ comment archive
to every LLM run:

1. Cost / runtime — even with map-reduce, putting the full corpus through
   2 models × 2 prompts × multiple stability seeds is expensive.

2. Direct comparability with the 2024 abstract — the original analyzed
   1,986 posts and 79,993 comments. Holding sample size roughly constant
   isolates "old GPT-4 vs modern LLMs" as the variable.

Sampling strategy: stratified random sampling by year to preserve temporal
distribution. Posts capped at ~2,000, comments at ~80,000. Seed pinned for
reproducibility, then we record the exact id list in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from load_corpus import Record, load_all, corpus_stats

DEFAULT_POST_TARGET = 2000
DEFAULT_COMMENT_TARGET = 80_000
SEED = 20260502


def stratified_sample(records: list[Record], target: int, rng: random.Random) -> list[Record]:
    if len(records) <= target:
        return list(records)
    by_year: dict[int, list[Record]] = defaultdict(list)
    for r in records:
        y = datetime.fromtimestamp(r.created_utc, tz=timezone.utc).year
        by_year[y].append(r)
    years = sorted(by_year)
    total = len(records)
    sampled: list[Record] = []
    # Allocate target proportionally to each year's actual share, with floor 1.
    per_year_target = {y: max(1, round(target * len(by_year[y]) / total)) for y in years}
    # Adjust rounding drift.
    drift = target - sum(per_year_target.values())
    if drift != 0:
        per_year_target[years[-1]] = max(1, per_year_target[years[-1]] + drift)
    for y in years:
        pool = by_year[y]
        k = min(per_year_target[y], len(pool))
        sampled.extend(rng.sample(pool, k))
    rng.shuffle(sampled)
    return sampled[:target]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--posts", type=int, default=DEFAULT_POST_TARGET)
    p.add_argument("--comments", type=int, default=DEFAULT_COMMENT_TARGET)
    p.add_argument("--seed", type=int, default=SEED)
    args = p.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    sample_path = project_root / "data" / "sample.jsonl"
    manifest_path = project_root / "data" / "sample.manifest.json"

    rng = random.Random(args.seed)
    records = load_all(raw_dir)
    full_stats = corpus_stats(records)
    print(f"Full corpus: {json.dumps(full_stats, indent=2)}")

    posts = [r for r in records if r.type == "post"]
    comments = [r for r in records if r.type == "comment"]

    sampled_posts = stratified_sample(posts, args.posts, rng)
    sampled_comments = stratified_sample(comments, args.comments, rng)
    sample = sampled_posts + sampled_comments
    rng.shuffle(sample)

    with sample_path.open("w", encoding="utf-8") as f:
        for rec in sample:
            f.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")

    sample_stats = corpus_stats(sample)
    sample_hash = hashlib.sha256(sample_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps({
        "seed": args.seed,
        "targets": {"posts": args.posts, "comments": args.comments},
        "sampled_ids": [r.id for r in sample],
        "sample_sha256": sample_hash,
        "stats": sample_stats,
    }, indent=2))

    print(f"\nSampled corpus: {json.dumps(sample_stats, indent=2)}")
    print(f"Wrote {sample_path}")
    print(f"Wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
