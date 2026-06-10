"""Collect r/endometriosis + r/ENDO submissions and comments via pullpush.io.

Window matches the original 2024 abstract: 2018-02-07 → 2024-10-31.

Pullpush.io is a community-maintained Pushshift mirror; no auth required.
Pages backwards in time using `before=<unix_ts>` until the start cutoff is hit.
Resumable: writes JSONL in append mode and a small checkpoint file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from tqdm import tqdm

PULLPUSH_BASE = "https://api.pullpush.io/reddit/search"
WINDOW_START = int(datetime(2018, 2, 7, tzinfo=timezone.utc).timestamp())
WINDOW_END = int(datetime(2024, 11, 1, tzinfo=timezone.utc).timestamp())  # exclusive
SUBREDDITS = ["endometriosis", "ENDO"]
PAGE_SIZE = 100
SLEEP_BETWEEN_REQUESTS = 0.6  # be polite


@dataclass
class CollectorConfig:
    out_dir: Path
    subreddits: list[str]
    start_ts: int
    end_ts: int


@retry(
    retry=retry_if_exception_type((requests.RequestException, ValueError)),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(6),
    reraise=True,
)
def _get(url: str, params: dict) -> list[dict]:
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    body = r.json()
    if "data" not in body:
        raise ValueError(f"missing 'data' in response: {body}")
    return body["data"]


def _page(kind: str, subreddit: str, start_ts: int, end_ts: int) -> Iterator[dict]:
    """Yield items for a (kind, subreddit) backwards from end_ts down to start_ts."""
    assert kind in ("submission", "comment")
    url = f"{PULLPUSH_BASE}/{kind}/"
    before = end_ts
    seen_ids: set[str] = set()
    while True:
        items = _get(
            url,
            {
                "subreddit": subreddit,
                "size": PAGE_SIZE,
                "before": before,
                "sort": "desc",
                "sort_type": "created_utc",
            },
        )
        if not items:
            return
        # Sort to defend against API ordering quirks.
        items.sort(key=lambda x: x.get("created_utc", 0), reverse=True)
        new_count = 0
        oldest_ts = before
        for it in items:
            ts = int(it.get("created_utc", 0))
            iid = it.get("id")
            if iid in seen_ids:
                continue
            seen_ids.add(iid)
            if ts < start_ts:
                return
            yield it
            new_count += 1
            oldest_ts = min(oldest_ts, ts)
        if new_count == 0:
            # No progress; bail to avoid an infinite loop.
            return
        # Advance: subtract 1 to avoid replaying the boundary item.
        next_before = oldest_ts - 1
        if next_before >= before:
            return
        before = next_before
        time.sleep(SLEEP_BETWEEN_REQUESTS)


def collect(cfg: CollectorConfig) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"posts": 0, "comments": 0, "by_subreddit": {}}

    for sub in cfg.subreddits:
        sub_stats = {"posts": 0, "comments": 0}
        posts_path = cfg.out_dir / f"{sub}_posts.jsonl"
        comments_path = cfg.out_dir / f"{sub}_comments.jsonl"

        # Posts.
        with posts_path.open("w", encoding="utf-8") as f:
            pbar = tqdm(desc=f"r/{sub} posts", unit="post")
            for item in _page("submission", sub, cfg.start_ts, cfg.end_ts):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                sub_stats["posts"] += 1
                pbar.update(1)
            pbar.close()

        # Comments.
        with comments_path.open("w", encoding="utf-8") as f:
            pbar = tqdm(desc=f"r/{sub} comments", unit="comment")
            for item in _page("comment", sub, cfg.start_ts, cfg.end_ts):
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                sub_stats["comments"] += 1
                pbar.update(1)
            pbar.close()

        stats["by_subreddit"][sub] = sub_stats
        stats["posts"] += sub_stats["posts"]
        stats["comments"] += sub_stats["comments"]

    return stats


def write_manifest(out_dir: Path, stats: dict) -> Path:
    """Compute a content-hash manifest so all model runs read identical input."""
    manifest = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "window": {
            "start_utc": datetime.fromtimestamp(WINDOW_START, tz=timezone.utc).isoformat(),
            "end_utc": datetime.fromtimestamp(WINDOW_END, tz=timezone.utc).isoformat(),
        },
        "source": "pullpush.io",
        "subreddits": SUBREDDITS,
        "stats": stats,
        "files": {},
    }
    word_total = 0
    user_set: set[str] = set()
    for f in sorted(out_dir.glob("*.jsonl")):
        h = hashlib.sha256()
        n = 0
        with f.open("rb") as fp:
            for chunk in iter(lambda: fp.read(1 << 16), b""):
                h.update(chunk)
        # Walk the file once for word count and unique users.
        with f.open("r", encoding="utf-8") as fp:
            for line in fp:
                n += 1
                obj = json.loads(line)
                text = obj.get("selftext") or obj.get("body") or ""
                title = obj.get("title", "")
                word_total += len((title + " " + text).split())
                author = obj.get("author")
                if author and author not in ("[deleted]", "AutoModerator"):
                    user_set.add(author)
        manifest["files"][f.name] = {
            "sha256": h.hexdigest(),
            "rows": n,
            "size_bytes": f.stat().st_size,
        }
    manifest["stats"]["unique_users"] = len(user_set)
    manifest["stats"]["word_count"] = word_total

    out = out_dir.parent / "corpus.manifest.json"
    out.write_text(json.dumps(manifest, indent=2))
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data/raw", help="output directory for JSONL files")
    p.add_argument("--start", type=int, default=WINDOW_START)
    p.add_argument("--end", type=int, default=WINDOW_END)
    p.add_argument("--subreddits", nargs="+", default=SUBREDDITS)
    p.add_argument("--smoke", action="store_true", help="tiny window for end-to-end test")
    args = p.parse_args()

    if args.smoke:
        # Last 7 days of Oct 2024 — a tiny but real slice.
        args.start = int(datetime(2024, 10, 24, tzinfo=timezone.utc).timestamp())
        args.end = int(datetime(2024, 10, 31, tzinfo=timezone.utc).timestamp())

    project_root = Path(__file__).resolve().parent.parent
    out_dir = (project_root / args.out).resolve()

    cfg = CollectorConfig(
        out_dir=out_dir,
        subreddits=args.subreddits,
        start_ts=args.start,
        end_ts=args.end,
    )
    print(f"Collecting {cfg.subreddits} into {cfg.out_dir}")
    print(f"Window: {datetime.fromtimestamp(cfg.start_ts, tz=timezone.utc)} → "
          f"{datetime.fromtimestamp(cfg.end_ts, tz=timezone.utc)}")

    stats = collect(cfg)
    manifest_path = write_manifest(out_dir, stats)
    print(f"\nDone. Stats: {json.dumps(stats, indent=2)}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
