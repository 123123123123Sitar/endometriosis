"""Audit the mentor-provided Endo Pushshift .zst dumps.

This is the first data-oriented step for the summer project: verify what
is actually in Endo_submissions.zst and Endo_comments.zst, compare it to
the 2024 poster numbers, and write a compact handoff memo. The report
intentionally contains aggregate counts only, not raw Reddit text.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import zstandard as zstd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
POSTER_START = 1517961600  # 2018-02-07 UTC
POSTER_END = 1730419199    # 2024-10-31 23:59:59 UTC


def utc_date(ts: int | None) -> str | None:
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def stream_zst(path: Path) -> Iterator[dict]:
    with path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            text = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in text:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def word_count(*parts: str) -> int:
    return sum(len((p or "").split()) for p in parts)


def in_window(obj: dict) -> bool:
    try:
        ts = int(obj.get("created_utc"))
    except (TypeError, ValueError):
        return False
    return POSTER_START <= ts <= POSTER_END


def is_deleted_text(text: str) -> bool:
    return (text or "").strip() in {"", "[removed]", "[deleted]"}


def audit_posts(path: Path) -> dict:
    raw_rows = kept_rows = 0
    users: set[str] = set()
    subreddits: Counter[str] = Counter()
    years: Counter[str] = Counter()
    raw_min = raw_max = kept_min = kept_max = None
    words = 0
    deleted_or_empty = 0

    for obj in stream_zst(path):
        raw_rows += 1
        try:
            ts = int(obj.get("created_utc"))
        except (TypeError, ValueError):
            continue
        raw_min = ts if raw_min is None else min(raw_min, ts)
        raw_max = ts if raw_max is None else max(raw_max, ts)
        if not in_window(obj):
            continue

        title = obj.get("title") or ""
        body = obj.get("selftext") or ""
        if not title and not body:
            deleted_or_empty += 1
            continue
        if is_deleted_text(body) and not title:
            deleted_or_empty += 1
            continue

        kept_rows += 1
        kept_min = ts if kept_min is None else min(kept_min, ts)
        kept_max = ts if kept_max is None else max(kept_max, ts)
        years[str(datetime.fromtimestamp(ts, tz=timezone.utc).year)] += 1
        subreddits[obj.get("subreddit") or ""] += 1
        author = obj.get("author")
        if author and author != "[deleted]":
            users.add(author)
        words += word_count(title, body)

    return {
        "raw_rows": raw_rows,
        "kept_rows": kept_rows,
        "deleted_or_empty_in_window": deleted_or_empty,
        "unique_users": len(users),
        "word_count": words,
        "raw_min_date": utc_date(raw_min),
        "raw_max_date": utc_date(raw_max),
        "kept_min_date": utc_date(kept_min),
        "kept_max_date": utc_date(kept_max),
        "subreddits": dict(sorted(subreddits.items())),
        "by_year": dict(sorted(years.items())),
    }


def audit_comments(path: Path) -> dict:
    raw_rows = kept_rows = 0
    users: set[str] = set()
    subreddits: Counter[str] = Counter()
    years: Counter[str] = Counter()
    raw_min = raw_max = kept_min = kept_max = None
    words = 0
    deleted_or_empty = 0

    for obj in stream_zst(path):
        raw_rows += 1
        try:
            ts = int(obj.get("created_utc"))
        except (TypeError, ValueError):
            continue
        raw_min = ts if raw_min is None else min(raw_min, ts)
        raw_max = ts if raw_max is None else max(raw_max, ts)
        if not in_window(obj):
            continue

        body = obj.get("body") or ""
        if is_deleted_text(body):
            deleted_or_empty += 1
            continue

        kept_rows += 1
        kept_min = ts if kept_min is None else min(kept_min, ts)
        kept_max = ts if kept_max is None else max(kept_max, ts)
        years[str(datetime.fromtimestamp(ts, tz=timezone.utc).year)] += 1
        subreddits[obj.get("subreddit") or ""] += 1
        author = obj.get("author")
        if author and author != "[deleted]":
            users.add(author)
        words += word_count(body)

    return {
        "raw_rows": raw_rows,
        "kept_rows": kept_rows,
        "deleted_or_empty_in_window": deleted_or_empty,
        "unique_users": len(users),
        "word_count": words,
        "raw_min_date": utc_date(raw_min),
        "raw_max_date": utc_date(raw_max),
        "kept_min_date": utc_date(kept_min),
        "kept_max_date": utc_date(kept_max),
        "subreddits": dict(sorted(subreddits.items())),
        "by_year": dict(sorted(years.items())),
    }


def fmt(n: int) -> str:
    return f"{n:,}"


def artifact_inventory() -> dict:
    chunks_dir = PROJECT_ROOT / "outputs" / "chunks"
    return {
        "raw_posts_jsonl_exists": (PROJECT_ROOT / "data" / "raw" / "endo_posts.jsonl").exists(),
        "raw_comments_jsonl_exists": (PROJECT_ROOT / "data" / "raw" / "endo_comments.jsonl").exists(),
        "sample_jsonl_exists": (PROJECT_ROOT / "data" / "sample.jsonl").exists(),
        "chunk_text_files": len(list(chunks_dir.glob("chunk_*.txt"))) if chunks_dir.exists() else 0,
        "chunk_theme_files": len(list(chunks_dir.glob("chunk_*.themes.json"))) if chunks_dir.exists() else 0,
        "reduced_theme_file_exists": (PROJECT_ROOT / "outputs" / "raw" / "claude_haiku_4_5__A__primary.json").exists(),
        "chunk_theme_reduce_input_exists": (PROJECT_ROOT / "outputs" / "_chunk_themes_for_reduce.json").exists(),
    }


def write_markdown(out_path: Path, audit: dict) -> None:
    filtered = audit["filtered_window"]
    sample = audit.get("analysis_sample")
    artifacts = audit["artifact_inventory"]
    poster = {
        "posts": 1986,
        "comments": 79993,
        "words": 4321905,
        "users": 17049,
    }

    lines = [
        "# Endo ZST Data Audit",
        "",
        "Aggregate audit of `Endo_submissions.zst` and `Endo_comments.zst` for the endometriosis Reddit/LLM project.",
        "No raw Reddit text is included in this memo.",
        "",
        "## Files",
        "",
    ]
    for kind, meta in audit["files"].items():
        lines.append(f"- `{kind}`: `{meta['path']}` ({meta['size_bytes']:,} bytes, sha256 `{meta['sha256'][:16]}...`)")

    lines += [
        "",
        "## What the supplied archive contains",
        "",
        f"- Subreddits present after filtering: {', '.join(filtered['subreddits']) or 'none'}",
        f"- Raw submissions rows: {fmt(audit['posts']['raw_rows'])}",
        f"- Raw comments rows: {fmt(audit['comments']['raw_rows'])}",
        f"- Raw date range: {audit['raw_date_range']['min']} to {audit['raw_date_range']['max']}",
        f"- Poster-window retained date range: {filtered['min_date']} to {filtered['max_date']}",
        "",
        "## Poster-window aggregate counts",
        "",
        "| Source | Posts | Comments | Words | Unique users |",
        "|---|---:|---:|---:|---:|",
        f"| 2024 poster | {fmt(poster['posts'])} | {fmt(poster['comments'])} | {fmt(poster['words'])} | {fmt(poster['users'])} |",
        f"| Supplied r/Endo ZST archive, filtered | {fmt(filtered['posts'])} | {fmt(filtered['comments'])} | {fmt(filtered['word_count'])} | {fmt(filtered['unique_users'])} |",
    ]
    if sample:
        lines.append(f"| Current analysis sample | {fmt(sample['posts'])} | {fmt(sample['comments'])} | {fmt(sample['word_count'])} | {fmt(sample['unique_users'])} |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- The supplied `.zst` archive contains `r/Endo` only; it does not contain `r/endometriosis`.",
        "- The archive ends on 2022-12-31, so it cannot by itself cover the poster's 2023-01 to 2024-10 interval.",
        "- Within its available date range, the Pushshift-style archive is much larger than the poster's API-collected post count.",
        f"- The current project already has analysis-ready `data/raw/endo_posts.jsonl`, `data/raw/endo_comments.jsonl`, `data/sample.jsonl`, {artifacts['chunk_text_files']} chunk files, {artifacts['chunk_theme_files']} per-chunk theme JSONs, and one reduced theme file derived from this archive.",
        "",
        "## Recommended next step",
        "",
        "Use this audit as the data handoff for Tomiko: the immediate manuscript-facing task is to present the supplied archive as a partial replication dataset, explicitly noting that it is `r/Endo` only and ends in 2022. The already-generated Haiku/Sonnet theme outputs can serve as preliminary findings; a funded or quota-enabled follow-up should add `r/endometriosis` and 2023-2024 data before making claims about the full poster window.",
        "",
        "## Year counts in retained window",
        "",
        "| Year | Posts | Comments |",
        "|---|---:|---:|",
    ]
    years = sorted(set(audit["posts"]["by_year"]) | set(audit["comments"]["by_year"]))
    for y in years:
        lines.append(f"| {y} | {fmt(audit['posts']['by_year'].get(y, 0))} | {fmt(audit['comments']['by_year'].get(y, 0))} |")

    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--submissions", type=Path, default=Path.home() / "Downloads" / "Endo_submissions.zst")
    p.add_argument("--comments", type=Path, default=Path.home() / "Downloads" / "Endo_comments.zst")
    p.add_argument("--out-json", type=Path, default=PROJECT_ROOT / "outputs" / "endo_zst_audit.json")
    p.add_argument("--out-md", type=Path, default=PROJECT_ROOT / "outputs" / "endo_zst_audit.md")
    args = p.parse_args()

    if not args.submissions.exists():
        args.submissions = Path.home() / "Downloads" / "reddit_data_zst" / "Endo_submissions.zst"
    if not args.comments.exists():
        args.comments = Path.home() / "Downloads" / "reddit_data_zst" / "Endo_comments.zst"

    posts = audit_posts(args.submissions)
    comments = audit_comments(args.comments)

    all_users = set()
    # Re-stream just authors in the retained window to avoid retaining raw text.
    for path, text_key in [(args.submissions, "selftext"), (args.comments, "body")]:
        for obj in stream_zst(path):
            if not in_window(obj):
                continue
            text = obj.get(text_key) or ""
            title = obj.get("title") or ""
            if path == args.submissions and not title and not text:
                continue
            if path == args.comments and is_deleted_text(text):
                continue
            author = obj.get("author")
            if author and author != "[deleted]":
                all_users.add(author)

    kept_min_dates = [d for d in [posts["kept_min_date"], comments["kept_min_date"]] if d]
    kept_max_dates = [d for d in [posts["kept_max_date"], comments["kept_max_date"]] if d]
    raw_min_dates = [d for d in [posts["raw_min_date"], comments["raw_min_date"]] if d]
    raw_max_dates = [d for d in [posts["raw_max_date"], comments["raw_max_date"]] if d]
    subreddits = sorted(set(posts["subreddits"]) | set(comments["subreddits"]))

    audit = {
        "files": {
            "submissions": {
                "path": str(args.submissions),
                "size_bytes": args.submissions.stat().st_size,
                "sha256": sha256(args.submissions),
            },
            "comments": {
                "path": str(args.comments),
                "size_bytes": args.comments.stat().st_size,
                "sha256": sha256(args.comments),
            },
        },
        "poster_window": {
            "start": "2018-02-07",
            "end": "2024-10-31",
        },
        "raw_date_range": {
            "min": min(raw_min_dates) if raw_min_dates else None,
            "max": max(raw_max_dates) if raw_max_dates else None,
        },
        "posts": posts,
        "comments": comments,
        "filtered_window": {
            "posts": posts["kept_rows"],
            "comments": comments["kept_rows"],
            "word_count": posts["word_count"] + comments["word_count"],
            "unique_users": len(all_users),
            "min_date": min(kept_min_dates) if kept_min_dates else None,
            "max_date": max(kept_max_dates) if kept_max_dates else None,
            "subreddits": subreddits,
        },
        "artifact_inventory": artifact_inventory(),
    }

    sample_manifest = PROJECT_ROOT / "data" / "sample.manifest.json"
    if sample_manifest.exists():
        audit["analysis_sample"] = json.loads(sample_manifest.read_text()).get("stats")

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(args.out_md, audit)

    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(json.dumps(audit["filtered_window"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
