"""Convert the 2024 Box archive (.zst Pushshift dumps) into pipeline jsonl.

Reads:
    ~/Downloads/reddit_data_zst/Endo_submissions.zst
    ~/Downloads/reddit_data_zst/Endo_comments.zst

Writes (filtered to 2018-02-07 .. 2024-10-31, drops fully-empty/bot rows):
    data/raw_2024_box/endo_posts.jsonl
    data/raw_2024_box/endo_comments.jsonl
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import zstandard as zstd

WINDOW_START = 1517961600  # 2018-02-07 UTC
WINDOW_END = 1730332800    # 2024-10-31 UTC


def stream_zst(path: Path):
    with path.open("rb") as fh:
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        with dctx.stream_reader(fh) as reader:
            buf = io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
            for line in buf:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def keep_post(obj: dict) -> bool:
    ts = obj.get("created_utc")
    if not isinstance(ts, int):
        try:
            ts = int(ts)
        except (TypeError, ValueError):
            return False
    if ts < WINDOW_START or ts > WINDOW_END:
        return False
    title = obj.get("title") or ""
    selftext = obj.get("selftext") or ""
    if not title and not selftext:
        return False
    if selftext in ("[removed]", "[deleted]") and not title:
        return False
    return True


def keep_comment(obj: dict) -> bool:
    ts = obj.get("created_utc")
    if not isinstance(ts, int):
        try:
            ts = int(ts)
        except (TypeError, ValueError):
            return False
    if ts < WINDOW_START or ts > WINDOW_END:
        return False
    body = obj.get("body") or ""
    if not body or body in ("[removed]", "[deleted]"):
        return False
    return True


def main():
    src = Path.home() / "Downloads" / "reddit_data_zst"
    out = Path(__file__).resolve().parent.parent / "data" / "raw_2024_box"
    out.mkdir(parents=True, exist_ok=True)

    posts_in = src / "Endo_submissions.zst"
    comments_in = src / "Endo_comments.zst"
    posts_out = out / "endo_posts.jsonl"
    comments_out = out / "endo_comments.jsonl"

    n_post_in = n_post_out = 0
    with posts_out.open("w", encoding="utf-8") as fp:
        for obj in stream_zst(posts_in):
            n_post_in += 1
            if keep_post(obj):
                fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_post_out += 1
            if n_post_in % 5000 == 0:
                sys.stderr.write(f"posts: {n_post_in} read, {n_post_out} kept\n")

    n_com_in = n_com_out = 0
    with comments_out.open("w", encoding="utf-8") as fp:
        for obj in stream_zst(comments_in):
            n_com_in += 1
            if keep_comment(obj):
                fp.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_com_out += 1
            if n_com_in % 25000 == 0:
                sys.stderr.write(f"comments: {n_com_in} read, {n_com_out} kept\n")

    print(json.dumps({
        "posts_read": n_post_in,
        "posts_kept": n_post_out,
        "comments_read": n_com_in,
        "comments_kept": n_com_out,
        "out_dir": str(out),
    }, indent=2))


if __name__ == "__main__":
    main()
