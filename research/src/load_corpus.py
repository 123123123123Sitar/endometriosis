"""Load + normalize the collected corpus into in-memory records.

Output record schema (used everywhere downstream):
    {
        "id":         str,   # reddit fullname-less id, prefixed t3_ or t1_ for type
        "type":       "post" | "comment",
        "subreddit":  str,
        "author":     str | None,  # None for [deleted]
        "created_utc": int,
        "title":      str,         # "" for comments
        "text":       str,         # body for comments, selftext for posts
        "permalink":  str,
        "parent_id":  str | None,  # comments only
    }

Filters:
- drops items where author/text/title are all missing
- keeps [deleted]/[removed] text but tags it (the LLM can ignore at its discretion)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator


@dataclass
class Record:
    id: str
    type: str
    subreddit: str
    author: str | None
    created_utc: int
    title: str
    text: str
    permalink: str
    parent_id: str | None

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def display_text(self) -> str:
        """Concatenated text used by the LLM."""
        if self.type == "post":
            return f"[{self.subreddit} POST] {self.title}\n\n{self.text}".strip()
        return f"[{self.subreddit} COMMENT] {self.text}".strip()


def _normalize_post(raw: dict) -> Record | None:
    rid = raw.get("id")
    if not rid:
        return None
    title = raw.get("title") or ""
    text = raw.get("selftext") or ""
    if not title and not text:
        return None
    author = raw.get("author")
    if author in ("[deleted]", None):
        author = None
    return Record(
        id=f"t3_{rid}",
        type="post",
        subreddit=raw.get("subreddit", ""),
        author=author,
        created_utc=int(raw.get("created_utc", 0)),
        title=title,
        text=text,
        permalink=raw.get("permalink", ""),
        parent_id=None,
    )


def _normalize_comment(raw: dict) -> Record | None:
    rid = raw.get("id")
    if not rid:
        return None
    text = raw.get("body") or ""
    if not text:
        return None
    author = raw.get("author")
    if author in ("[deleted]", None):
        author = None
    return Record(
        id=f"t1_{rid}",
        type="comment",
        subreddit=raw.get("subreddit", ""),
        author=author,
        created_utc=int(raw.get("created_utc", 0)),
        title="",
        text=text,
        permalink=raw.get("permalink", ""),
        parent_id=raw.get("parent_id"),
    )


def iter_records(raw_dir: Path) -> Iterator[Record]:
    for f in sorted(raw_dir.glob("*_posts.jsonl")):
        with f.open("r", encoding="utf-8") as fp:
            for line in fp:
                obj = json.loads(line)
                rec = _normalize_post(obj)
                if rec is not None:
                    yield rec
    for f in sorted(raw_dir.glob("*_comments.jsonl")):
        with f.open("r", encoding="utf-8") as fp:
            for line in fp:
                obj = json.loads(line)
                rec = _normalize_comment(obj)
                if rec is not None:
                    yield rec


def load_all(raw_dir: Path) -> list[Record]:
    return list(iter_records(raw_dir))


def corpus_stats(records: list[Record]) -> dict:
    posts = [r for r in records if r.type == "post"]
    comments = [r for r in records if r.type == "comment"]
    users = {r.author for r in records if r.author}
    words = sum(len((r.title + " " + r.text).split()) for r in records)
    return {
        "posts": len(posts),
        "comments": len(comments),
        "unique_users": len(users),
        "word_count": words,
        "by_subreddit": {
            sub: {
                "posts": sum(1 for r in posts if r.subreddit == sub),
                "comments": sum(1 for r in comments if r.subreddit == sub),
            }
            for sub in sorted({r.subreddit for r in records if r.subreddit})
        },
    }


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"
    recs = load_all(raw_dir)
    print(json.dumps(corpus_stats(recs), indent=2))
