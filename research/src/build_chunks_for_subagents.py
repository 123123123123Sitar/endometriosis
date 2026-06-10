"""Pre-chunk the sample so each chunk can be handed to a Haiku subagent
as a flat text file. Subagents then extract themes from their assigned
chunk and write a JSON output back to disk; an aggregator rolls those
into the existing .map.jsonl format the rest of the pipeline expects.

Output layout:
    outputs/chunks/chunk_001.txt
    outputs/chunks/chunk_001.meta.json   # n_records, ids, word count
    ...
    outputs/chunks/manifest.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from load_corpus import Record  # noqa: E402
from run_models import chunk_records, render_chunk  # noqa: E402

CHUNK_TOKENS = 100000
OVERLAP_TOKENS = 2000


def load_sample() -> list[Record]:
    p = PROJECT_ROOT / "data" / "sample.jsonl"
    out: list[Record] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.append(Record(**obj))
    return out


def main() -> int:
    out_dir = PROJECT_ROOT / "outputs" / "chunks"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("chunk_*.txt"):
        old.unlink()
    for old in out_dir.glob("chunk_*.meta.json"):
        old.unlink()
    for old in out_dir.glob("chunk_*.themes.json"):
        old.unlink()

    records = load_sample()
    print(f"Loaded {len(records)} records", flush=True)
    chunks = chunk_records(records, CHUNK_TOKENS, OVERLAP_TOKENS)
    print(f"Built {len(chunks)} chunks @ {CHUNK_TOKENS} target tokens", flush=True)

    manifest = []
    for i, chunk in enumerate(chunks, 1):
        text = render_chunk(chunk)
        chunk_path = out_dir / f"chunk_{i:03d}.txt"
        chunk_path.write_text(text, encoding="utf-8")
        meta = {
            "chunk_index": i,
            "n_records": len(chunk),
            "word_count": sum(len(r.display_text().split()) for r in chunk),
            "ids": [r.id for r in chunk],
            "chunk_path": str(chunk_path.relative_to(PROJECT_ROOT)),
            "themes_path": f"outputs/chunks/chunk_{i:03d}.themes.json",
        }
        (out_dir / f"chunk_{i:03d}.meta.json").write_text(json.dumps(meta, indent=2))
        manifest.append(meta)

    (out_dir / "manifest.json").write_text(json.dumps({
        "n_chunks": len(chunks),
        "chunk_tokens": CHUNK_TOKENS,
        "overlap_tokens": OVERLAP_TOKENS,
        "chunks": manifest,
    }, indent=2))
    print(f"Wrote {len(chunks)} chunk files to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
