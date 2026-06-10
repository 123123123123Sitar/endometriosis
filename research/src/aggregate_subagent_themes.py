"""Aggregate per-chunk subagent theme JSONs into the pipeline's expected
.map.jsonl format, plus a single bundled file the reduce subagent can
read in one go.

Reads:
    outputs/chunks/chunk_NNN.themes.json
    outputs/chunks/manifest.json

Writes:
    outputs/raw/claude_haiku_4_5__A__primary.map.jsonl
    outputs/_chunk_themes_for_reduce.json
"""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    chunks_dir = PROJECT_ROOT / "outputs" / "chunks"
    raw_dir = PROJECT_ROOT / "outputs" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((chunks_dir / "manifest.json").read_text())
    n_chunks = manifest["n_chunks"]

    map_path = raw_dir / "claude_haiku_4_5__A__primary.map.jsonl"
    bundle_path = PROJECT_ROOT / "outputs" / "_chunk_themes_for_reduce.json"

    bundle: list[list[dict]] = []
    total_themes = 0
    with map_path.open("w", encoding="utf-8") as mf:
        for i in range(1, n_chunks + 1):
            theme_file = chunks_dir / f"chunk_{i:03d}.themes.json"
            meta_file = chunks_dir / f"chunk_{i:03d}.meta.json"
            if not theme_file.exists():
                print(f"  MISSING themes for chunk {i}; recording empty list")
                themes: list[dict] = []
            else:
                obj = json.loads(theme_file.read_text())
                themes = obj.get("themes", [])
            meta = json.loads(meta_file.read_text())
            mf.write(json.dumps({
                "chunk_index": i,
                "n_records": meta["n_records"],
                "themes": themes,
                "usage": {"input_tokens": 0, "output_tokens": 0,
                          "note": "ran via Haiku subagent; per-chunk token usage logged separately"},
                "elapsed_s": None,
            }, ensure_ascii=False) + "\n")
            bundle.append(themes)
            total_themes += len(themes)

    bundle_path.write_text(json.dumps(bundle, ensure_ascii=False, indent=2))
    print(f"Wrote {map_path}")
    print(f"Wrote {bundle_path}")
    print(f"Total chunk-level themes: {total_themes} across {n_chunks} chunks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
