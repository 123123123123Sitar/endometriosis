"""Deduplicate per-chunk themes from a single-model run via embedding clustering.

Why this exists: Anthropic credits ran out before the map-reduce REDUCE step
finished for the Claude Haiku 4.5 / prompt A run. We have 75 chunks of
themes (~896 total) but no model-merged final theme list. This script does
the merge locally instead, using Gemini embeddings which still have quota.

Cluster-survival is itself a reliability signal: themes that show up in
many chunks ("supported by k chunks") are stronger than singletons. The
output JSON matches the schema run_models.py would have written, so the
rest of the pipeline (grounding, literature anchor, reliability_score,
build_report) doesn't need to know the merge happened locally.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from embeddings import embed as gemini_embed  # noqa: E402

CLUSTER_THRESHOLD = 0.88  # tight: only obviously-same-topic chunk themes merge
MIN_CHUNK_SUPPORT = 1     # keep all clusters; cluster_size becomes a signal


def union_find(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--map-file", required=True,
                   help="path to a {model}__{prompt}__{run}.map.jsonl file")
    p.add_argument("--threshold", type=float, default=CLUSTER_THRESHOLD)
    p.add_argument("--min-support", type=int, default=MIN_CHUNK_SUPPORT,
                   help="drop clusters supported by fewer than this many chunks")
    args = p.parse_args()

    map_path = Path(args.map_file)
    if not map_path.exists():
        print(f"missing: {map_path}", file=sys.stderr)
        return 2

    # Reuse the run_id, model_id, prompt_id parsed from the filename.
    base = map_path.stem.replace(".map", "")
    parts = base.split("__")
    if len(parts) != 3:
        print(f"unexpected filename: {map_path.name}", file=sys.stderr)
        return 2
    model_id, prompt_id, run_id = parts

    raw_themes = []
    chunk_assignment = []
    for line_no, line in enumerate(map_path.open("r", encoding="utf-8")):
        obj = json.loads(line)
        chunk_index = obj["chunk_index"]
        for t in obj.get("themes", []):
            raw_themes.append(t)
            chunk_assignment.append(chunk_index)
    if not raw_themes:
        print("no themes to dedupe", file=sys.stderr)
        return 1

    print(f"Loaded {len(raw_themes)} chunk-level themes from "
          f"{len(set(chunk_assignment))} chunks. Embedding...", flush=True)

    texts = [
        f"{(t.get('theme') or '').strip()}: {(t.get('description') or '').strip()}"
        for t in raw_themes
    ]
    emb = gemini_embed(texts, batch=100)

    # Pairwise similarity. With ~900 themes, the (n, n) matrix is ~3MB —
    # fine for a workstation. Build the edge list at the threshold.
    print("Computing pairwise similarity and clustering...", flush=True)
    sim = emb @ emb.T
    n = len(texts)
    edges: list[tuple[int, int]] = []
    for i in range(n):
        # vectorised: indices j > i with sim above threshold
        js = np.where(sim[i, i + 1:] >= args.threshold)[0]
        for j in js:
            edges.append((i, int(i + 1 + j)))
    components = union_find(n, edges)
    components.sort(key=lambda c: -len(c))

    # Build merged theme entries.
    merged: list[dict] = []
    for cid, members in enumerate(components):
        chunk_set = sorted({chunk_assignment[i] for i in members})
        if len(chunk_set) < args.min_support:
            continue
        # Pick the longest description as the canonical entry.
        best = max(members, key=lambda i: len(raw_themes[i].get("description") or ""))
        canonical = raw_themes[best]
        # Merge supporting quotes (deduplicated, capped).
        seen_q: set[str] = set()
        merged_quotes: list[str] = []
        for m in members:
            for q in raw_themes[m].get("supporting_quotes") or []:
                key = q.strip().lower()
                if key and key not in seen_q:
                    seen_q.add(key)
                    merged_quotes.append(q.strip())
                if len(merged_quotes) >= 5:
                    break
            if len(merged_quotes) >= 5:
                break
        # Merged severity: take majority among members.
        sev_counts: dict[str, int] = defaultdict(int)
        for m in members:
            s = raw_themes[m].get("severity_signal")
            if s in ("low", "medium", "high"):
                sev_counts[s] += 1
        severity = max(sev_counts.items(), key=lambda kv: kv[1])[0] if sev_counts else "medium"

        merged.append({
            "theme": canonical.get("theme", ""),
            "description": canonical.get("description", ""),
            "supporting_quotes": merged_quotes[:3],
            "frequency_estimate": (
                f"supported by {len(chunk_set)}/"
                f"{len(set(chunk_assignment))} chunks"
            ),
            "severity_signal": severity,
            "chunk_support": len(chunk_set),
            "cluster_member_count": len(members),
        })

    print(f"\nMerged {len(raw_themes)} -> {len(merged)} themes "
          f"(threshold={args.threshold}, min_support={args.min_support})", flush=True)

    # Top by chunk support.
    print("\nTop themes by chunk support:")
    for t in sorted(merged, key=lambda x: -x["chunk_support"])[:15]:
        print(f"  k={t['chunk_support']:>3}  {t['theme']}")

    # Write the dedup theme file in the same schema as run_models.py output.
    out_path = PROJECT_ROOT / "outputs" / "raw" / f"{model_id}__{prompt_id}__{run_id}.json"
    out_path.write_text(json.dumps({
        "model_id": model_id,
        "model": "claude-haiku-4-5-20251001",
        "prompt_id": prompt_id,
        "run_id": run_id,
        "temperature": 0.0,
        "merge_method": "local_embedding_cluster",
        "merge_threshold": args.threshold,
        "n_chunks": len(set(chunk_assignment)),
        "n_raw_chunk_themes": len(raw_themes),
        "themes": sorted(merged, key=lambda x: -x["chunk_support"]),
    }, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
