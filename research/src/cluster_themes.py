"""Cluster themes across models + prompts to measure cross-LLM agreement.

Approach:
  1. Embed every (model, prompt, theme) tuple's "theme: description" text
     using Gemini text-embedding-004.
  2. Build a similarity graph; threshold edges at sim >= 0.80.
  3. Take connected components — each component is a "global theme cluster".
  4. Report:
     - n_clusters
     - per-cluster membership (which model/prompt/theme produced it)
     - pairwise overlap matrix (Jaccard) between models
     - prompt-sensitivity (themes that survive prompt A AND B per model)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from embeddings import embed as gemini_embed  # noqa: E402

CLUSTER_THRESHOLD = 0.87  # tuned for gemini-embedding-001 (high baseline similarity)


def connected_components(n: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return list(groups.values())


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", help="theme JSON files; default = all in outputs/raw")
    p.add_argument("--threshold", type=float, default=CLUSTER_THRESHOLD)
    p.add_argument("--out", default="eval/clusters.json")
    args = p.parse_args()

    raw_dir = PROJECT_ROOT / "outputs" / "raw"
    if args.inputs:
        files = [Path(x) for x in args.inputs]
    else:
        files = sorted(f for f in raw_dir.glob("*.json") if not f.name.endswith(".map.jsonl"))
    if not files:
        print("no input files found", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for f in files:
        run = json.loads(f.read_text())
        for ti, t in enumerate(run.get("themes", [])):
            text = (t.get("theme", "") + ": " + t.get("description", "")).strip()
            if not text:
                continue
            rows.append({
                "source": f.name,
                "model_id": run.get("model_id", ""),
                "prompt_id": run.get("prompt_id", ""),
                "run_id": run.get("run_id", ""),
                "theme_index": ti,
                "theme": t.get("theme", ""),
                "description": t.get("description", ""),
                "text": text,
            })

    if not rows:
        print("no themes extracted", file=sys.stderr)
        return 1

    print(f"Embedding {len(rows)} themes...")
    emb = gemini_embed([r["text"] for r in rows])
    sim = emb @ emb.T

    edges = []
    n = len(rows)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= args.threshold:
                edges.append((i, j))

    components = connected_components(n, edges)
    components.sort(key=lambda c: -len(c))

    # Cluster summary.
    clusters = []
    for cid, members in enumerate(components):
        sources = sorted({rows[i]["source"] for i in members})
        models = sorted({rows[i]["model_id"] for i in members})
        prompts = sorted({rows[i]["prompt_id"] for i in members})
        clusters.append({
            "cluster_id": cid,
            "size": len(members),
            "n_models": len(models),
            "n_prompts": len(prompts),
            "models": models,
            "prompts": prompts,
            "members": [
                {
                    "source": rows[i]["source"],
                    "model_id": rows[i]["model_id"],
                    "prompt_id": rows[i]["prompt_id"],
                    "theme": rows[i]["theme"],
                    "description": rows[i]["description"],
                }
                for i in members
            ],
        })

    # Pairwise model agreement (Jaccard over the set of cluster_ids each model touches).
    models = sorted({r["model_id"] for r in rows})
    model_cluster_sets: dict[str, set[int]] = {m: set() for m in models}
    for cid, c in enumerate(clusters):
        for m in c["members"]:
            model_cluster_sets[m["model_id"]].add(cid)
    pairwise = []
    for i, mi in enumerate(models):
        for mj in models[i:]:
            j = jaccard(model_cluster_sets[mi], model_cluster_sets[mj])
            pairwise.append({"a": mi, "b": mj, "jaccard": j})

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "threshold": args.threshold,
        "n_themes_in": len(rows),
        "n_clusters": len(clusters),
        "clusters": clusters,
        "pairwise_jaccard": pairwise,
    }, indent=2, ensure_ascii=False))

    print(f"\nWrote {out_path}")
    print(f"  {len(rows)} themes -> {len(clusters)} clusters at sim >= {args.threshold}")
    multi = sum(1 for c in clusters if c["n_models"] >= 2)
    print(f"  {multi} clusters span >= 2 models (cross-model agreement)")
    for pw in pairwise:
        if pw["a"] != pw["b"]:
            print(f"  {pw['a']} vs {pw['b']}: Jaccard = {pw['jaccard']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
