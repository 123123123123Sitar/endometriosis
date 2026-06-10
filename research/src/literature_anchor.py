"""Match LLM-extracted themes to the Young 2015 systematic-review themes.

A theme is "literature-supported" if its embedding cosine similarity to any
Young 2015 theme is >= MATCH_THRESHOLD. The matched anchor theme is recorded.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from embeddings import embed as gemini_embed  # noqa: E402

MATCH_THRESHOLD = 0.05  # tuned for TF-IDF; corresponds to ~p70 of theme-anchor sim,
                        # catches obvious lexical overlap, misses paraphrases.


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--threshold", type=float, default=MATCH_THRESHOLD)
    p.add_argument("--out", default="eval/literature_anchor.json")
    args = p.parse_args()

    anchor_path = PROJECT_ROOT / "eval" / "young_2015_themes.json"
    anchor = json.loads(anchor_path.read_text())
    anchor_texts = [
        f"{t['label']}: {t['description']}" for t in anchor["themes"]
    ]
    print(f"Anchor: {len(anchor_texts)} Young 2015 themes")

    raw_dir = PROJECT_ROOT / "outputs" / "raw"
    rows: list[dict] = []
    for f in sorted(raw_dir.glob("*.json")):
        if f.name.endswith(".map.jsonl"):
            continue
        run = json.loads(f.read_text())
        for ti, t in enumerate(run.get("themes", [])):
            text = f"{t.get('theme','')}: {t.get('description','')}"
            rows.append({
                "source_file": f.name,
                "model_id": run.get("model_id", ""),
                "prompt_id": run.get("prompt_id", ""),
                "theme_index": ti,
                "text": text,
            })

    if not rows:
        print("no themes to score", file=sys.stderr)
        return 0

    # Single embed call so the TF-IDF vocabulary is fit on the combined
    # anchor+theme corpus. Two separate calls would fit on anchors only and
    # truncate theme vocabulary, inflating similarity artificially.
    print(f"Embedding {len(anchor_texts) + len(rows)} texts...")
    combined = gemini_embed(anchor_texts + [r["text"] for r in rows])
    anchor_emb = combined[:len(anchor_texts)]
    theme_emb = combined[len(anchor_texts):]

    sim = theme_emb @ anchor_emb.T  # (n_themes, n_anchors)

    out: list[dict] = []
    for i, r in enumerate(rows):
        best_idx = int(np.argmax(sim[i]))
        best_score = float(sim[i, best_idx])
        out.append({
            "source_file": r["source_file"],
            "model_id": r["model_id"],
            "prompt_id": r["prompt_id"],
            "theme_index": r["theme_index"],
            "best_anchor_id": anchor["themes"][best_idx]["id"],
            "best_anchor_label": anchor["themes"][best_idx]["label"],
            "score": best_score,
            "matched": bool(best_score >= args.threshold),
        })

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    matched = sum(1 for r in out if r["matched"])
    print(f"\nWrote {out_path}")
    print(f"  {matched}/{len(out)} themes matched a Young 2015 theme at sim >= {args.threshold}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
