"""Quote + claim grounding verification.

For every theme produced by every model, check whether each supporting
quote can be located in the source corpus. We report:

  - exact_match:  the quote appears verbatim (case-insensitive, whitespace-
                  collapsed) in some record's text
  - bm25_top1_score:  best BM25 score against the corpus
  - embedding_top1_sim: best cosine similarity against the corpus

A quote is considered grounded if exact_match is True OR
embedding_top1_sim >= 0.80 (semantic-equivalence threshold tuned on a
small dev set; revisit before final manuscript).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from load_corpus import Record  # noqa: E402
from embeddings import embed as gemini_embed  # noqa: E402

EMBED_THRESHOLD = 0.87  # tuned for gemini-embedding-001 (high baseline similarity)
WHITESPACE_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip().lower()


def load_sample_records() -> list[Record]:
    path = PROJECT_ROOT / "data" / "sample.jsonl"
    out: list[Record] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out.append(Record(**obj))
    return out


def build_bm25(records: list[Record]) -> tuple[BM25Okapi, list[str]]:
    docs = [normalize(r.title + " " + r.text) for r in records]
    tokenized = [d.split() for d in docs]
    return BM25Okapi(tokenized), docs


def exact_match(quote: str, normalized_docs: list[str]) -> bool:
    q = normalize(quote)
    if not q:
        return False
    return any(q in d for d in normalized_docs)


def score_quote(
    quote: str,
    bm25: BM25Okapi,
    docs: list[str],
    quote_emb: np.ndarray,
    doc_emb: np.ndarray,
) -> dict:
    em = exact_match(quote, docs)
    qn = normalize(quote)
    bm25_scores = bm25.get_scores(qn.split()) if qn else np.array([0.0])
    bm25_top = float(np.max(bm25_scores)) if bm25_scores.size else 0.0
    sims = doc_emb @ quote_emb
    sim_top = float(np.max(sims))
    grounded = bool(em or sim_top >= EMBED_THRESHOLD)
    return {
        "exact_match": em,
        "bm25_top1": bm25_top,
        "embedding_top1": sim_top,
        "grounded": grounded,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", help="theme JSON files; default = all in outputs/raw")
    p.add_argument("--out", default="eval/grounding.json")
    p.add_argument("--no-embeddings", action="store_true",
                   help="skip embedding step (exact + bm25 only)")
    args = p.parse_args()

    raw_dir = PROJECT_ROOT / "outputs" / "raw"
    if args.inputs:
        files = [Path(x) for x in args.inputs]
    else:
        files = sorted(raw_dir.glob("*.json"))
        files = [f for f in files if not f.name.endswith(".map.jsonl")]

    records = load_sample_records()
    print(f"Indexing {len(records)} records for BM25...", flush=True)
    bm25, docs = build_bm25(records)

    if args.no_embeddings:
        doc_emb = None
    else:
        print("Embedding corpus (Gemini text-embedding-004)...", flush=True)
        doc_emb = gemini_embed([r.title + " " + r.text for r in records])

    results = {}
    for f in files:
        run = json.loads(f.read_text())
        themes = run.get("themes", [])
        all_quotes: list[tuple[int, int, str]] = []
        for ti, t in enumerate(themes):
            for qi, q in enumerate(t.get("supporting_quotes", []) or []):
                all_quotes.append((ti, qi, q))
        if not all_quotes:
            results[f.name] = {"themes": []}
            continue

        if doc_emb is not None:
            print(f"Embedding {len(all_quotes)} quotes from {f.name}...", flush=True)
            quote_embs = gemini_embed([q for _, _, q in all_quotes])
        else:
            quote_embs = np.zeros((len(all_quotes), 1), dtype=np.float32)

        per_theme: dict[int, list[dict]] = {}
        for (ti, qi, q), qe in zip(all_quotes, quote_embs):
            if doc_emb is not None:
                score = score_quote(q, bm25, docs, qe, doc_emb)
            else:
                score = {
                    "exact_match": exact_match(q, docs),
                    "bm25_top1": float(np.max(bm25.get_scores(normalize(q).split()))),
                    "embedding_top1": None,
                    "grounded": exact_match(q, docs),
                }
            per_theme.setdefault(ti, []).append({"quote_index": qi, **score})

        results[f.name] = {
            "n_themes": len(themes),
            "n_quotes": len(all_quotes),
            "per_theme": [
                {
                    "theme_index": ti,
                    "theme": themes[ti].get("theme", ""),
                    "quotes": per_theme.get(ti, []),
                    "grounding_rate": (
                        sum(1 for s in per_theme.get(ti, []) if s["grounded"])
                        / max(1, len(per_theme.get(ti, [])))
                    ),
                }
                for ti in range(len(themes))
            ],
            "overall_grounding_rate": (
                sum(1 for ts in per_theme.values() for s in ts if s["grounded"])
                / max(1, sum(len(ts) for ts in per_theme.values()))
            ),
        }

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\nWrote {out_path}")
    for name, r in results.items():
        print(f"  {name}: grounded {r['overall_grounding_rate']:.1%} of {r['n_quotes']} quotes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
