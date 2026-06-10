"""Combine all reliability signals into a per-theme score in [0, 1].

Inputs (each optional but recommended):
  - eval/clusters.json     (cross-LLM agreement)
  - eval/grounding.json    (faithfulness)
  - eval/judgments.json    (LLM-as-judge rubric)
  - eval/literature_anchor.json (Young 2015 alignment)
  - eval/stability.json    (temperature/prompt stability — produced separately)

Each signal is normalized to [0, 1] and the composite is the unweighted mean
of the available signals. Manuscript reports both the composite and the
component scores so reviewers can see the breakdown.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def safe_load(p: Path):
    if not p.exists():
        return None
    return json.loads(p.read_text())


def cluster_lookup(clusters: dict | None) -> dict:
    """Map (source_file, theme_index) -> cluster size."""
    if not clusters:
        return {}
    out = {}
    for c in clusters.get("clusters", []):
        for m in c["members"]:
            out[(m["source"], )] = c  # placeholder if theme_index isn't tracked
    # The cluster file stores theme indices via member ordering — re-derive.
    out = {}
    for c in clusters.get("clusters", []):
        for m in c["members"]:
            # We need theme_index; cluster_themes.py stored "theme" only, but
            # ordering is preserved per source file. Recover by counting.
            out.setdefault(m["source"], []).append({
                "theme": m["theme"],
                "cluster_size": c["size"],
                "n_models": c["n_models"],
            })
    return out


def grounding_lookup(grounding: dict | None) -> dict:
    if not grounding:
        return {}
    out = {}
    for fname, payload in grounding.items():
        per_theme = payload.get("per_theme", [])
        out[fname] = {pt["theme_index"]: pt for pt in per_theme}
    return out


def judge_lookup(judgments: list | None) -> dict:
    if not judgments:
        return {}
    out: dict = {}
    for j in judgments:
        key = (j["source_file"], j["theme_index"])
        out.setdefault(key, []).append(j)
    return out


def aggregate_judge(records: list[dict]) -> dict:
    dims = ["faithfulness", "comprehensiveness", "clinical_relevance",
            "specificity", "non_redundancy"]
    bucket = {d: [] for d in dims}
    for r in records:
        s = r.get("scores", {})
        if "error" in s:
            continue
        for d in dims:
            v = s.get(d, {})
            if isinstance(v, dict) and isinstance(v.get("score"), (int, float)):
                bucket[d].append(float(v["score"]))
    means = {d: (sum(v) / len(v) if v else None) for d, v in bucket.items()}
    avail = [v for v in means.values() if v is not None]
    composite = sum(avail) / len(avail) if avail else None
    return {"means": means, "composite_1_5": composite}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="eval/reliability.json")
    args = p.parse_args()

    clusters = safe_load(PROJECT_ROOT / "eval" / "clusters.json")
    grounding = safe_load(PROJECT_ROOT / "eval" / "grounding.json")
    judgments = safe_load(PROJECT_ROOT / "eval" / "judgments.json")
    anchor = safe_load(PROJECT_ROOT / "eval" / "literature_anchor.json")

    cluster_by_src = {}
    if clusters:
        for c in clusters.get("clusters", []):
            for m in c["members"]:
                cluster_by_src.setdefault(m["source"], []).append({
                    "theme": m["theme"],
                    "cluster_size": c["size"],
                    "n_models": c["n_models"],
                })

    grounding_by_src = grounding_lookup(grounding)
    judge_by_key = judge_lookup(judgments)
    anchor_by_key = {(a["source_file"], a["theme_index"]): a
                     for a in (anchor or [])}

    raw_dir = PROJECT_ROOT / "outputs" / "raw"
    rows: list[dict] = []
    for f in sorted(raw_dir.glob("*.json")):
        if f.name.endswith(".map.jsonl"):
            continue
        run = json.loads(f.read_text())
        for ti, t in enumerate(run.get("themes", [])):
            cluster_info = None
            if f.name in cluster_by_src and ti < len(cluster_by_src[f.name]):
                cluster_info = cluster_by_src[f.name][ti]

            grnd = (grounding_by_src.get(f.name, {}) or {}).get(ti)
            grounding_rate = grnd.get("grounding_rate") if grnd else None

            judges = judge_by_key.get((f.name, ti), [])
            judge_summary = aggregate_judge(judges) if judges else None

            anchored = anchor_by_key.get((f.name, ti))

            # Normalize signals to [0, 1].
            signals = {}
            if cluster_info:
                # 0 if alone, 1 if all 2 models agree.
                signals["cross_model_agreement"] = (
                    (cluster_info["n_models"] - 1) / max(1, 1)
                )
            if grounding_rate is not None:
                signals["grounding"] = float(grounding_rate)
            if judge_summary and judge_summary["composite_1_5"] is not None:
                signals["judge"] = (judge_summary["composite_1_5"] - 1) / 4.0
            if anchored:
                signals["literature_anchor"] = 1.0 if anchored.get("matched") else 0.0

            composite = (sum(signals.values()) / len(signals)) if signals else None
            rows.append({
                "source_file": f.name,
                "model_id": run.get("model_id", ""),
                "prompt_id": run.get("prompt_id", ""),
                "theme_index": ti,
                "theme": t.get("theme", ""),
                "signals": signals,
                "composite_reliability": composite,
                "cluster_info": cluster_info,
                "judge_summary": judge_summary,
                "literature_match": anchored,
            })

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"themes": rows}, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path} with {len(rows)} themes")
    if rows:
        scored = [r["composite_reliability"] for r in rows if r["composite_reliability"] is not None]
        if scored:
            print(f"  mean composite reliability: {sum(scored)/len(scored):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
