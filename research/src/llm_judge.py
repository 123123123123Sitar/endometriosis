"""LLM-as-judge with rubric scoring + rotating judge.

For every theme produced by a candidate model+prompt run, ask each judge
model to rate it on five dimensions:
  - faithfulness     (does the description match what the corpus says?)
  - comprehensiveness (does the theme cover the underlying phenomenon?)
  - clinical_relevance (would a clinician find this useful?)
  - specificity      (is it precise vs hand-wavy?)
  - non_redundancy   (is it distinct from other themes in the same run?)

Each is scored 1-5 with a one-sentence justification. We rotate the judge
across the available models so every theme gets >= 1 cross-model rating.

Self-judging is recorded but excluded from headline numbers.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
load_dotenv(PROJECT_ROOT / ".env")

from run_models import ModelSpec, make_provider  # noqa: E402

import json
import re


def _parse_rubric_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        from json_repair import repair_json
        return repair_json(text, return_objects=True)

JUDGE_SYSTEM = """You are an expert qualitative-research methodologist evaluating themes that
another AI system extracted from a Reddit corpus about endometriosis. You
will be given:
  1. A single candidate theme (with description + supporting quotes).
  2. The other themes from the same extraction run, for context.

Score the candidate theme on five dimensions, each 1-5:
  - faithfulness:        does the description plausibly reflect what
                         endometriosis-community discussions say?
  - comprehensiveness:   does the theme capture the breadth of the
                         underlying phenomenon (vs being one narrow case)?
  - clinical_relevance:  would a clinician find this useful for patient care?
  - specificity:         is the theme precise, or vague/hand-wavy?
  - non_redundancy:      is this distinct from the other themes provided?

Respond with strict JSON:
{
  "faithfulness":       {"score": int, "justification": "..."},
  "comprehensiveness":  {"score": int, "justification": "..."},
  "clinical_relevance": {"score": int, "justification": "..."},
  "specificity":        {"score": int, "justification": "..."},
  "non_redundancy":     {"score": int, "justification": "..."}
}"""


def load_models(cfg_path: Path) -> list[ModelSpec]:
    cfg = yaml.safe_load(cfg_path.read_text())
    specs = []
    for entry in cfg.get("modern_2026", []):
        specs.append(ModelSpec(
            id=entry["id"],
            provider=entry["provider"],
            model=entry["model"],
            display_name=entry["display_name"],
            api_key_env=entry["api_key_env"],
            max_tokens_out=entry.get("max_tokens_out", 8192),
        ))
    return specs


def render_user(candidate: dict, others: list[dict]) -> str:
    cand = json.dumps(candidate, ensure_ascii=False, indent=2)
    others_brief = json.dumps(
        [{"theme": t.get("theme", ""), "description": t.get("description", "")}
         for t in others],
        ensure_ascii=False, indent=2,
    )
    return f"CANDIDATE THEME:\n{cand}\n\nOTHER THEMES IN SAME RUN:\n{others_brief}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", help="theme JSON files; default = all in outputs/raw")
    p.add_argument("--out", default="eval/judgments.json")
    p.add_argument("--include-self", action="store_true",
                   help="also have each model judge its own themes")
    args = p.parse_args()

    raw_dir = PROJECT_ROOT / "outputs" / "raw"
    if args.inputs:
        files = [Path(x) for x in args.inputs]
    else:
        files = sorted(f for f in raw_dir.glob("*.json") if not f.name.endswith(".map.jsonl"))

    cfg_path = PROJECT_ROOT / "config" / "models.yaml"
    specs = load_models(cfg_path)
    judges = {s.id: make_provider(s) for s in specs}
    print(f"Judges: {list(judges)}")

    out: list[dict] = []
    for f in files:
        run = json.loads(f.read_text())
        themes = run.get("themes", [])
        candidate_model = run.get("model_id", "")
        for ti, t in enumerate(themes):
            others = [u for j, u in enumerate(themes) if j != ti]
            user = render_user(t, others)
            for judge_id, provider in judges.items():
                if not args.include_self and judge_id == candidate_model:
                    continue
                try:
                    text, usage = provider.complete(JUDGE_SYSTEM, user, temperature=0.0)
                    obj = _parse_rubric_json(text)
                except Exception as e:
                    obj = {"error": str(e)}
                    usage = {}
                out.append({
                    "source_file": f.name,
                    "candidate_model": candidate_model,
                    "prompt_id": run.get("prompt_id", ""),
                    "theme_index": ti,
                    "theme": t.get("theme", ""),
                    "judge_model": judge_id,
                    "scores": obj,
                    "usage": usage,
                })
                print(f"  {candidate_model}/{run.get('prompt_id','')} theme {ti} judged by {judge_id}", flush=True)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    # Aggregate.
    dims = ["faithfulness", "comprehensiveness", "clinical_relevance",
            "specificity", "non_redundancy"]
    by_model: dict[str, dict[str, list[float]]] = {}
    for j in out:
        cm = j["candidate_model"]
        scores = j.get("scores", {})
        if "error" in scores:
            continue
        bucket = by_model.setdefault(cm, {d: [] for d in dims})
        for d in dims:
            v = scores.get(d, {})
            if isinstance(v, dict) and isinstance(v.get("score"), (int, float)):
                bucket[d].append(float(v["score"]))

    print("\nMean rubric scores by candidate model:")
    for m, b in by_model.items():
        means = {d: round(sum(v) / max(1, len(v)), 2) for d, v in b.items()}
        print(f"  {m}: {means}")
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
