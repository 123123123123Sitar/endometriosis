"""Multi-LLM theme extraction over the sampled corpus.

Pipeline (per model × per prompt variant):
  1. MAP   — for each chunk of the corpus, ask the model for a theme JSON.
  2. REDUCE — feed all per-chunk theme lists back to the model and ask
              it to merge into a global theme list.

Outputs:
  outputs/raw/{model_id}__{prompt_id}__{run_id}.json   per-run final themes
  outputs/raw/{model_id}__{prompt_id}__{run_id}.map.jsonl   per-chunk themes
  outputs/run_index.json   index of all runs with timing/cost

The model abstraction is deliberately thin so we can drop in more providers
later without changing the orchestration code.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import yaml
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Add src/ to the import path for sibling utilities.
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from load_corpus import Record  # noqa: E402

WORDS_PER_TOKEN = 0.75  # rough; we err on the side of smaller chunks


# ---------------------------------------------------------------------------
# Provider adapters
# ---------------------------------------------------------------------------


@dataclass
class ModelSpec:
    id: str
    provider: str
    model: str
    display_name: str
    api_key_env: str
    max_tokens_out: int = 8192


class ProviderError(RuntimeError):
    pass


class AnthropicProvider:
    def __init__(self, spec: ModelSpec):
        import anthropic
        self.spec = spec
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise ProviderError(f"missing env {spec.api_key_env}")
        self.client = anthropic.Anthropic(api_key=api_key)

    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def complete(self, system: str, user: str, temperature: float) -> tuple[str, dict]:
        resp = self.client.messages.create(
            model=self.spec.model,
            max_tokens=self.spec.max_tokens_out,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        usage = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
        }
        return text, usage


class GeminiProvider:
    def __init__(self, spec: ModelSpec):
        import google.generativeai as genai
        self.spec = spec
        api_key = os.environ.get(spec.api_key_env)
        if not api_key:
            raise ProviderError(f"missing env {spec.api_key_env}")
        genai.configure(api_key=api_key)
        self._genai = genai
        self.model = genai.GenerativeModel(spec.model)

    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=60),
        stop=stop_after_attempt(5),
        reraise=True,
    )
    def complete(self, system: str, user: str, temperature: float) -> tuple[str, dict]:
        cfg = self._genai.GenerationConfig(
            temperature=temperature,
            max_output_tokens=self.spec.max_tokens_out,
            response_mime_type="application/json",
        )
        # Gemini API: pass system as system_instruction at model construction
        # OR prepend to user. We prepend for simplicity since system_instruction
        # is set per-request in newer SDKs but not consistently.
        prompt = f"{system}\n\n---\n\n{user}"
        resp = self.model.generate_content(prompt, generation_config=cfg)
        text = resp.text or ""
        usage = {
            "input_tokens": getattr(resp.usage_metadata, "prompt_token_count", 0),
            "output_tokens": getattr(resp.usage_metadata, "candidates_token_count", 0),
        }
        return text, usage


def make_provider(spec: ModelSpec):
    if spec.provider == "anthropic":
        return AnthropicProvider(spec)
    if spec.provider == "google":
        return GeminiProvider(spec)
    raise ProviderError(f"unsupported provider {spec.provider}")


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def chunk_records(
    records: list[Record], chunk_tokens: int, overlap_tokens: int
) -> list[list[Record]]:
    """Pack records into chunks ~chunk_tokens long, with small overlap."""
    chunks: list[list[Record]] = []
    cur: list[Record] = []
    cur_words = 0
    word_target = int(chunk_tokens * WORDS_PER_TOKEN)
    overlap_words = int(overlap_tokens * WORDS_PER_TOKEN)

    for r in records:
        n = len(r.display_text().split())
        if cur_words + n > word_target and cur:
            chunks.append(cur)
            # Build overlap from the tail of the previous chunk.
            tail: list[Record] = []
            tail_words = 0
            for prev in reversed(cur):
                tw = len(prev.display_text().split())
                if tail_words + tw > overlap_words:
                    break
                tail.insert(0, prev)
                tail_words += tw
            cur = tail
            cur_words = tail_words
        cur.append(r)
        cur_words += n
    if cur:
        chunks.append(cur)
    return chunks


def render_chunk(chunk: list[Record]) -> str:
    return "\n\n".join(r.display_text() for r in chunk)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


PROMPT_A = """You are an analyst summarizing patient experiences and perspectives in an
endometriosis-themed Reddit community. Read the posts and comments provided
and identify the main themes regarding endometriosis CARE AND TREATMENT.

For each theme, output:
  - theme: a short noun phrase
  - description: 2-3 sentence summary in your own words
  - supporting_quotes: up to 3 verbatim quotes (<= 25 words each) from the corpus
  - frequency_estimate: rough fraction of relevant posts mentioning the theme
  - severity_signal: low | medium | high (how distressed users sound)

Return strict JSON: {"themes": [...]}. Do not invent quotes; only use text
that appears verbatim in the input."""


PROMPT_B = """You are reviewing patient-written discussions in an online endometriosis
community. Without bias toward any particular treatment philosophy, identify
the recurring topics that come up around how endometriosis is managed,
treated, or experienced in clinical settings.

For each topic, return:
  - theme: short label
  - description: a faithful 2-3 sentence paraphrase
  - supporting_quotes: up to 3 short verbatim excerpts (<= 25 words each)
  - frequency_estimate: approximate share of relevant posts where it appears
  - severity_signal: low | medium | high based on the affective tone

Output strict JSON: {"themes": [...]}. Quote only text that is literally
present in the input."""


REDUCE_INSTRUCTION = """You will be given multiple per-chunk theme lists below, each produced by
analyzing a slice of the same corpus. Merge them into a single global
theme list:
  - de-duplicate themes that describe the same underlying phenomenon
  - prefer the clearer description; do NOT invent new themes
  - propagate supporting_quotes (deduplicated) up to a max of 3 per theme
  - frequency_estimate should reflect the merged evidence ("most chunks",
    "minority of chunks", etc.)
Output strict JSON: {"themes": [...]} in the same schema as the inputs."""


PROMPTS = {"A": PROMPT_A, "B": PROMPT_B}


# ---------------------------------------------------------------------------
# JSON parsing (defensive — models occasionally wrap in ```json ... ```)
# ---------------------------------------------------------------------------


def extract_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    def _wrap(parsed):
        if isinstance(parsed, list):
            return {"themes": parsed}
        if isinstance(parsed, dict):
            if "themes" in parsed:
                return parsed
            for v in parsed.values():
                if isinstance(v, list):
                    return {"themes": v}
        raise ValueError("no themes array found")

    try:
        return _wrap(json.loads(text))
    except json.JSONDecodeError:
        # Fall back to lenient parser for malformed-but-recoverable JSON.
        try:
            from json_repair import repair_json
            repaired = repair_json(text, return_objects=True)
            return _wrap(repaired)
        except Exception as e:
            raise ValueError(f"JSON parse + repair failed: {e}") from e


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


@dataclass
class RunResult:
    model_id: str
    prompt_id: str
    run_id: str
    temperature: float
    started_at: str
    finished_at: str
    n_chunks: int
    final_themes: list[dict]
    map_themes: list[list[dict]] = field(default_factory=list)
    usage: dict = field(default_factory=dict)


def run_one(
    spec: ModelSpec,
    records: list[Record],
    prompt_id: str,
    temperature: float,
    run_id: str,
    chunk_tokens: int,
    overlap_tokens: int,
    out_dir: Path,
) -> RunResult:
    provider = make_provider(spec)
    chunks = chunk_records(records, chunk_tokens, overlap_tokens)
    print(f"[{spec.id}/{prompt_id}/{run_id}] {len(chunks)} chunks", flush=True)

    started = datetime.now(timezone.utc).isoformat()
    map_themes: list[list[dict]] = []
    map_path = out_dir / f"{spec.id}__{prompt_id}__{run_id}.map.jsonl"
    total_usage = {"input_tokens": 0, "output_tokens": 0}

    system = PROMPTS[prompt_id]
    with map_path.open("w", encoding="utf-8") as mf:
        for i, chunk in enumerate(chunks, 1):
            rendered = render_chunk(chunk)
            t0 = time.time()
            try:
                text, usage = provider.complete(system, rendered, temperature)
                obj = extract_json(text)
                themes = obj.get("themes", [])
            except Exception as e:
                themes = []
                usage = {"input_tokens": 0, "output_tokens": 0, "error": str(e)}
            map_themes.append(themes)
            mf.write(json.dumps({
                "chunk_index": i,
                "n_records": len(chunk),
                "themes": themes,
                "usage": usage,
                "elapsed_s": round(time.time() - t0, 2),
            }) + "\n")
            mf.flush()
            total_usage["input_tokens"] += usage.get("input_tokens", 0)
            total_usage["output_tokens"] += usage.get("output_tokens", 0)
            if i % 5 == 0 or i == len(chunks):
                print(f"  chunk {i}/{len(chunks)} themes={len(themes)}", flush=True)

    # Reduce step.
    reduce_user = "PER-CHUNK THEME LISTS:\n\n" + json.dumps(map_themes, indent=2)
    try:
        text, usage = provider.complete(REDUCE_INSTRUCTION, reduce_user, temperature)
        final_themes = extract_json(text).get("themes", [])
        total_usage["input_tokens"] += usage.get("input_tokens", 0)
        total_usage["output_tokens"] += usage.get("output_tokens", 0)
    except Exception as e:
        print(f"  reduce step failed: {e}", flush=True)
        final_themes = [t for chunk_themes in map_themes for t in chunk_themes]

    finished = datetime.now(timezone.utc).isoformat()
    final_path = out_dir / f"{spec.id}__{prompt_id}__{run_id}.json"
    final_path.write_text(json.dumps({
        "model_id": spec.id,
        "model": spec.model,
        "prompt_id": prompt_id,
        "run_id": run_id,
        "temperature": temperature,
        "started_at": started,
        "finished_at": finished,
        "n_chunks": len(chunks),
        "themes": final_themes,
        "usage": total_usage,
    }, indent=2, ensure_ascii=False))

    return RunResult(
        model_id=spec.id,
        prompt_id=prompt_id,
        run_id=run_id,
        temperature=temperature,
        started_at=started,
        finished_at=finished,
        n_chunks=len(chunks),
        final_themes=final_themes,
        map_themes=map_themes,
        usage=total_usage,
    )


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


def load_sample() -> list[Record]:
    sample_path = PROJECT_ROOT / "data" / "sample.jsonl"
    records: list[Record] = []
    with sample_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            records.append(Record(**obj))
    return records


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", help="model ids; default = all in config")
    p.add_argument("--prompts", nargs="+", default=["A", "B"], choices=["A", "B"])
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--run-id", default="primary")
    p.add_argument("--chunk-tokens", type=int, default=100000,
                   help="big chunks reduce # of API calls; both Haiku 4.5 (200k) "
                        "and Gemini Flash (1M) handle this easily")
    p.add_argument("--overlap-tokens", type=int, default=2000)
    p.add_argument("--limit", type=int, default=None,
                   help="cap records (use for smoke test)")
    args = p.parse_args()

    cfg_path = PROJECT_ROOT / "config" / "models.yaml"
    out_dir = PROJECT_ROOT / "outputs" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = load_models(cfg_path)
    if args.models:
        specs = [s for s in specs if s.id in args.models]
        if not specs:
            print(f"no specs match {args.models}", file=sys.stderr)
            return 2

    records = load_sample()
    if args.limit:
        records = records[: args.limit]
    print(f"Loaded {len(records)} records for analysis", flush=True)

    results = []
    for spec in specs:
        for pid in args.prompts:
            r = run_one(
                spec=spec,
                records=records,
                prompt_id=pid,
                temperature=args.temperature,
                run_id=args.run_id,
                chunk_tokens=args.chunk_tokens,
                overlap_tokens=args.overlap_tokens,
                out_dir=out_dir,
            )
            results.append(r)
            print(
                f"[{spec.id}/{pid}/{args.run_id}] done. "
                f"{len(r.final_themes)} themes, "
                f"in={r.usage.get('input_tokens',0)} out={r.usage.get('output_tokens',0)}",
                flush=True,
            )

    # Append to JSONL so multiple processes can write concurrently without
    # racing on a read-modify-write of a JSON array. The index JSON is rebuilt
    # from this file on demand by build_index() / build_report.py.
    index_jsonl = PROJECT_ROOT / "outputs" / "run_index.jsonl"
    with index_jsonl.open("a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps({
                "model_id": r.model_id,
                "prompt_id": r.prompt_id,
                "run_id": r.run_id,
                "temperature": r.temperature,
                "started_at": r.started_at,
                "finished_at": r.finished_at,
                "n_chunks": r.n_chunks,
                "n_themes": len(r.final_themes),
                "usage": r.usage,
            }, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
