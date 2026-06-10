# Reproducing the analysis

This document walks through every step needed to regenerate the manuscript
results from a clean checkout. The pipeline is intentionally a chain of
small command-line steps so that each intermediate artifact can be inspected
or re-run independently.

There are two paths through the pipeline:

1. A **smoke run on synthetic data** that exercises the full pipeline end to
   end without touching pullpush.io or any LLM-based collection step. This
   should complete in a few minutes and is the fastest way to confirm that
   your environment is wired up correctly.
2. A **full run on the real corpus**, which collects from pullpush.io,
   subsamples, and runs every model and reliability metric.

Both paths share the same downstream commands; only the input file
differs.

## 1. Prerequisites

- **Python 3.13.** The code uses `from __future__ import annotations`
  pervasively but otherwise targets the standard library plus the packages
  pinned in `requirements.txt`. Earlier 3.x versions may work but are not
  tested.
- **API keys** exported in your shell environment (or placed in a `.env`
  file at the repository root, which `python-dotenv` will load):
  - `ANTHROPIC_API_KEY` — required for Claude Haiku 4.5 theme extraction
    and for the rotating-judge step.
  - `GOOGLE_GENERATIVE_AI_API_KEY` — required for Gemini 2.5 Flash theme
    extraction, Gemini embeddings (used by clustering, grounding, and the
    Young 2015 anchor step), and the rotating-judge step.
- **Network access** to `api.anthropic.com`, `generativelanguage.googleapis.com`,
  and (for the real-corpus path only) `api.pullpush.io`.

The repository does not redistribute the collected Reddit corpus. You either
collect it yourself with `src/collect_corpus.py` or run on the bundled
synthetic sample.

## 2. Environment setup

```
python3.13 -m venv .venv
.venv/bin/pip install -U pip
.venv/bin/pip install -r requirements.txt
```

All commands below assume the virtualenv is on PATH; either prefix each
command with `.venv/bin/` or `source .venv/bin/activate` first.

Place your keys in a project-root `.env` file if you prefer that to
exporting them:

```
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_GENERATIVE_AI_API_KEY=AIza...
```

## 3. Quick smoke run on synthetic data

The repository ships with `data/synthetic_sample.jsonl`, a 100-record
fictional corpus that mirrors the real schema (`id`, `type`,
`subreddit`, `author`, `created_utc`, `title`, `text`, `permalink`,
`parent_id`). It is just large enough to exercise chunking, the
map-reduce theme extraction step, and every reliability metric without
running up a meaningful API bill.

Copy it into the path the downstream pipeline expects:

```
cp data/synthetic_sample.jsonl data/sample.jsonl
```

Then run the pipeline. Each step writes its own artifact, and every step
after `run_models.py` reads from the previous step's output rather than
from the LLM directly, so you can re-run any single stage without
re-paying for upstream calls.

```
# 1. Theme extraction: Claude Haiku 4.5 + Gemini 2.5 Flash, prompts A and B.
#    For the smoke run we cap chunk size and record count so it finishes fast.
python -m src.run_models \
    --temperature 0.0 \
    --run-id smoke \
    --chunk-tokens 4000 \
    --limit 100

# Produces:
#   outputs/raw/<model_id>__<prompt_id>__smoke.json        final theme list per run
#   outputs/raw/<model_id>__<prompt_id>__smoke.map.jsonl   per-chunk theme lists
#   outputs/run_index.json                                 timing + token usage index

# 2. Quote grounding: do the supporting_quotes appear in the corpus?
python -m src.grounding
# Produces eval/grounding.json with exact-match, BM25, and embedding
# similarity scores per quote, plus an overall grounding rate per run.

# 3. Cluster themes across all (model, prompt) pairs to measure agreement.
python -m src.cluster_themes
# Produces eval/clusters.json with connected-component cluster IDs,
# cluster membership, and a pairwise Jaccard agreement matrix.

# 4. Match each extracted theme to the Young 2015 systematic-review anchors.
python -m src.literature_anchor
# Produces eval/literature_anchor.json with the best-matching anchor
# theme and cosine similarity for every extracted theme.

# 5. LLM-as-judge with rotating judge (each model rates the other's themes).
python -m src.llm_judge
# Produces eval/judgments.json with rubric scores on faithfulness,
# comprehensiveness, clinical_relevance, specificity, and non_redundancy.

# 6. Combine every signal into a per-theme reliability score in [0, 1].
python -m src.reliability_score
# Produces eval/reliability.json. The composite is the unweighted mean
# of every available signal; component scores are preserved so
# reviewers can recompute under different weightings.

# 7. Render the manuscript tables and figures from the eval artifacts.
python -m src.build_report
# Produces:
#   manuscript/tables/{corpus_stats,run_index,anchor_coverage}.{tex,csv}
#   manuscript/figures/{grounding,cluster_heatmap,reliability}.png
```

If every step exits 0, the pipeline is wired up correctly. The smoke
run is not meaningful as a research result — the synthetic corpus is
designed to be schema-faithful, not population-faithful.

## 4. Running on the real corpus

The real corpus is collected fresh from pullpush.io, a community-run
Pushshift mirror. Pullpush is not always reliable; the collector retries
with exponential backoff and is resumable.

```
# 1. Collect r/endometriosis and r/ENDO across the published 2018-02-07
#    to 2024-10-31 window. This is the slowest step (tens of minutes
#    to hours depending on pullpush load) and is rate-limited politely.
python -m src.collect_corpus --out data/raw

# Produces:
#   data/raw/endometriosis_posts.jsonl
#   data/raw/endometriosis_comments.jsonl
#   data/raw/ENDO_posts.jsonl
#   data/raw/ENDO_comments.jsonl
#   data/corpus.manifest.json   sha256 + row counts + word total per file

# 2. Stratified-by-year subsample down to roughly the 2024 abstract size
#    (about 2,000 posts and 80,000 comments). Seeded for reproducibility.
python -m src.sample_corpus

# Produces:
#   data/sample.jsonl              the analysis-ready corpus
#   data/sample.manifest.json      seed, targets, sampled IDs, sha256
```

From here, every step is identical to the smoke run, but without the
`--limit 100` flag and at the production chunk size. Expect the full
sweep to take several hours of wall-clock time on Gemini Flash free tier
because of the rate cap (see Section 6).

```
python -m src.run_models --run-id primary
python -m src.grounding
python -m src.cluster_themes
python -m src.literature_anchor
python -m src.llm_judge
python -m src.reliability_score
python -m src.build_report
```

Stability runs (temperature 0.7, multiple seeds) are produced by re-running
`src.run_models` with a different `--run-id` and `--temperature`; the run
index file accumulates them.

## 5. Re-using a frozen corpus

Because pullpush.io can return slightly different rows on different days
(retroactive deletions, missing cache entries, etc.), all model runs in a
single experiment must read identical input. We anchor that with a content
hash manifest.

After `src/collect_corpus.py` finishes, `data/corpus.manifest.json` records:

- `collected_at` — UTC timestamp of the collection run
- `window` — start and end of the requested time window
- `stats.posts`, `stats.comments`, `stats.unique_users`, `stats.word_count`
- `files.<filename>.sha256` — SHA-256 of every JSONL file
- `files.<filename>.rows`, `files.<filename>.size_bytes`

The sample step writes its own manifest (`data/sample.manifest.json`)
containing the seed, the per-id list of records that ended up in
`sample.jsonl`, and an additional sha256 over the sample file.

To verify identical input across reruns:

```
# Recompute the SHA-256 over each raw file and diff against the manifest.
python - <<'PY'
import hashlib, json, pathlib
m = json.loads(pathlib.Path("data/corpus.manifest.json").read_text())
for name, meta in m["files"].items():
    h = hashlib.sha256(pathlib.Path("data/raw") / name).read_bytes() if False else \
        hashlib.sha256(pathlib.Path("data/raw", name).read_bytes()).hexdigest()
    status = "OK" if h == meta["sha256"] else "MISMATCH"
    print(f"{status}  {name}  expected={meta['sha256'][:12]}  actual={h[:12]}")
PY

# And likewise for the sample.
python -c "import hashlib, json, pathlib; \
m=json.loads(pathlib.Path('data/sample.manifest.json').read_text()); \
h=hashlib.sha256(pathlib.Path('data/sample.jsonl').read_bytes()).hexdigest(); \
print('OK' if h==m['sample_sha256'] else 'MISMATCH', h)"
```

Workflow recommendation: collect once, commit `corpus.manifest.json` and
`sample.manifest.json` to your local working tree, archive the raw JSONL
files outside of git, and re-verify before each model run. If any sha256
fails to match, do not proceed — the downstream eval files will silently
disagree about which records they reference.

## 6. Rate limits and costs

### 6.1 Gemini 2.5 Flash (free tier)

Gemini Flash on the free tier is throttled at roughly the published RPM cap
for `gemini-2.5-flash` (currently around 10 requests per minute, but
verify against the live quota page before a run). Practical implications:

- The `tenacity` retry decorator in `src/run_models.py` will absorb 429
  responses with exponential backoff, but a long sweep can spend most of
  its wall-clock time waiting on the per-minute window to reset.
- The map-reduce step issues one request per chunk plus one final reduce
  request per (model, prompt) pair. At production chunk size (8000
  tokens, ~6000 words), the full sample produces approximately 100-200
  Gemini chunks per prompt variant.
- If you have access to a paid quota, set `GOOGLE_GENERATIVE_AI_API_KEY`
  to that project's key and the same code path will run at a much higher
  RPM without any code change.
- Gemini embeddings (`gemini-embedding-001`, used by `grounding.py`,
  `cluster_themes.py`, and `literature_anchor.py`) share the same key
  and quota. Embedding the full sample plus all extracted theme texts is
  the next-largest source of requests after extraction itself.

### 6.2 Anthropic batch API hint

For the Claude Haiku 4.5 extraction step, the per-chunk requests are
embarrassingly parallel and read-only with respect to each other. The
Anthropic Message Batches API delivers roughly 50% off the synchronous
per-token rate and is well suited to this workload. The current
`src/run_models.py` uses the synchronous `messages.create` endpoint
because it keeps the failure mode simple (one chunk fails, retry that
chunk), but a batch-API adapter is a drop-in replacement at the
`AnthropicProvider.complete` boundary if cost matters more than
debug-loop latency.

The same code also benefits from prompt caching: the system prompt
(`PROMPT_A` or `PROMPT_B`) is constant across every chunk in a run.
Enabling prompt caching trims input-token billing on each chunk after the
first; see Anthropic's prompt caching docs for the header to set on the
underlying SDK.

### 6.3 Rough cost estimate for the full sweep

The numbers below are order-of-magnitude estimates at 2026-05 list
prices, assuming the production sample (~2,000 posts + ~80,000
comments, ~4M words). Treat them as upper bounds; prompt caching and
the batch API both meaningfully reduce them.

| Stage                                | Calls   | Tokens (in/out, approx) | Cost (USD, approx) |
|--------------------------------------|---------|--------------------------|--------------------|
| Theme extraction (Haiku 4.5, A+B)    | ~400    | 4M / 200k                | $5-10              |
| Theme extraction (Gemini Flash, A+B) | ~400    | 4M / 200k                | free tier (slow)   |
| Reduce step (both models, A+B)       | 4       | 200k / 20k               | <$1                |
| Grounding embeddings (Gemini)        | corpus  | 4M tokens embedded       | free tier          |
| Cluster + anchor embeddings (Gemini) | themes  | <100k tokens             | free tier          |
| LLM judge (Haiku judges Gemini, etc) | ~200    | 500k / 50k               | $2-3               |
| Stability sweep (3 seeds, T=0.7)     | 3x ext. | scales with above        | $15-30 total       |

Expect the headline figure to land in the **$25-50** range for a
single-pass primary run plus stability sweep, dominated by Anthropic
spend. The free-tier Gemini path is effectively zero dollars but
extends wall-clock time substantially.

If you only run the smoke pipeline on `data/synthetic_sample.jsonl`,
total spend is well under one dollar even without batching or caching.
