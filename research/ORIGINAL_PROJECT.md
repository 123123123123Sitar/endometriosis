# Endometriosis Reddit + LLM study (2026 continuation)

Continuation of a 2024 GPT-4 thematic analysis of `r/endometriosis` and `r/ENDO`.
Corpus snapshot: 1,986 posts / 79,993 comments / 17,049 users / 4,321,905 words
(02/07/2018 – 10/31/2024).

## Goals

1. Re-run with a current LLM under the available budget: **Claude Haiku
   4.5** via Claude Code subagents. GPT-4 (2024) themes are retained as
   a frozen published baseline (no OpenAI key available, so the 2024 run
   cannot be reproduced). A Gemini second-extractor run remains a planned
   follow-up because free-tier quota blocked the full extraction.
2. Rate the LLM interpretations rigorously without human raters.
3. Produce a peer-reviewable manuscript (venue TBD).

## Data

The senior mentor provided the original 2024-study corpus as a
Pushshift-format archive for `r/Endo`, ending 2022-12-31. The current
analysis filters that archive to the overlapping 2018-02-07 window,
samples 2,000 posts and 80,000 comments, and documents the missing
`r/endometriosis` and 2023-2024 coverage in Methods.

## Layout

```
config/      pinned model IDs + dates
data/        frozen corpus + manifest hashes (NOT redistributed)
prompts/     primary + paraphrased prompts
src/         runners, grounding, judge, clustering
outputs/     per-model JSON theme outputs
eval/        reliability tables + figures
manuscript/  LaTeX + bib + reporting checklist
```

## Reliability scoring (5 metrics)

1. Chunk-support / within-extractor recurrence across disjoint chunks
2. Grounding / faithfulness (exact + BM25/embedding retrieval)
3. Anchored validity vs Young 2015 (PMID 25183531)
4. LLM-as-judge with rotating judge (chain-of-thought rubric)
5. Cross-model and prompt/temperature stability are planned follow-ups

Combined into a per-theme reliability score 0–1.

## Ethics

Public, pseudonymous data. Manuscript publishes paraphrased quotes only — no
verbatim user posts, no DMs, no usernames. Expected IRB exemption (public
human-subjects exemption category).
