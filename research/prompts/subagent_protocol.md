# Subagent theme-extraction protocol

This file documents the subagent workflow used by the completed
`claude_haiku_4_5__A__primary` run. The exact interactive Claude Code
subagent messages were not preserved in the repository at run time, so
this file should be treated as the canonical protocol for reruns and as
a reconstruction of the completed run's instructions from the committed
code, manuscript notes, and output artifacts.

## Current completed run

- Extractor: Claude Haiku 4.5
- Prompt variant: A
- Temperature: 0.0
- Chunking: `chunk_tokens=100000`, `overlap_tokens=2000`
- Chunk count: 74
- Chunk theme files: `outputs/chunks/chunk_NNN.themes.json`
- Aggregated map output:
  `outputs/raw/claude_haiku_4_5__A__primary.map.jsonl`
- Reduce output:
  `outputs/raw/claude_haiku_4_5__A__primary.json`
- Reduce model: Claude Sonnet 4.6 subagent
- Judge models: Claude Sonnet 4.6 and Claude Opus 4.7 subagents

## Map prompt for each chunk

Use Prompt A from `src/run_models.py` / `prompts/theme_extraction.md`.
For each chunk file, the subagent should read only its assigned
`outputs/chunks/chunk_NNN.txt` file and write
`outputs/chunks/chunk_NNN.themes.json`.

Required output:

```json
{
  "themes": [
    {
      "theme": "short noun phrase",
      "description": "2-3 sentence faithful summary",
      "supporting_quotes": ["up to 3 verbatim quotes, each <=25 words"],
      "frequency_estimate": "rough fraction or qualitative frequency",
      "severity_signal": "low | medium | high"
    }
  ]
}
```

The subagent must not invent quotes. Quote fields should contain text
literally present in the assigned chunk.

## Reduce prompt

Input: `outputs/_chunk_themes_for_reduce.json`, produced by
`src/aggregate_subagent_themes.py`.

Instruction:

```text
You will be given multiple per-chunk theme lists, each produced by
analyzing a slice of the same corpus. Merge them into a single global
theme list:
  - de-duplicate themes that describe the same underlying phenomenon
  - prefer the clearer description; do NOT invent new themes
  - propagate supporting_quotes (deduplicated) up to a max of 3 per theme
  - frequency_estimate should reflect the merged evidence ("most chunks",
    "minority of chunks", etc.)
  - add chunk_support: number of distinct chunks contributing evidence
    to the merged theme
  - add raw_member_count: audit count emitted by the reducer for raw
    theme assignments associated with the merged theme; this is
    descriptive only and is not used in reliability scoring
Output strict JSON: {"themes": [...]}.
```

## Judge rubric

Each judge subagent scores every final theme on a 1-5 scale for:

- faithfulness
- comprehensiveness
- clinical_relevance
- specificity
- non_redundancy

Each dimension should include a one-sentence justification. Outputs are
stored in `eval/judgments_sonnet.json`, `eval/judgments_opus.json`, and
combined in `eval/judgments.json`.
