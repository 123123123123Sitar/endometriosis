# Theme-extraction prompts

Two prompts were specified for the original API pipeline so we could
measure prompt sensitivity. The completed manuscript run currently uses
Prompt A only, with Claude Haiku 4.5 subagents. The subagent-specific
rerun protocol is documented in `prompts/subagent_protocol.md`.

Prompt A is the original 2024 prompt transcribed from available notes;
the verbatim 2024 prompt has not been recovered. Prompt B is an
independently-paraphrased variant designed to test robustness, not to
lead the model toward the same conclusions.

The API map-reduce pipeline calls each prompt twice:
  - MAP step:    summarize themes within a single chunk (~8k tokens of corpus).
  - REDUCE step: merge per-chunk theme lists into a single global theme list.

The completed subagent run used 100k-token chunks and a Sonnet subagent
for reduction rather than this synchronous API path.

Output for both steps is structured JSON (see `output_schema` below).

---

## Prompt A — original 2024 prompt (placeholder, replace if exact text recovered)

```
You are an analyst summarizing patient experiences and perspectives in an
endometriosis-themed Reddit community. Read the posts and comments provided
and identify the main themes regarding endometriosis CARE AND TREATMENT.

For each theme, output:
  - theme: a short noun phrase
  - description: 2–3 sentence summary in your own words
  - supporting_quotes: up to 3 verbatim quotes (≤25 words each) from the corpus
  - frequency_estimate: rough fraction of relevant posts mentioning the theme
  - severity_signal: low | medium | high (how distressed users sound)

Return strict JSON conforming to the schema. Do not invent quotes; only use
text that appears verbatim in the input.
```

## Prompt B — independent paraphrase

```
You are reviewing patient-written discussions in an online endometriosis
community. Without bias toward any particular treatment philosophy, identify
the recurring topics that come up around how endometriosis is managed,
treated, or experienced in clinical settings.

For each topic you identify, return:
  - theme: short label
  - description: a faithful 2–3 sentence paraphrase
  - supporting_quotes: up to 3 short verbatim excerpts (≤25 words each)
  - frequency_estimate: approximate share of relevant posts where it appears
  - severity_signal: low | medium | high based on the affective tone

Output must be strict JSON. Quote only text that is literally present in the
input — do not paraphrase inside quote fields.
```

## output_schema (JSON Schema, draft 2020-12)

```json
{
  "type": "object",
  "required": ["themes"],
  "properties": {
    "themes": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["theme", "description", "supporting_quotes", "frequency_estimate", "severity_signal"],
        "properties": {
          "theme": {"type": "string", "minLength": 3, "maxLength": 80},
          "description": {"type": "string", "minLength": 30, "maxLength": 600},
          "supporting_quotes": {
            "type": "array",
            "items": {"type": "string", "minLength": 5, "maxLength": 200},
            "maxItems": 3
          },
          "frequency_estimate": {"type": "string"},
          "severity_signal": {"enum": ["low", "medium", "high"]}
        }
      }
    }
  }
}
```

---

## REDUCE-step instruction (appended to either prompt)

```
You will be given multiple per-chunk theme lists, each produced by analyzing
a slice of the same corpus. Merge them into a single global theme list:
  - de-duplicate themes that describe the same underlying phenomenon
  - prefer the clearer description; do NOT invent new themes
  - propagate supporting_quotes (deduplicated) up to a max of 3 per theme
  - frequency_estimate should reflect the merged evidence ("most chunks",
    "minority of chunks", etc.)
Output strict JSON in the same schema.
```
