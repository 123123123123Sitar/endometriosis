"""Optional map-reduce theme analysis over the extracted corpus."""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential


THEME_PROMPT = """Analyze this patient-written endometriosis corpus and identify
recurring themes about care, diagnosis, symptoms, treatment, and clinical
experiences. For each theme return a short label, a faithful description, up
to three short supporting excerpts copied exactly from the input, and a
severity signal of low, medium, or high.

Return only JSON in this form:
{"themes": [{"theme": "...", "description": "...", "supporting_quotes": [],
"severity_signal": "low|medium|high"}]}

Do not invent evidence and do not include identifying metadata."""


REDUCE_PROMPT = """Merge the supplied per-chunk endometriosis themes into one
deduplicated global list. Keep only themes supported by the supplied results.
Preserve up to three exact supporting quotes per theme.

Return only JSON in the same {"themes": [...]} format."""


def _chunks(path: Path, target_chars: int = 120_000) -> Iterator[str]:
    current: list[str] = []
    size = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if size + len(line) > target_chars and current:
                yield "".join(current)
                current = []
                size = 0
            current.append(line)
            size += len(line)
    if current:
        yield "".join(current)


def _parse_json(text: str) -> dict:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    value = json.loads(text)
    if isinstance(value, list):
        return {"themes": value}
    if isinstance(value, dict) and isinstance(value.get("themes"), list):
        return value
    raise ValueError("model response did not contain a themes list")


class ThemeAnalyzer:
    def __init__(self, api_key: str, model: str):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def _complete(self, prompt: str, content: str) -> dict:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            temperature=0,
            system=prompt,
            messages=[{"role": "user", "content": content}],
        )
        text = "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        )
        return _parse_json(text)

    def analyze(self, corpus_path: Path, output_path: Path) -> dict:
        mapped: list[dict] = []
        for index, chunk in enumerate(_chunks(corpus_path), 1):
            result = self._complete(THEME_PROMPT, chunk)
            mapped.append({"themes": result["themes"]})
            print(f"  analyzed chunk {index}: {len(result['themes'])} themes")

        if not mapped:
            final = {"themes": [], "chunks_analyzed": 0, "model": self.model}
        else:
            reduction_round = 0
            while len(mapped) > 1:
                reduction_round += 1
                reduced_batches: list[dict] = []
                for start in range(0, len(mapped), 20):
                    batch = mapped[start : start + 20]
                    reduced_batches.append(
                        self._complete(
                            REDUCE_PROMPT,
                            json.dumps(batch, ensure_ascii=False),
                        )
                    )
                mapped = reduced_batches
                print(
                    f"  reduction round {reduction_round}: "
                    f"{len(mapped)} result batch(es)"
                )
            reduced = mapped[0]
            final = {
                "themes": reduced["themes"],
                "chunks_analyzed": index,
                "model": self.model,
            }
        output_path.write_text(
            json.dumps(final, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        return final
