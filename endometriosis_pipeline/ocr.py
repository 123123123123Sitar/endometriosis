"""Anthropic-backed OCR for images and scanned PDF pages."""

from __future__ import annotations

import base64
import io
import os
from pathlib import Path

from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential


OCR_PROMPT = """Transcribe every visible word in this image exactly as written.
Preserve reading order and useful line breaks. Include headings, labels,
captions, table cells, and handwritten text when legible. Do not summarize,
interpret, correct, or add commentary. Return only the transcription."""


class MissingApiKeyError(RuntimeError):
    """Raised when an API-backed operation is requested without a key."""


class AnthropicOCR:
    def __init__(self, api_key: str | None = None, model: str | None = None):
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise MissingApiKeyError(
                "ANTHROPIC_API_KEY is required for images and scanned PDFs. "
                "Add it to .env or use --no-ocr."
            )

        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model or os.getenv(
            "ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
        )

    @staticmethod
    def _prepare_image(data: bytes, media_type: str) -> tuple[bytes, str]:
        """Keep API payloads bounded while retaining readable document detail."""
        if len(data) <= 4_500_000 and media_type in {
            "image/jpeg",
            "image/png",
            "image/gif",
            "image/webp",
        }:
            return data, media_type

        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB")
            max_side = 2400
            if max(image.size) > max_side:
                scale = max_side / max(image.size)
                image = image.resize(
                    (int(image.width * scale), int(image.height * scale)),
                    Image.Resampling.LANCZOS,
                )
            out = io.BytesIO()
            image.save(out, format="JPEG", quality=88, optimize=True)
            return out.getvalue(), "image/jpeg"

    @retry(
        wait=wait_exponential(multiplier=2, min=2, max=30),
        stop=stop_after_attempt(4),
        reraise=True,
    )
    def transcribe_bytes(self, data: bytes, media_type: str) -> str:
        data, media_type = self._prepare_image(data, media_type)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=8192,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64.b64encode(data).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": OCR_PROMPT},
                    ],
                }
            ],
        )
        return "".join(
            block.text
            for block in response.content
            if getattr(block, "type", "") == "text"
        ).strip()

    def transcribe_file(self, path: Path) -> str:
        media_types = {
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png": "image/png",
            ".gif": "image/gif",
            ".webp": "image/webp",
            ".tif": "image/tiff",
            ".tiff": "image/tiff",
            ".bmp": "image/bmp",
        }
        return self.transcribe_bytes(path.read_bytes(), media_types[path.suffix.lower()])
