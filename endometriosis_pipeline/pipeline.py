"""End-to-end extraction orchestration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .extractors import UnsupportedFileError, discover_files, extractor_for
from .ocr import AnthropicOCR, MissingApiKeyError


@dataclass
class FileResult:
    source: str
    output: str | None
    status: str
    method: str | None = None
    units: int = 0
    characters: int = 0
    words: int = 0
    sha256: str | None = None
    error: str | None = None


def _output_path(output_dir: Path, relative_source: Path) -> Path:
    return output_dir / "files" / relative_source.parent / f"{relative_source.name}.txt"


def run_extraction(
    input_dir: Path,
    output_dir: Path,
    api_key: str | None = None,
    model: str | None = None,
    use_ocr: bool = True,
    force_ocr: bool = False,
) -> dict:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = discover_files(input_dir)
    if not files:
        raise FileNotFoundError(f"no input files found in {input_dir}")

    ocr = None
    if use_ocr:
        try:
            ocr = AnthropicOCR(api_key=api_key, model=model)
        except MissingApiKeyError:
            # Machine-readable files still work. OCR-only files are recorded
            # as errors with a precise message.
            ocr = None

    combined_path = output_dir / "all_text.txt"
    records_path = output_dir / "records.jsonl"
    results: list[FileResult] = []

    with (
        combined_path.open("w", encoding="utf-8") as combined,
        records_path.open("w", encoding="utf-8") as records,
    ):
        for source in files:
            relative = source.relative_to(input_dir)
            destination = _output_path(output_dir, relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            print(f"Extracting {relative} ...")

            try:
                extractor = extractor_for(source, ocr=ocr, force_ocr=force_ocr)
                digest = hashlib.sha256()
                unit_count = character_count = word_count = 0
                wrote_header = False

                with destination.open("w", encoding="utf-8") as per_file:
                    for unit in extractor.units:
                        text = unit.text.strip()
                        if not text:
                            continue
                        if not wrote_header:
                            combined.write(f"\n\n===== {relative.as_posix()} =====\n\n")
                            wrote_header = True

                        payload = text + "\n\n"
                        per_file.write(payload)
                        combined.write(payload)
                        digest.update(payload.encode("utf-8"))
                        unit_count += 1
                        character_count += len(text)
                        word_count += len(text.split())
                        records.write(
                            json.dumps(
                                {
                                    "source": relative.as_posix(),
                                    "unit": unit.unit,
                                    "text": text,
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )

                result = FileResult(
                    source=relative.as_posix(),
                    output=str(destination.relative_to(output_dir)),
                    status="ok" if unit_count else "empty",
                    method=extractor.method,
                    units=unit_count,
                    characters=character_count,
                    words=word_count,
                    sha256=digest.hexdigest(),
                )
            except (UnsupportedFileError, OSError, ValueError, RuntimeError) as error:
                if destination.exists():
                    destination.unlink()
                result = FileResult(
                    source=relative.as_posix(),
                    output=None,
                    status="error",
                    error=str(error),
                )
                print(f"  skipped: {error}")
            results.append(result)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "files_seen": len(files),
        "files_ok": sum(result.status == "ok" for result in results),
        "files_empty": sum(result.status == "empty" for result in results),
        "files_error": sum(result.status == "error" for result in results),
        "total_characters": sum(result.characters for result in results),
        "total_words": sum(result.words for result in results),
        "files": [asdict(result) for result in results],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest
