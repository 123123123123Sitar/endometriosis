"""Command-line interface."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from .analyze import ThemeAnalyzer
from .pipeline import run_extraction

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract all text from a folder, with OCR fallback for scans."
    )
    parser.add_argument("--input", type=Path, default=Path("input"))
    parser.add_argument("--output", type=Path, default=Path("output"))
    parser.add_argument(
        "--no-ocr",
        action="store_true",
        help="never call the API; images and scanned-only PDF pages will be skipped",
    )
    parser.add_argument(
        "--force-ocr",
        action="store_true",
        help="OCR every PDF page even when it already contains embedded text",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="also produce output/themes.json from the extracted corpus",
    )
    parser.add_argument("--model", help="override ANTHROPIC_MODEL")
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv(PROJECT_ROOT / ".env")
    args = build_parser().parse_args(argv)
    api_key = os.getenv("ANTHROPIC_API_KEY")
    model = args.model or os.getenv(
        "ANTHROPIC_MODEL", "claude-haiku-4-5-20251001"
    )

    try:
        manifest = run_extraction(
            input_dir=args.input,
            output_dir=args.output,
            api_key=api_key,
            model=model,
            use_ocr=not args.no_ocr,
            force_ocr=args.force_ocr,
        )
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(
        f"\nDone: {manifest['files_ok']} files, "
        f"{manifest['total_words']:,} words -> {args.output / 'all_text.txt'}"
    )
    if manifest["files_error"]:
        print(
            f"{manifest['files_error']} file(s) could not be extracted; "
            f"see {args.output / 'manifest.json'}"
        )

    if args.analyze:
        if not api_key:
            print(
                "error: --analyze requires ANTHROPIC_API_KEY in .env",
                file=sys.stderr,
            )
            return 2
        print("\nAnalyzing extracted text ...")
        analyzer = ThemeAnalyzer(api_key=api_key, model=model)
        final = analyzer.analyze(
            args.output / "all_text.txt",
            args.output / "themes.json",
        )
        print(
            f"Analysis done: {len(final['themes'])} themes -> "
            f"{args.output / 'themes.json'}"
        )

    return 1 if manifest["files_ok"] == 0 else 0
