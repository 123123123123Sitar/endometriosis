from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import zstandard

from endometriosis_pipeline.pipeline import run_extraction


class PipelineTests(unittest.TestCase):
    def test_extracts_text_jsonl_and_csv_without_api(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()

            (input_dir / "notes.txt").write_text(
                "First note.\nSecond note.\n", encoding="utf-8"
            )
            (input_dir / "reddit.jsonl").write_text(
                json.dumps(
                    {
                        "id": "abc",
                        "author": "private-user",
                        "title": "A title",
                        "selftext": "The post body.",
                    }
                )
                + "\n"
                + json.dumps({"body": "A comment.", "permalink": "/r/test"})
                + "\n",
                encoding="utf-8",
            )
            (input_dir / "table.csv").write_text(
                "heading,value\npain,severe\n", encoding="utf-8"
            )

            manifest = run_extraction(
                input_dir=input_dir,
                output_dir=output_dir,
                use_ocr=False,
            )

            self.assertEqual(manifest["files_ok"], 3)
            combined = (output_dir / "all_text.txt").read_text(encoding="utf-8")
            self.assertIn("First note.", combined)
            self.assertIn("A title", combined)
            self.assertIn("The post body.", combined)
            self.assertIn("A comment.", combined)
            self.assertNotIn("private-user", combined)
            self.assertIn("pain\tsevere", combined)

            records = [
                json.loads(line)
                for line in (output_dir / "records.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertGreaterEqual(len(records), 4)

    def test_streams_zstandard_json_lines(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            raw = (
                json.dumps({"title": "Compressed title", "selftext": "Compressed body"})
                + "\n"
                + json.dumps({"body": "Compressed comment"})
                + "\n"
            ).encode("utf-8")
            (input_dir / "archive.zst").write_bytes(
                zstandard.ZstdCompressor().compress(raw)
            )

            manifest = run_extraction(
                input_dir=input_dir,
                output_dir=output_dir,
                use_ocr=False,
            )

            self.assertEqual(manifest["files_ok"], 1)
            combined = (output_dir / "all_text.txt").read_text(encoding="utf-8")
            self.assertIn("Compressed title", combined)
            self.assertIn("Compressed body", combined)
            self.assertIn("Compressed comment", combined)

    def test_records_unsupported_file_in_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            (input_dir / "archive.bin").write_bytes(b"\x00\x01")

            manifest = run_extraction(
                input_dir=input_dir,
                output_dir=output_dir,
                use_ocr=False,
            )

            self.assertEqual(manifest["files_error"], 1)
            self.assertEqual(manifest["files"][0]["status"], "error")


if __name__ == "__main__":
    unittest.main()
