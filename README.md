# Endometriosis Text Extraction Pipeline

A single-command pipeline for extracting text from research files. It handles
large Reddit JSONL/Zstandard archives, ordinary documents, PDFs, and images.
Machine-readable text is extracted locally. Images and scanned PDF pages use
Anthropic only for exact transcription.

The optional analysis step recreates the core purpose of the original
endometriosis LLM project: map-reduce theme extraction over the full corpus.
The original multi-model reliability and manuscript workflow is retained under
`research/` for advanced reproduction work.

## Quick start

Python 3.11 or newer is recommended.

```bash
git clone https://github.com/123123123123Sitar/endometriosis.git
cd endometriosis

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Add input files

Put files anywhere under `input/`. Subfolders are supported.

```bash
cp /path/to/your/files/* input/
```

Supported inputs:

- `.txt`, `.md`, `.rst`, `.log`
- `.json`, `.jsonl`, `.ndjson`
- `.zst` containing JSON Lines, including Pushshift Reddit archives
- `.csv`, `.tsv`
- `.pdf`, `.docx`, `.html`
- `.jpg`, `.jpeg`, `.png`, `.gif`, `.webp`, `.tif`, `.tiff`, `.bmp`

For Reddit objects, the extractor keeps content such as `title`, `selftext`,
and `body` while excluding common identifiers such as author, ID, permalink,
and timestamp.

### 2. Add the API key

```bash
cp .env.example .env
```

Open `.env` and set:

```dotenv
ANTHROPIC_API_KEY=your-real-key
```

The key is required for images, scanned PDF pages, and `--analyze`. Plain text,
JSON, CSV, DOCX, HTML, and PDFs with embedded text can be processed locally
without API calls.

### 3. Run

```bash
python run.py
```

That is the complete extraction workflow.

To also extract endometriosis themes from all collected text:

```bash
python run.py --analyze
```

## Output

Every run replaces the generated files under `output/`:

| Path | Contents |
| --- | --- |
| `output/all_text.txt` | All extracted text combined, with source headers |
| `output/files/` | One text file per input file |
| `output/records.jsonl` | Streaming records with source, unit/page, and text |
| `output/manifest.json` | Counts, extraction method, hashes, and errors |
| `output/themes.json` | Optional themes created by `--analyze` |

Input files, output files, and `.env` are ignored by Git so private research
data and API keys are not committed.

## Useful commands

Extract without making any API requests:

```bash
python run.py --no-ocr
```

Force OCR on every PDF page:

```bash
python run.py --force-ocr
```

Use different folders:

```bash
python run.py --input /path/to/files --output /path/to/results
```

Select a different Anthropic model:

```bash
python run.py --model MODEL_ID
```

Run the tests:

```bash
python -m unittest discover -s tests -v
```

## What it does

1. Recursively finds files under the input folder.
2. Streams large text, JSONL, and `.zst` archives to keep memory use bounded.
3. Extracts embedded text from PDF, DOCX, HTML, JSON, and tabular files.
4. Sends only images or scanned PDF pages to Anthropic for transcription.
5. Writes per-file text, one combined corpus, JSONL records, and a manifest.
6. With `--analyze`, chunks the combined corpus, extracts themes, and merges
   duplicate themes into one JSON result.

The OCR prompt requests transcription only. It does not ask the model to
summarize, diagnose, or alter source text. OCR can still make mistakes, so
verify important excerpts against the original file.

```bash
cd research
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Raw corpus files and quote-bearing model outputs are intentionally absent.
