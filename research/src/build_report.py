"""Generate the tables + figures used in the manuscript.

Reads:
  data/corpus.manifest.json
  data/sample.manifest.json
  outputs/run_index.json
  outputs/raw/*.json
  eval/clusters.json
  eval/grounding.json
  eval/judgments.json
  eval/literature_anchor.json
  eval/reliability.json

Writes:
  manuscript/figures/*.png
  manuscript/tables/*.tex
  manuscript/tables/*.csv
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / "manuscript" / "figures"
TBL_DIR = PROJECT_ROOT / "manuscript" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TBL_DIR.mkdir(parents=True, exist_ok=True)


def safe_load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def _tex_escape(s: str) -> str:
    """Escape characters that LaTeX treats specially in text mode."""
    return (s.replace("\\", r"\textbackslash{}")
             .replace("&", r"\&")
             .replace("%", r"\%")
             .replace("$", r"\$")
             .replace("#", r"\#")
             .replace("_", r"\_")
             .replace("{", r"\{")
             .replace("}", r"\}")
             .replace("~", r"\textasciitilde{}")
             .replace("^", r"\textasciicircum{}"))


def write_tex_table(path: Path, header: list[str], rows: list[list[str]],
                    caption: str, label: str, colspec: str | None = None,
                    small: bool = False) -> None:
    cols = colspec or "l" + "r" * (len(header) - 1)
    h_esc = [_tex_escape(c) for c in header]
    r_esc = [[_tex_escape(c) for c in row] for row in rows]
    body = " & ".join(h_esc) + r" \\" + "\n\\midrule\n"
    body += "\n".join(" & ".join(r) + r" \\" for r in r_esc)
    size = "\\small\n" if small else ""
    tex = f"""\\begin{{table}}[ht]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{size}\\setlength{{\\tabcolsep}}{{4pt}}
\\begin{{tabular}}{{{cols}}}
\\toprule
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    path.write_text(tex)


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    import csv
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def table_corpus_stats() -> None:
    full = safe_load(PROJECT_ROOT / "data" / "corpus.manifest.json")
    sample = safe_load(PROJECT_ROOT / "data" / "sample.manifest.json")
    if not full:
        return
    full_stats = full["stats"]
    sample_stats = sample["stats"] if sample else {}
    rows = [
        ["Posts", str(full_stats["posts"]), str(sample_stats.get("posts", "—"))],
        ["Comments", str(full_stats["comments"]), str(sample_stats.get("comments", "—"))],
        ["Unique users", str(full_stats["unique_users"]), str(sample_stats.get("unique_users", "—"))],
        ["Words", f"{full_stats['word_count']:,}", f"{sample_stats.get('word_count', '—'):,}" if sample_stats else "—"],
    ]
    rows.insert(0, ["", "Full archive", "Analysis sample"])
    rows[0] = ["Metric", "Full archive", "Analysis sample"]
    write_tex_table(
        TBL_DIR / "corpus_stats.tex",
        rows[0],
        rows[1:],
        caption="Mentor-provided Pushshift-format r/Endo archive filtered to the overlapping 2018-02-07 to 2022-12-31 window. The analysis sample was stratified by year to match the size of the original 2024 study.",
        label="tab:corpus",
    )
    write_csv(TBL_DIR / "corpus_stats.csv", rows[0], rows[1:])


def table_run_index() -> None:
    # Prefer the JSONL append-log (safe for concurrent writers); fall back
    # to the legacy run_index.json if it exists.
    jsonl = PROJECT_ROOT / "outputs" / "run_index.jsonl"
    legacy = PROJECT_ROOT / "outputs" / "run_index.json"
    idx = None
    if jsonl.exists():
        idx = [json.loads(line) for line in jsonl.read_text().splitlines() if line.strip()]
    elif legacy.exists():
        idx = safe_load(legacy)
    if not idx:
        return
    header = ["Model", "Prompt", "Run", "Chunks", "Themes", "In tokens", "Out tokens"]
    rows = []
    for r in idx:
        rows.append([
            r["model_id"], r["prompt_id"], r["run_id"],
            str(r["n_chunks"]), str(r["n_themes"]),
            f"{r['usage'].get('input_tokens', 0):,}",
            f"{r['usage'].get('output_tokens', 0):,}",
        ])
    write_tex_table(
        TBL_DIR / "run_index.tex",
        header, rows,
        caption="Completed theme extraction run.",
        label="tab:runs",
    )
    write_csv(TBL_DIR / "run_index.csv", header, rows)


def figure_grounding() -> None:
    g = safe_load(PROJECT_ROOT / "eval" / "grounding.json")
    if not g:
        return
    names = list(g.keys())
    rates = [g[n].get("overall_grounding_rate", 0.0) for n in names]
    fig, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(names))))
    ax.barh(names, rates, color="#4C78A8")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Quote grounding rate")
    ax.set_title("Per-run quote grounding (exact + BM25)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "grounding.png", dpi=150)
    plt.close(fig)


def figure_cluster_heatmap() -> None:
    c = safe_load(PROJECT_ROOT / "eval" / "clusters.json")
    if not c:
        return
    pairs = c.get("pairwise_jaccard", [])
    models = sorted({p["a"] for p in pairs} | {p["b"] for p in pairs})
    if len(models) < 2:
        return
    n = len(models)
    M = np.zeros((n, n))
    for p in pairs:
        i, j = models.index(p["a"]), models.index(p["b"])
        M[i, j] = M[j, i] = p["jaccard"]
    fig, ax = plt.subplots(figsize=(1.5 + 0.7 * n, 1.5 + 0.7 * n))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(range(n), models, rotation=45, ha="right")
    ax.set_yticks(range(n), models)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                    color="white" if M[i, j] < 0.5 else "black", fontsize=9)
    ax.set_title("Pairwise theme-cluster Jaccard between models")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "cluster_heatmap.png", dpi=150)
    plt.close(fig)


def table_reliability_per_theme() -> None:
    r = safe_load(PROJECT_ROOT / "eval" / "reliability.json")
    if not r:
        return
    # Pull chunk_support directly from the source themes file.
    src_themes_path = PROJECT_ROOT / "outputs" / "raw" / "claude_haiku_4_5__A__primary.json"
    src_themes = (json.loads(src_themes_path.read_text()).get("themes", [])
                  if src_themes_path.exists() else [])
    chunk_support_by_theme = {
        t.get("theme", ""): t.get("chunk_support", "") for t in src_themes
    }
    rows = []
    for t in sorted(r["themes"], key=lambda x: -(x.get("composite_reliability") or 0)):
        sig = t.get("signals", {})
        rows.append([
            t.get("theme", ""),
            str(chunk_support_by_theme.get(t.get("theme", ""), "")),
            f"{sig.get('grounding', 0.0):.2f}" if "grounding" in sig else "",
            f"{sig.get('judge', 0.0):.2f}" if "judge" in sig else "",
            f"{sig.get('literature_anchor', 0.0):.2f}" if "literature_anchor" in sig else "",
            f"{(t.get('composite_reliability') or 0):.2f}",
        ])
    write_tex_table(
        TBL_DIR / "reliability_per_theme.tex",
        ["Theme", "k_chunk", "Ground", "Judge", "Anchor", "Composite"],
        rows,
        caption="Per-theme reliability scores. Ground = exact + BM25 quote-grounding rate. Judge = mean rubric score from Sonnet + Opus rotating judges, normalized to [0, 1]. Anchor = lexical match to Young 2015 (binary). Composite is the unweighted mean of available signals.",
        label="tab:reliability",
        colspec=r"p{0.44\linewidth}rrrrr",
        small=True,
    )
    write_csv(TBL_DIR / "reliability_per_theme.csv",
              ["Theme", "k_chunk", "Ground", "Judge", "Anchor", "Composite"], rows)


def figure_reliability_distribution() -> None:
    r = safe_load(PROJECT_ROOT / "eval" / "reliability.json")
    if not r:
        return
    by_model: dict[str, list[float]] = defaultdict(list)
    for t in r.get("themes", []):
        if t.get("composite_reliability") is not None:
            by_model[t["model_id"]].append(t["composite_reliability"])
    if not by_model:
        return
    models = sorted(by_model)
    data = [by_model[m] for m in models]
    fig, ax = plt.subplots(figsize=(1.5 + 1.2 * len(models), 4))
    ax.boxplot(data, tick_labels=models)
    ax.set_ylabel("Composite reliability (0-1)")
    ax.set_title("Per-theme composite reliability by model")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliability.png", dpi=150)
    plt.close(fig)


def table_anchor_coverage() -> None:
    a = safe_load(PROJECT_ROOT / "eval" / "literature_anchor.json")
    anchor = safe_load(PROJECT_ROOT / "eval" / "young_2015_themes.json")
    if not a or not anchor:
        return
    ids = [t["id"] for t in anchor["themes"]]
    by_anchor = {i: 0 for i in ids}
    by_anchor_models: dict[str, set] = {i: set() for i in ids}
    for r in a:
        if r["matched"]:
            by_anchor[r["best_anchor_id"]] += 1
            by_anchor_models[r["best_anchor_id"]].add(r["model_id"])
    header = ["Anchor theme (Young 2015)", "Matched themes", "Models contributing"]
    rows = []
    for t in anchor["themes"]:
        rows.append([
            t["label"],
            str(by_anchor[t["id"]]),
            str(len(by_anchor_models[t["id"]])),
        ])
    write_tex_table(
        TBL_DIR / "anchor_coverage.tex",
        header, rows,
        caption="Coverage of Young 2015 themes by extracted themes (TF-IDF cosine similarity threshold 0.05).",
        label="tab:anchor",
    )
    write_csv(TBL_DIR / "anchor_coverage.csv", header, rows)


def main() -> int:
    table_corpus_stats()
    table_run_index()
    table_anchor_coverage()
    table_reliability_per_theme()
    figure_grounding()
    figure_cluster_heatmap()
    figure_reliability_distribution()
    print("Tables and figures regenerated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
