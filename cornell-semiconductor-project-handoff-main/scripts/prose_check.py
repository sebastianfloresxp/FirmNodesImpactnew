#!/usr/bin/env python3
"""Run LanguageTool grammar/style checks on Markdown prose.

Strips fenced code blocks, inline code, and URLs before checking so that
code snippets don't trigger false positives. Defaults to LanguageTool 6.3
(requires Java 17+); set PROSE_LT_VERSION=5.9 for Java 9-16.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INCLUDES = ["*.md"]
DEFAULT_EXCLUDES = [
    "report/**",
    "research/**",
    "presentation/**",
    "data/**",
    "artifacts/**",
    "results/**",
    "logs/**",
    ".venv-arch/**",
    "node_modules/**",
    # Internal planning/audit documents — not deliverable content
    "PROSE_FIX.md",
    "DEAD_CODE.md",
    "QA_REPORT.md",
    "STYLE_REFERENCE_GUIDE.md",
    "TODO.md",
    "TODO_M1_M3.md",
    "BILLING.md",
]

FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
URL_RE = re.compile(r"https?://\S+")
MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
HTML_RE = re.compile(r"<[^>]+>")

# Project-specific terms that LT's spell-checker shouldn't flag
ALLOWLIST = {
    "conda",
    "miniconda",
    "FactSet",
    "PyG",
    "CUDA",
    "USAspending",
    "HPO",
    "Dockerfile",
    "rclone",
    "Rclone",
    "dbt",
    "Makefile",
    "pyproject",
    "GraphSAGE",
    "TGNN",
    "Node2Vec",
    "Codespell",
    "Pyright",
    "Bandit",
    "Ruff",
    "Vulture",
    "openjdk",
    "Optuna",
    "PyTorch",
    "scikit-learn",
    "FAISS",
    "DuckDB",
    "Parquet",
    "repo",
    "subgraph",
    "codebook",
    "semiconductor",
    "semiconductors",
    # Government / regulatory acronyms
    "ANAG",
    "NDAA",
    "DFARS",
    "DoDI",
    "HASC",
    "DAIMS",
    "AppA",
    "AppB",
    "AppC",
    "IRs",
    "tpp",
    "NDIA",
    "NTIA",
    "IJCAI",
    "NSCAI",
    "USITC",
    # Author and institution names
    "McGillis",
    "Reichert",
    "Motwani",
    "Pease",
    "kaplan",
    "Bekker",
    "Bhattacharya",
    "Brin",
    "Craighead",
    "Elkan",
    "Fellegi",
    "Goodfellow",
    "Gopal",
    "Guin",
    "Jaccard",
    "Hauck",
    "Heusel",
    "Hoehn",
    "Holme",
    "Kanarik",
    "Kivelä",
    "Kuon",
    "Matsuo",
    "Neamen",
    "Neisser",
    "Nickolls",
    "Ning",
    "Pareja",
    "Pergolizzi",
    "Pettit",
    "Plessis",
    "Puurunen",
    "Renesas",
    "Shivakumar",
    "Taur",
    "Tehranipoor",
    "Vergun",
    "Weste",
    "Xu",
    # Library and tool names (appear in prose, not caught by code-stripping)
    "networkx",
    "scipy",
    "numpy",
    "matplotlib",
    "sklearn",
    "graphsage",
    "twotower",
    "tgnn",
    "n2v",
    "topk",
    "node2vec",
    "fuzzywuzzy",
    "pandas",
    "seaborn",
    "plotly",
    # TeX/LaTeX tools and concepts
    "xelatex",
    "pdflatex",
    "pdfLaTeX",
    "Biber",
    "hyperref",
    "TikZ",
    # Project-specific technical terms
    "frontmatter",
    "hardcoded",
    "READMEs",
    "mixup",
    "chokepoint",
    "chokepoints",
    "crossview",
    "Decile",
    "sme",
    "dod",
    "misattribution",
    # Data and database terms
    "RBICS",
    "JEDEC",
    "IRDS",
    "HIR",
    "HHI",
    "OOF",
    "OSTI",
    "OSTEOMED",
    # Technical prose terms
    "agentic",
    "booktitle",
    "booktitles",
    "centralities",
    "codenames",
    "codespell",
    "contextlib",
    "cornell-semiconductor-project",
    "dataclass",
    "DataFrame",
    "datetime",
    "dedup",
    "docstring",
    "DoD‑anchored",
    "DOIs",
    "downweight",
    "Downweight",
    "du",
    "dyad",
    "emb",
    "ent",
    "EntityGraph",
    "entrypoints",
    "erroring",
    "eval",
    "EvolveGCN",
    "ffi",
    "fi",
    "Formalisms",
    "fuze",
    "GaN",
    "global-τ",
    "GraphSAGE-specific",
    "Graphviz",
    "highconf",
    "hoverable",
    "howpublished",
    "Hualien",
    "identificational",
    "json",
    "lensing",
    "Mw",
    "non-init",
    "nosec",
    "params",
    "parencite",
    "parencites",
    "pathlib",
    "pymarkdown",
    "pyright",
    "reframings",
    "- rclone",
    "rollup",
    "runbook",
    "SHAs",
    "snapshotting",
    "stdlib",
    "stochastics",
    "subagent-driven-development",
    "Subgraph",
    "subgraphs",
    "Subgraphs",
    "subprocess",
    "supply‑chain",
    "SWaP",
    "sys",
    "tabulars",
    "tex-ready",
    "TGAT",
    "Tohoku",
    "toml",
    "TwoHopCache",
    "TwoTower",
    "uncited",
    "UnionFind",
    "unreachability",
    "workstream",
    "writeup",
    "writeup-ready",
    "XeLaTeX",
    "Zotero",
    "Kreps",
    "WICKR",
}


def line_col_from_offset(text: str, offset: int) -> tuple[int, int]:
    """Translate a character offset into (1-based line, 1-based column)."""
    line = text.count("\n", 0, offset) + 1
    col = offset - (text.rfind("\n", 0, offset) if text.rfind("\n", 0, offset) != -1 else -1)
    return line, col


def strip_markdown(text: str) -> str:
    """Remove code/URLs/HTML so LanguageTool only sees prose.

    Inline code is replaced with the word 'code' rather than whitespace so that
    surrounding sentence structure remains grammatical (no double spaces next to
    punctuation, which trigger false COMMA_PARENTHESIS_WHITESPACE hits).
    """
    text = FENCE_RE.sub(lambda m: "\n" * m.group(0).count("\n"), text)
    text = INLINE_CODE_RE.sub("code", text)
    text = MD_LINK_RE.sub(r"\1", text)
    text = URL_RE.sub("URL", text)
    text = HTML_RE.sub(" ", text)
    return text


def collect_files(root: Path, excludes: list[str]) -> list[Path]:
    files: list[Path] = []
    for pat in DEFAULT_INCLUDES:
        for p in root.rglob(pat):
            rel = p.relative_to(root)
            if any(rel.match(ex) for ex in excludes):
                continue
            if any(part.startswith(".") for part in rel.parts):
                continue
            files.append(p)
    return sorted(files)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", help="Specific Markdown files (defaults to all repo .md)"
    )
    parser.add_argument("--language", default="en-US")
    parser.add_argument(
        "--disable",
        default=(
            "WHITESPACE_RULE,EN_QUOTES,UPPERCASE_SENTENCE_START,"
            "ARROWS,LC_AFTER_PERIOD,SETUP_VERB,CD_NN,"
            "COMMA_COMPOUND_SENTENCE,COMMA_COMPOUND_SENTENCE_2,"
            "BY_DEFAULT_COMMA,NEEDS_FIXED,ID_CASING,"
            "POSSESSIVE_APOSTROPHE,"
            "EN_COMPOUNDS_MULTI_MODEL,EN_COMPOUNDS_META_MODEL,"
            "EN_COMPOUNDS_MISSION_CRITICAL,NUMBERS_IN_WORDS"
        ),
        help="Comma-separated LanguageTool rule IDs to skip",
    )
    args = parser.parse_args()

    try:
        import language_tool_python
    except ImportError:
        print("error: language-tool-python not installed. Run `make qa-install`.", file=sys.stderr)
        return 2

    lt_version = os.environ.get("PROSE_LT_VERSION", "6.3")
    os.environ.setdefault("LTP_JAR_DIR_NAME", f"LanguageTool-{lt_version}")

    try:
        tool = language_tool_python.LanguageTool(
            args.language,
            language_tool_download_version=lt_version,
        )
    except Exception as exc:
        print(f"error: could not start LanguageTool ({exc}).", file=sys.stderr)
        print(
            "hint: LanguageTool requires Java. Set PROSE_LT_VERSION=6.3 for Java 17+.",
            file=sys.stderr,
        )
        return 2

    disabled = {r.strip() for r in args.disable.split(",") if r.strip()}

    files = (
        [Path(p) for p in args.paths] if args.paths else collect_files(REPO_ROOT, DEFAULT_EXCLUDES)
    )
    if not files:
        print("no markdown files to check")
        return 0

    total = 0
    for f in files:
        text = strip_markdown(f.read_text(encoding="utf-8"))
        rel = f.relative_to(REPO_ROOT) if f.is_absolute() else f
        for m in tool.check(text):
            if m.rule_id in disabled:
                continue
            flagged = text[m.offset : m.offset + m.error_length]
            if m.rule_id == "MORFOLOGIK_RULE_EN_US" and flagged.strip(".,;:") in ALLOWLIST:
                continue
            line, col = line_col_from_offset(text, m.offset)
            snippet = m.context.replace("\n", " ")
            print(f"{rel}:{line}:{col}: [{m.rule_id}] {m.message} — '{flagged}' in '{snippet}'")
            total += 1

    tool.close()
    print(f"\n{total} issue(s) across {len(files)} file(s)")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
