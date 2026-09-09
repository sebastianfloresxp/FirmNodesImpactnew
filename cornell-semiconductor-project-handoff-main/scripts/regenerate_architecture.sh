#!/usr/bin/env bash
# Regenerate architecture graphs for docs/architecture/
# Requires: Python >=3.10
# HTML output also requires graphviz (apt: sudo apt-get install graphviz)
# Usage: bash scripts/regenerate_architecture.sh
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

# Create/activate venv if needed
if [ ! -d .venv-arch ]; then
  echo "Creating venv with Python 3.13..."
  uv venv --python 3.13 .venv-arch 2>/dev/null || python3.13 -m venv .venv-arch
fi
source .venv-arch/bin/activate

# Install pyan3 from GitHub (requires >=3.10)
if ! command -v pyan3 &>/dev/null; then
  echo "Installing pyan3..."
  uv pip install git+https://github.com/Technologicat/pyan.git 2>/dev/null || \
    pip install git+https://github.com/Technologicat/pyan.git
fi

# Check for graphviz — needed for HTML output
if command -v dot &>/dev/null; then
  HAS_DOT=true
else
  HAS_DOT=false
  echo "graphviz not found — generating .dot files only (install with: sudo apt-get install graphviz)"
fi

OUTDIR=docs/architecture
ABSROOT="$(pwd)/"

# Generate .dot always; .html only when graphviz is available.
# Usage: find ... | pyan_graph <base_output_path> [extra pyan3 flags...]
pyan_graph() {
  local base="$1"; shift
  local files
  files=$(cat)
  [ -z "$files" ] && return 0
  echo "$files" | xargs pyan3 "$@" --dot --file "${base}.dot" 2>/dev/null
  if $HAS_DOT; then
    echo "$files" | xargs pyan3 "$@" --html --file "${base}.html" 2>/dev/null
  fi
}

echo "=== Module-level overview ==="
find src/ -name "*.py" -not -name "__init__.py" -not -name "00_*_hpo_*" | \
  pyan_graph "$OUTDIR/module_dependencies" --module-level --uses --no-defines --colored --grouped

echo "=== Per-module call graphs ==="
for mod in tgnn n2v_temporal graphsage node2vec twotower ensemble heuristics ch3; do
  echo -n "  $mod... "
  find "src/${mod}/" -name "*.py" -not -name "__init__.py" -not -name "00_*_hpo_*" 2>/dev/null | \
    pyan_graph "$OUTDIR/calls_${mod}" --uses --no-defines --colored --grouped --annotated
  echo "done"
done

echo "=== Analysis graphs ==="
for submod in chapter2 chapter4; do
  echo -n "  analysis/$submod... "
  find "src/analysis/${submod}/" -name "*.py" -not -name "__init__.py" 2>/dev/null | \
    pyan_graph "$OUTDIR/calls_analysis_${submod}" --uses --no-defines --colored --grouped --annotated
  echo "done"
done

echo "=== Data processing ==="
find src/data_processing/ -name "*.py" -not -name "__init__.py" | \
  pyan_graph "$OUTDIR/calls_data_processing" --uses --no-defines --colored --grouped --annotated

echo "=== Scoped subgraphs ==="
# Ch4 M0-M3
find src/analysis/chapter4/modules/ -name "m[0-3]_*.py" | \
  pyan_graph "$OUTDIR/calls_ch4_M0_M3" --uses --no-defines --colored --grouped --annotated
# Ch4 M4-M6
find src/analysis/chapter4/modules/ -name "m[4-6]_*.py" | \
  pyan_graph "$OUTDIR/calls_ch4_M4_M6" --uses --no-defines --colored --grouped --annotated
# Ch4 M7-M9
find src/analysis/chapter4/modules/ -name "m[7-9]_*.py" | \
  pyan_graph "$OUTDIR/calls_ch4_M7_M9" --uses --no-defines --colored --grouped --annotated
# Ch2 figures
find src/analysis/chapter2/ -name "fig_*.py" -o -name "scorecard_common.py" | \
  pyan_graph "$OUTDIR/calls_ch2_figures" --uses --no-defines --colored --grouped --annotated
# Ch2 tables
find src/analysis/chapter2/ -name "tab_*.py" -o -name "scorecard_common.py" -o -name "build_*.py" | \
  pyan_graph "$OUTDIR/calls_ch2_tables" --uses --no-defines --colored --grouped --annotated

echo "=== Stripping absolute paths ==="
for f in "$OUTDIR"/*.dot "$OUTDIR"/*.html; do
  [ -f "$f" ] || continue
  sed -i "s|${ABSROOT}||g" "$f"
  HTML_ROOT=$(echo "$ABSROOT" | sed 's/-/\&#45;/g')
  sed -i "s|${HTML_ROOT}||g" "$f"
done

remaining=$(grep -rl "/home/\|/Users/" "$OUTDIR"/*.dot 2>/dev/null | wc -l || true)
echo "Absolute paths remaining: $remaining"

dot_count=$(ls "$OUTDIR"/*.dot 2>/dev/null | wc -l || true)
html_count=$(ls "$OUTDIR"/*.html 2>/dev/null | wc -l || true)
echo ""
if $HAS_DOT; then
  echo "Done. Generated ${dot_count} .dot + ${html_count} .html files."
else
  echo "Done. Generated ${dot_count} .dot files (no .html — graphviz not installed)."
fi
