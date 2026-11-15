#!/usr/bin/env bash
# Build build/main.pdf for the paper (works from any working directory).
set -euo pipefail
cd "$(dirname "$0")"
# Make the EDM class (template_src/edm_article.cls) visible to TeX.
export TEXINPUTS="$PWD/template_src:${TEXINPUTS:-}"
mkdir -p build
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex
echo "PDF: $PWD/build/main.pdf"
