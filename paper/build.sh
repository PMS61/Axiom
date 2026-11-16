#!/usr/bin/env bash
# Build build/main.pdf (anonymous submission) and build/main_camera.pdf (camera-ready)
# for the paper (works from any working directory).
set -euo pipefail
cd "$(dirname "$0")"
# Make the EDM class (template_src/edm_article.cls) visible to TeX.
export TEXINPUTS="$PWD/template_src:${TEXINPUTS:-}"
mkdir -p build
for doc in main main_camera; do
  latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build "$doc.tex"
done
echo "PDF: $PWD/build/main.pdf"
echo "PDF: $PWD/build/main_camera.pdf"
