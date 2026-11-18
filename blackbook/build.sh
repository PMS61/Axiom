#!/usr/bin/env bash
# Build main.pdf from the blackbook directory (works from any working directory).
set -euo pipefail
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
