# Paper Phase 3 report: writing (2026-11-17)

## What was built
- `paper/`: the official EDM template (`template_src/edm_article.cls`, from the EDM 2024 template zip linked on the EDM 2026 author page), `main.tex`, 8 section files, `references.bib` (14 entries) and `build.sh`.
- Generated tables and figures, which are never typed by hand:
  - `paper/scripts/make_tables.py` writes `tables/{overall,paired,seeds,srsb}.tex`.
  - `make_figures.py` writes the gap-bin and reliability figures. Reliability bins with fewer than 30 reviews are omitted.
- Every number in the prose carries a `% FACTS:` or `% PROTOCOL` tag. `paper/scripts/check_paper.py` (task 4.1) enforces this. It also checks anonymity in the source and the PDF text, and reports page counts.

## Review
Each task went through its own verify loop. The main problems caught and fixed:
- Unescaped `%` signs deleted three CIs.
- The draft said v1 does not converge on Duolingo only; it does not converge on Anki either.
- It said v2 and v3 lie "between the constant and GBDT" on Duolingo; in fact they are worse than the constant.
- Calibration was overclaimed on Anki.
- A lexeme-effect number (0.0003) came from the leaky sweep. The correct value is 0.0012, and phase-7-experiments.md now has a correction note.
- Wrong author names in two bib entries.
- The abstract ran over 200 words.
- The checker could be fooled by PROTOCOL tag windows and by `\href` labels. Both holes are fixed and tested.

## State
- The PDF is 9 pages, and the body ends on page 9 together with the references.
- The abstract is 197 words.
- check_paper.py exits 0.
- test_check_paper.py and test_paper_tables.py pass (36 tests).

## Known limits
- The EDM 2027 page limit is not yet published.
- intro.tex holds a `% TODO: release link` comment for the camera-ready version.
- For the remaining LOW items, see BACKLOG.md.
