#!/usr/bin/env python3
"""Automated checks for the paper (PLAN task 4.1). Run after paper/build.sh.

    /home/amal/fyp/Axiom/.venv/bin/python paper/scripts/check_paper.py

Checks:
  1. Numbers. Every number in the abstract (main.tex), sections/*.tex and tables/*.tex must
     match paper/FACTS.md after rounding to the paper number's decimals. A percent (74\\%)
     matches a FACTS share times 100, or a FACTS value written with a percent sign. Numbers on
     a line tagged % PROTOCOL are exempt from the FACTS match.
  2. Tags. Every body line (abstract and sections) that contains a number needs a % FACTS: or
     % PROTOCOL comment on one of the next two lines. Tables are exempt from tagging.
  3. Anonymity. Forbidden terms in the .tex/.bib source and in the PDF text.
  4. Pages. Total pages and the page where the References heading starts (reported only).

Exit status 0 when no problem is found, 1 otherwise. Reviewed exceptions are kept in
ALLOW_NUMBERS and ALLOW_ANON below; each needs a reason.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal
from pathlib import Path

PAPER_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PDF = PAPER_DIR / "build" / "main.pdf"

# ---- Reviewed exceptions (each entry needs a reason) -------------------------------------
# Number exceptions: raw number text as written in the source (e.g. "2024") -> reason.
ALLOW_NUMBERS: dict[str, str] = {
    "2018": "name of the SLAM 2018 shared task (a workshop year in prose, not a result)",
}
# Anonymity exceptions: term exactly as listed in ANON_TERMS -> reason.
ALLOW_ANON: dict[str, str] = {}

# ---- Anonymity -----------------------------------------------------------------------------
ANON_TERMS = [
    "VJTI", "Veermata", "Mumbai", "FYP", "final year", "Kaggle", "amalverma", "Axiom",
    "PMS61", "BayesianScheduler", "blackbook", "Verma", "Pagnis", "Sankhe", "Khandelwal",
]
# Third-party URLs and the TODO-SRSB markers are allowed; the rest of their line is still checked.
ALLOWED_URL_RE = re.compile(
    r"https?://(?:github\.com/open-spaced-repetition|huggingface\.co/datasets/open-spaced-repetition)\S*"
)
ALLOWED_MARKER_RE = re.compile(r"TODO-SRSB")

# ---- Numbers -------------------------------------------------------------------------------
# Optional sign (not preceded by a word character or dot, so "v1" and "4-14" parse correctly),
# integer with optional thousands separators or a decimal (or .358), optional percent.
NUM_RE = re.compile(r"(?<![\w.])(-?)((?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)(\\?%)?")

# LaTeX that holds no prose numbers: labels, refs, citations, inputs, figure sizes, rules.
_PROSE_STRIP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\\cmidrule(?:\([^)]*\))?\{[^}]*\}"), " "),
    (re.compile(r"\\cline\{[^}]*\}"), " "),
    (re.compile(r"\\(?:eq|c|auto|page)?ref\{[^}]*\}"), " "),
    (re.compile(r"\\label\{[^}]*\}"), " "),
    (re.compile(r"\\cite\w*(?:\[[^\]]*\])*\{[^}]*\}"), " "),
    (re.compile(r"\\(?:input|include|bibliography|bibliographystyle|url|href|includegraphics"
                r"|begin|end)(?:\[[^\]]*\])*\{[^}]*\}"), " "),
    (re.compile(r"\\(?:setlength|hspace|vspace|addtolength)\*?(?:\{[^}]*\}){1,2}"), " "),
    (re.compile(r"(\\(?:sub)*section\*?\{)\s*\d+(?:\.\d+)*\s+"), r"\1"),  # section numbers
    (re.compile(r"\d*\.?\d+(?:in|cm|mm|pt|em|ex|bp)\b"), " "),              # figure sizes
    (re.compile(r"\d*\.?\d+\s*\\(?:linewidth|textwidth|columnwidth|paperwidth)"), " "),
    (re.compile(r"\\[A-Za-z]+\*?"), " "),                                    # command names
]


@dataclass(frozen=True)
class Num:
    raw: str
    value: Decimal
    decimals: int
    percent: bool


def split_comment(line: str) -> tuple[str, str | None]:
    """Split a line into (body, comment). An escaped \\% is not a comment."""
    for m in re.finditer(r"%", line):
        before = line[: m.start()]
        backslashes = len(before) - len(before.rstrip("\\"))
        if backslashes % 2 == 0:
            return before, line[m.start():]
    return line, None


def prose(text: str) -> str:
    """Return the text with LaTeX that carries no prose numbers removed."""
    s = text.replace("$-$", "-").replace("\\textminus", "-").replace("\u2212", "-")
    s = s.replace("--", "-")
    for rx, rep in _PROSE_STRIP:
        s = rx.sub(rep, s)
    return s


def find_numbers(text: str, paper: bool = True) -> list[Num]:
    """All numbers in text. For the paper, integers below 10 are ignored (PLAN 4.1)."""
    out: list[Num] = []
    for m in NUM_RE.finditer(text):
        sign, body, pct = m.group(1), m.group(2), m.group(3)
        digits = body.replace(",", "")
        value = Decimal(digits)
        if sign:
            value = -value
        decimals = len(digits.split(".", 1)[1]) if "." in digits else 0
        num = Num(raw=m.group(0), value=value, decimals=decimals, percent=bool(pct))
        if paper and not num.percent and decimals == 0 and abs(value) < 10:
            continue
        out.append(num)
    return out


PERCENT_WORD_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?) percent\b")


def parse_facts(text: str) -> list[Num]:
    """Every number in FACTS.md, with no size filter. "20 percent" counts as a percent."""
    out: list[Num] = []
    for line in text.splitlines():
        out.extend(find_numbers(line, paper=False))
        for m in PERCENT_WORD_RE.finditer(line):
            digits = m.group(1)
            decimals = len(digits.split(".", 1)[1]) if "." in digits else 0
            out.append(Num(raw=m.group(0), value=Decimal(digits), decimals=decimals, percent=True))
    return out


def _round_to(value: Decimal, decimals: int) -> Decimal:
    return value.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)


def _round_options(value: Decimal, decimals: int) -> set[Decimal]:
    """Rounded value at `decimals`. An exact tie (e.g. 0.6805 at 3 decimals) may be written
    either way, because the generated tables round floats, so both sides are accepted."""
    options = {_round_to(value, decimals)}
    scaled = value.scaleb(decimals + 1)
    if scaled == scaled.to_integral_value() and int(abs(scaled)) % 10 == 5:
        options.add(value.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_DOWN))
    return options


def matches_facts(num: Num, facts: list[Num]) -> bool:
    """True if num equals a FACTS number after rounding FACTS to num's decimals."""
    if num.percent:
        candidates = [f.value for f in facts if f.percent]
        candidates += [f.value * 100 for f in facts if abs(f.value) <= 1]
    else:
        candidates = [f.value for f in facts]
    return any(num.value in _round_options(c, num.decimals) for c in candidates)


def tag_kind(line: str) -> str | None:
    """'FACTS' or 'PROTOCOL' if the line is a comment carrying that tag, else None."""
    s = line.strip()
    if not s.startswith("%"):
        return None
    m = re.match(r"%\s*(FACTS:|PROTOCOL)", s)
    if not m:
        return None
    return "PROTOCOL" if m.group(1) == "PROTOCOL" else "FACTS"


def nearest_tag(lines: list[str], i: int) -> str | None:
    """Tag on the next two lines after line i (or on line i itself, as a trailing comment)."""
    _, comment = split_comment(lines[i])
    if comment is not None and tag_kind(comment):
        return tag_kind(comment)
    for j in (i + 1, i + 2):
        if j < len(lines):
            kind = tag_kind(lines[j])
            if kind:
                return kind
    return None


def _snip(text: str) -> str:
    return " ".join(text.split())[:100]


def check_body(name: str, lines: list[str], facts: list[Num],
               start: int = 0, end: int | None = None) -> list[str]:
    """Number and tag checks for prose lines lines[start:end] (0-based indices)."""
    problems: list[str] = []
    end = len(lines) if end is None else end
    for i in range(start, end):
        if lines[i].strip().startswith("%"):
            continue
        body, _ = split_comment(lines[i])
        nums = find_numbers(prose(body))
        if not nums:
            continue
        tag = nearest_tag(lines, i)
        if tag is None:
            problems.append(f"{name}:{i + 1}: untagged number line (no % FACTS: or % PROTOCOL "
                            f"within 2 lines): {_snip(body)}")
        for n in nums:
            if tag == "PROTOCOL" or n.raw in ALLOW_NUMBERS:
                continue
            if not matches_facts(n, facts):
                problems.append(f"{name}:{i + 1}: number {n.raw} not in FACTS.md: {_snip(body)}")
    return problems


def check_table(name: str, lines: list[str], facts: list[Num]) -> list[str]:
    """Tables are not tagged, but every number must match FACTS.md."""
    problems: list[str] = []
    for i, line in enumerate(lines):
        if line.strip().startswith("%"):
            continue
        body, _ = split_comment(line)
        for n in find_numbers(prose(body)):
            if n.raw in ALLOW_NUMBERS:
                continue
            if not matches_facts(n, facts):
                problems.append(f"{name}:{i + 1}: table number {n.raw} not in FACTS.md")
    return problems


def anon_hits(text: str, where: str) -> list[str]:
    """Forbidden terms in text. where labels the source (file, or PDF page)."""
    cleaned = ALLOWED_URL_RE.sub(" ", text)
    cleaned = ALLOWED_MARKER_RE.sub(" ", cleaned)
    hits: list[str] = []
    for no, line in enumerate(cleaned.splitlines(), 1):
        for term in ANON_TERMS:
            if term in ALLOW_ANON:
                continue
            if re.search(r"\b" + re.escape(term) + r"\b", line, re.IGNORECASE):
                hits.append(f"{where}:{no}: anonymity term '{term}': {_snip(line)}")
    return hits


def _run(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def pdf_page_texts(pdf: Path) -> list[str]:
    """Text of each PDF page, via pdftotext -f/-l."""
    info = _run(["pdfinfo", str(pdf)])
    n = int(re.search(r"^Pages:\s+(\d+)", info, re.MULTILINE).group(1))
    return [_run(["pdftotext", "-f", str(p), "-l", str(p), str(pdf), "-"])
            for p in range(1, n + 1)]


def references_page(pages: list[str]) -> int | None:
    """1-based page on which a References (or Bibliography) heading line appears."""
    for idx, text in enumerate(pages, 1):
        for line in text.splitlines():
            if re.fullmatch(r"\s*(references|bibliography)\s*", line, re.IGNORECASE):
                return idx
    return None


def run_checks(paper_dir: Path, pdf: Path | None) -> tuple[list[str], list[str]]:
    """Return (problems, notes). pdf=None skips the PDF checks."""
    problems: list[str] = []
    notes: list[str] = []
    facts = parse_facts((paper_dir / "FACTS.md").read_text(encoding="utf-8"))

    def rel(p: Path) -> str:
        return str(p.relative_to(paper_dir))

    main_tex = paper_dir / "main.tex"
    main_lines = main_tex.read_text(encoding="utf-8").splitlines()
    begin = next((i for i, l in enumerate(main_lines) if "\\begin{abstract}" in l), None)
    end = next((i for i, l in enumerate(main_lines) if "\\end{abstract}" in l), None)
    if begin is None or end is None:
        problems.append("main.tex: abstract not found")
    else:
        problems += check_body("main.tex", main_lines, facts, begin + 1, end)

    for f in sorted((paper_dir / "sections").glob("*.tex")):
        problems += check_body(rel(f), f.read_text(encoding="utf-8").splitlines(), facts)
    for f in sorted((paper_dir / "tables").glob("*.tex")):
        problems += check_table(rel(f), f.read_text(encoding="utf-8").splitlines(), facts)

    sources = set(paper_dir.glob("*.tex")) | set(paper_dir.glob("*.bib"))
    sources |= set((paper_dir / "sections").glob("*.tex")) | set((paper_dir / "tables").glob("*.tex"))
    for f in sorted(sources):
        problems += anon_hits(f.read_text(encoding="utf-8"), rel(f))

    if pdf is None:
        notes.append("PDF checks skipped")
    elif not pdf.exists():
        problems.append(f"PDF not found: {pdf} (run paper/build.sh)")
    else:
        pages = pdf_page_texts(pdf)
        for p, text in enumerate(pages, 1):
            problems += anon_hits(text, f"PDF p{p}")
        ref = references_page(pages)
        if ref is None:
            problems.append("PDF: References heading not found")
            notes.append(f"pages total {len(pages)}")
        else:
            notes.append(f"pages total {len(pages)}; body pages 1-{ref - 1}; "
                         f"References start on page {ref}; reference pages {ref}-{len(pages)}")
    return problems, notes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--paper-dir", type=Path, default=PAPER_DIR)
    ap.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    args = ap.parse_args(argv)
    problems, notes = run_checks(args.paper_dir, args.pdf)
    for note in notes:
        print(f"note: {note}")
    for p in problems:
        print(f"PROBLEM: {p}")
    print(f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
