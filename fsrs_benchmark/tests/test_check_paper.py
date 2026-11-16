"""Tests for paper/scripts/check_paper.py: number matching, rounding, percents, tags, anonymity."""
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "paper" / "scripts" / "check_paper.py"
_spec = importlib.util.spec_from_file_location("check_paper", SCRIPT)
cp = importlib.util.module_from_spec(_spec)
sys.modules["check_paper"] = cp  # dataclasses looks the module up by name
_spec.loader.exec_module(cp)

FACTS_TEXT = """\
- Constant (per user, smoothed): log loss 0.2723, Brier 0.0722 (source: results/paper_duolingo/metrics.csv)
- Constant (global): log loss 0.4544 (source: results/paper_anki/metrics.csv)
- Constant (per user): log loss 0.3725 (source: x)
- HLR (Settles & Meeder): log loss 0.6805 (source: x)
- FSRS-6 (optimised) fallback to default weights, Duolingo: 115110 of 115222 users (source: x)
- Recall share of scored events (base rate): Anki 0.831; Duolingo 0.9198 (source: x)
- GBDT (LightGBM): lower log loss for 82 of 100 users, share 0.8200 (source: x)
- GBDT minus FSRS-6 (optimised), log loss: diff -0.0210, 95% CI [-0.0284, -0.0144] (source: x)
- Validation split: last 20 percent of TRAIN days (source: x)
"""


@pytest.fixture
def facts():
    return cp.parse_facts(FACTS_TEXT)


def _body(text, tail=("% FACTS: Constant floor log loss",)):
    """A body line followed by a tag comment."""
    return [text, *tail]


# ---- number extraction ---------------------------------------------------------------------

def test_extracts_decimal_thousands_percent_and_minus():
    nums = cp.find_numbers(cp.prose(r"a 0.2723, $-$0.0210, .358, 115,222, 74\% and 115222"))
    raws = [n.raw for n in nums]
    assert raws == ["0.2723", "-0.0210", ".358", "115,222", "74\\%", "115222"]


def test_small_integers_ignored_but_decimals_kept():
    nums = cp.find_numbers(cp.prose("5 users, 9 bins, 0.5 threshold, 10 cards"))
    assert [n.raw for n in nums] == ["0.5", "10"]


def test_latex_labels_refs_citations_and_sizes_ignored():
    text = (r"see Table~\ref{tab:12} and \cite{smith2019} with "
            r"\includegraphics[width=3.3in]{figures/f12}, \label{eq:55}")
    assert cp.find_numbers(cp.prose(text)) == []


def test_comment_lines_and_trailing_comments_not_counted():
    lines = ["% 0.9999 in a comment", "plain text % 0.8888 trailing", "\\% sign 0.5 kept"]
    facts = cp.parse_facts("")
    assert cp.check_body("t.tex", lines, facts) == [
        "t.tex:3: untagged number line (no % FACTS: or % PROTOCOL within 2 lines): \\% sign 0.5 kept",
        "t.tex:3: number 0.5 not in FACTS.md: \\% sign 0.5 kept",
    ]


def test_dashes_in_ranges_do_not_make_negative_numbers():
    nums = cp.find_numbers(cp.prose("bins 4-14d and 2--3 days"))
    assert [n.raw for n in nums] == ["14"]  # the range dash is not a minus sign


# ---- FACTS matching -------------------------------------------------------------------------

def test_exact_match(facts):
    body = _body("Log loss is 0.2723 here.")
    assert cp.check_body("t.tex", body, facts) == []


def test_mismatch_reported(facts):
    body = _body("Log loss is 0.2724 here.")
    problems = cp.check_body("t.tex", body, facts)
    assert problems == ["t.tex:1: number 0.2724 not in FACTS.md (checked only against FACTS lines that share a keyword with the tag): Log loss is 0.2724 here."]


def test_rounding_to_paper_decimals(facts):
    # FACTS 0.4544 rounds to 0.454 at three decimals, so the table form matches.
    assert cp.check_body("t.tex", _body("0.454 and 0.4544"), facts) == []
    problems = cp.check_body("t.tex", _body("0.455"), facts)
    assert len(problems) == 1 and "0.455" in problems[0]


def test_exact_tie_accepts_either_side(facts):
    # FACTS 0.6805 at three decimals is an exact tie: 0.680 (float-style) and 0.681 both pass.
    hlr = ("% FACTS: HLR log loss",)
    assert cp.check_body("t.tex", _body("0.680 and 0.681", hlr), facts) == []
    assert cp.check_body("t.tex", _body("0.682", hlr), facts)


def test_percent_matches_share_times_100(facts):
    # FACTS share 0.8200 -> 82 percent; 0.831 -> 83.1 percent.
    share = ("% FACTS: GBDT share of users",)
    assert cp.check_body("t.tex", _body(r"82\% of users and 83.1\%", share), facts) == []
    assert cp.check_body("t.tex", _body(r"84\% of users", share), facts)


def test_percent_sign_literal_and_percent_word(facts):
    # "95% CI" literal in FACTS; "20 percent" in FACTS matches the paper's 20\%.
    tail = ("% FACTS: GBDT CI validation split",)
    assert cp.check_body("t.tex", _body(r"95\% CI and 20\% of days", tail), facts) == []


def test_thousands_separator_matches_plain_integer(facts):
    tail = ("% FACTS: FSRS-6 fallback Duolingo",)
    assert cp.check_body("t.tex", _body("115,222 users, 115222 users", tail), facts) == []
    assert cp.check_body("t.tex", _body("115,223 users", tail), facts)


def test_dollar_minus_matches_negative_fact(facts):
    body = _body(r"the difference is $-$0.0210, with CI [$-$0.0284, $-$0.0144]", ("% FACTS: GBDT minus FSRS-6 CI",))
    assert cp.check_body("t.tex", body, facts) == []


# ---- tags -----------------------------------------------------------------------------------

def test_missing_tag_reported(facts):
    problems = cp.check_body("t.tex", ["Log loss is 0.2723.", "Next prose."], facts)
    assert problems == [
        "t.tex:1: untagged number line (no % FACTS: or % PROTOCOL within 2 lines): Log loss is 0.2723."
    ]


def test_tag_within_two_lines_counts_but_not_three(facts):
    ok = ["Log loss is 0.2723.", "more prose", "% FACTS: Constant floor log loss"]
    assert cp.check_body("t.tex", ok, facts) == []
    late = ["Log loss is 0.2723.", "more prose", "more prose", "% FACTS: Constant floor log loss"]
    assert any("untagged" in p for p in cp.check_body("t.tex", late, facts))


def test_protocol_tag_exempts_facts_match(facts):
    body = ["% PROTOCOL: EVALUATION.md 2. Not a result.", "We keep 1000 cards per user."]
    assert cp.check_body("t.tex", body, facts) == []


def test_tables_need_no_tag_but_must_match(facts):
    assert cp.check_table("t.tex", ["0.2723 & 0.3725"], facts) == []
    problems = cp.check_table("t.tex", ["0.9999 & 0.3725"], facts)
    assert problems == ["t.tex:1: table number 0.9999 not in FACTS.md"]


# ---- anonymity ------------------------------------------------------------------------------

def test_anonymity_hit_and_clean_text():
    assert cp.anon_hits("We thank Mumbai colleagues.", "t.tex")
    assert cp.anon_hits("A clean sentence about users.", "t.tex") == []


def test_anonymity_is_case_insensitive_and_word_bounded():
    assert cp.anon_hits("submitted by verma", "t.tex")
    assert cp.anon_hits("an axiomatic approach", "t.tex") == []


def test_allowed_urls_and_todo_markers_are_not_hits():
    text = (r"\url{https://github.com/open-spaced-repetition/fsrs-benchmark} "
            "https://huggingface.co/datasets/open-spaced-repetition/x TODO-SRSB check")
    assert cp.anon_hits(text, "t.tex") == []
    assert cp.anon_hits("see github.com/amalverma/repo", "t.tex")


def test_references_page_found():
    pages = ["Title\nAbstract", "Body text\nmore", "References\n[1] x"]
    assert cp.references_page(pages) == 3
    assert cp.references_page(["no heading here"]) is None


# ---- whole-paper runs on a synthetic tree --------------------------------------------------

def _make_paper(root: Path, main_abstract: str, section: str, table: str = "") -> Path:
    (root / "sections").mkdir(parents=True)
    (root / "tables").mkdir()
    (root / "FACTS.md").write_text(FACTS_TEXT, encoding="utf-8")
    (root / "main.tex").write_text(
        "\\documentclass{x}\n\\begin{document}\n\\begin{abstract}\n"
        f"{main_abstract}\n\\end{{abstract}}\n\\end{{document}}\n",
        encoding="utf-8",
    )
    (root / "sections" / "a.tex").write_text(section, encoding="utf-8")
    (root / "tables" / "t.tex").write_text(table, encoding="utf-8")
    (root / "references.bib").write_text("@misc{x, title={Y}}\n", encoding="utf-8")
    return root


def test_run_checks_clean_synthetic_paper(tmp_path):
    paper = _make_paper(
        tmp_path / "p",
        main_abstract="Log loss is 0.2723.\n% FACTS: Constant smoothed",
        section="Duolingo gives 0.3725.\n% FACTS: Constant user log loss\n",
        table="0.454 & 0.4544\n",
    )
    problems, notes = cp.run_checks(paper, None)
    assert problems == []
    assert notes == ["PDF checks skipped"]


def test_run_checks_reports_each_kind_of_problem(tmp_path):
    paper = _make_paper(
        tmp_path / "p",
        main_abstract="Log loss is 0.2723.\n% FACTS: Constant smoothed",
        section="Bad number 0.9999.\n% FACTS: Constant user log loss\nUntagged 0.3003.\nPlain.\nAuthor: Verma.\n",
        table="0.1234 & 0.3003\n",
    )
    problems, _ = cp.run_checks(paper, None)
    text = "\n".join(problems)
    assert "a.tex:1: number 0.9999 not in FACTS.md" in text
    assert "a.tex:3: untagged number line" in text
    assert "t.tex:1: table number 0.1234 not in FACTS.md" in text
    assert "anonymity term 'Verma'" in text


def test_main_exit_code_is_nonzero_on_problems(tmp_path):
    paper = _make_paper(tmp_path / "p", main_abstract="Plain abstract.", section="Plain.\n")
    assert cp.main(["--paper-dir", str(paper), "--pdf", str(tmp_path / "missing.pdf")]) == 1


# ---- tag scope, PROTOCOL placement, 2018 and URLs (review round 2) ----------------------------

def test_protocol_tag_after_a_number_line_does_not_exempt_it(facts):
    # The injected case: a PROTOCOL comment placed after the number must not exempt it.
    problems = cp.check_body("t.tex", ["Anki GBDT log loss is 0.4321.", "% PROTOCOL: test"], facts)
    assert any("untagged" in p for p in problems)
    assert any("0.4321" in p and "not in FACTS.md" in p for p in problems)


def test_protocol_tag_above_covers_only_its_own_line(facts):
    lines = ["% PROTOCOL: x", "Anki GBDT log loss is 0.4321.", "Anki GBDT log loss is 0.4322."]
    problems = cp.check_body("t.tex", lines, facts)
    assert problems and all(p.startswith("t.tex:3:") for p in problems)


def test_facts_tag_must_share_a_keyword_with_the_fact_line(facts):
    # 0.2723 is a Constant line; a tag about HLR must not accept it.
    problems = cp.check_body("t.tex", ["Log loss is 0.2723 here.", "% FACTS: HLR log loss"], facts)
    assert len(problems) == 1 and "share a keyword" in problems[0]


def test_keywords_ignore_generic_words():
    kw = cp.keywords_of("Anki GBDT log loss 0.2")
    assert {"anki", "gbdt"} <= kw and "log" not in kw and "loss" not in kw


def test_2018_is_a_problem_in_prose_but_not_inside_cite():
    facts = cp.parse_facts("")
    assert cp.check_body("t.tex", ["The SLAM 2018 shared task drew systems."], facts)
    assert cp.check_body("t.tex", ["The SLAM task \\cite{settles2018slam} drew systems."], facts) == []


def test_href_with_allowed_url_still_checks_link_text():
    url = "https://github.com/open-spaced-repetition/srs-benchmark"
    assert any("Axiom" in h for h in cp.anon_hits("\\href{" + url + "}{Axiom code}", "t.tex"))
    assert cp.anon_hits("\\url{" + url + "}", "t.tex") == []


def test_page_note_says_body_runs_to_the_references_page():
    note = cp.page_note(9, 9)
    assert "body text on pages 1-9" in note and "reference pages 9-9" in note
