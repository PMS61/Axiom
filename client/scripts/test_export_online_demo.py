"""Tests for export_online_demo.py. Run from the repo root:

  .venv/bin/python -m pytest -q client/scripts/test_export_online_demo.py
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_online_demo as ex  # noqa: E402


@pytest.fixture(scope="module")
def seed():
    return ex.build_online_seed()


def test_top_level_keys(seed):
    assert set(seed) == {"final", "perUser", "winsVsFsrs6", "winsVsGbdt", "replayCheck", "cards", "fsrsNote"}
    assert seed["fsrsNote"] == ex.FSRS_NOTE


def test_final_table_headline(seed):
    table = seed["final"]["table"]
    assert table["Online layer (ours, final)"]["weighted"] == pytest.approx(0.29991, abs=1e-5)
    assert table["FSRS-6"]["weighted"] == pytest.approx(0.33832, abs=1e-5)
    assert "bootstrap" in seed["final"]


def test_per_user(seed):
    rows = seed["perUser"]
    assert len(rows) == 1000
    assert len({r["user"] for r in rows}) == 1000
    assert all(set(r) == {"user", "size", "ours", "gbdt", "fsrs6"} for r in rows)
    assert seed["winsVsFsrs6"] == sum(1 for r in rows if r["ours"] < r["fsrs6"])
    assert seed["winsVsGbdt"] == sum(1 for r in rows if r["ours"] < r["gbdt"])


def test_replay_matches_recorded_user_logloss(seed):
    recorded = {}
    with open(ex.ONLINE_JSONL, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            recorded[int(r["user"])] = r["metrics"]["LogLoss"]
    checks = seed["replayCheck"]
    assert sorted(c["user"] for c in checks) == sorted(ex.REPLAY_USERS)
    for c in checks:
        assert abs(c["logLoss"] - recorded[c["user"]]) <= 1e-4
        assert c["logLoss"] == pytest.approx(c["recordedLogLoss"], abs=1e-4)


def test_cards(seed):
    cards = seed["cards"]
    assert [c["label"] for c in cards] == ["Card A", "Card B", "Card C", "Card D"]
    assert [c["pickedFor"] for c in cards] == ["top", "top", "top", "fsrs_wins"]
    for card in cards:
        scored = [r for r in card["reviews"] if r["scored"]]
        assert len(scored) == card["nScored"] >= 8
        assert {r["y"] for r in scored} == {0, 1}
        for r in scored:
            for key in ("pOnline", "pGbdt", "pFsrsDefault"):
                assert 0 < r[key] < 1
        for r in card["reviews"]:
            assert set(r) >= {"review", "day", "rating", "y", "scored"}
        assert all(not r["scored"] or "pOnline" in r for r in card["reviews"])


def test_written_file_is_small_and_strict_json(tmp_path):
    out = ex.main(tmp_path / "online.json")
    assert out.stat().st_size < 400 * 1024
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text
    data = json.loads(text, parse_constant=lambda c: pytest.fail(f"non-strict JSON constant {c}"))
    assert "cards" in data and "perUser" in data


def _jsonl_by_user(path):
    out = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                assert int(r["user"]) not in out, "duplicate user"
                out[int(r["user"])] = r
    return out


def test_fsrs6_rows_match_users_and_sizes(seed):
    fsrs = _jsonl_by_user(ex.FSRS_JSONL)
    online = _jsonl_by_user(ex.ONLINE_JSONL)
    gbdt = _jsonl_by_user(ex.GBDT_JSONL)
    for row in seed["perUser"]:
        u = row["user"]
        assert u in fsrs and fsrs[u]["size"] == online[u]["size"] == gbdt[u]["size"] == row["size"]
        assert row["fsrs6"] == pytest.approx(fsrs[u]["metrics"]["LogLoss"], abs=1e-6)
        assert row["ours"] == pytest.approx(online[u]["metrics"]["LogLoss"], abs=1e-6)
        assert row["gbdt"] == pytest.approx(gbdt[u]["metrics"]["LogLoss"], abs=1e-6)


def test_wins_counts_recomputed_from_files(seed):
    rows = seed["perUser"]
    assert seed["winsVsFsrs6"] == sum(r["ours"] < r["fsrs6"] for r in rows)
    assert seed["winsVsGbdt"] == sum(r["ours"] < r["gbdt"] for r in rows)


def test_cards_are_time_ordered_and_flagged(seed):
    for card in seed["cards"]:
        reviews = card["reviews"]
        assert [r["review"] for r in reviews] == list(range(1, len(reviews) + 1))
        assert all(a["day"] <= b["day"] for a, b in zip(reviews, reviews[1:]))
        assert all(("pFsrsDefault" in r) == r["scored"] for r in reviews)


def test_no_card_or_review_ids_in_output(tmp_path):
    text = ex.main(tmp_path / "online.json").read_text(encoding="utf-8")
    for token in ('"card_id"', '"review_th"', '"event_id"', '"cardId"'):
        assert token not in text
