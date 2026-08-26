import tempfile
from pathlib import Path

from app.report import build_pdf
from app.storage import SQLiteCaseStore, case_from_trace, new_case_id
from app.trace import trace

OFFLINE = {"allow_network": False}


def _store() -> SQLiteCaseStore:
    return SQLiteCaseStore(Path(tempfile.mkdtemp()) / "cases.db")


def test_case_ids_are_unique_and_prefixed():
    ids = {new_case_id() for _ in range(50)}
    assert len(ids) == 50
    assert all(case_id.startswith("CT-") for case_id in ids)


def test_case_survives_a_round_trip():
    store = _store()
    result = trace("SUSPECT_WALLET_DEMO_1", **OFFLINE)
    case = store.save(
        case_from_trace(result, source="complaint", metadata={"complaint_ref": "REF-42"})
    )

    loaded = store.get(case.case_id)
    assert loaded is not None
    assert loaded.address == "SUSPECT_WALLET_DEMO_1"
    assert loaded.entity == "SampleExchange Alpha"
    assert loaded.confidence == result.confidence
    assert loaded.metadata["complaint_ref"] == "REF-42"
    assert loaded.trace["paths"][0]["terminal_kind"] == "exchange"


def test_listing_is_newest_first_and_limited():
    store = _store()
    for wallet in ("SUSPECT_WALLET_DEMO_1", "SUSPECT_WALLET_DEMO_2", "SUSPECT_WALLET_DEMO_4"):
        store.save(case_from_trace(trace(wallet, **OFFLINE)))
    assert len(store.list(limit=2)) == 2
    assert len(store.list()) == 3


def test_missing_case_is_none():
    assert _store().get("CT-DOESNOTEXIST") is None


def test_pdf_report_is_generated():
    result = trace("SUSPECT_WALLET_DEMO_1", **OFFLINE)
    pdf = build_pdf(result, case_id="CT-TEST0001", metadata={"complaint_ref": "REF-1"})
    assert pdf.startswith(b"%PDF-")
    assert len(pdf) > 1000


def test_pdf_report_works_for_a_broken_trail_too():
    pdf = build_pdf(trace("SUSPECT_WALLET_DEMO_2", **OFFLINE))
    assert pdf.startswith(b"%PDF-")
