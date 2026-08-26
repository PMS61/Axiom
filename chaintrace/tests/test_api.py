import os
import tempfile

# Point the case database at a throwaway file before the app imports config.
os.environ.setdefault("CHAINTRACE_DB", os.path.join(tempfile.mkdtemp(), "test_cases.db"))

from fastapi.testclient import TestClient  # noqa: E402

from app.api import app  # noqa: E402

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_trace_endpoint():
    response = client.post(
        "/trace", json={"address": "SUSPECT_WALLET_DEMO_1", "allow_network": False}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "attributed"
    assert body["attributed_entity"] == "SampleExchange Alpha"


def test_trace_requires_an_address():
    assert client.post("/trace", json={"address": "   "}).status_code == 400


def test_complaint_runs_the_same_engine_and_opens_a_case():
    response = client.post(
        "/complaint",
        json={
            "wallet": "SUSPECT_WALLET_DEMO_1",
            "complaint_ref": "REF-1",
            "amount_reported": 46000,
            "allow_network": False,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["case_id"].startswith("CT-")
    assert body["trace"]["attributed_entity"] == "SampleExchange Alpha"
    assert body["time_to_result_ms"] > 0

    listed = client.get("/cases").json()
    assert any(case["case_id"] == body["case_id"] for case in listed["cases"])

    fetched = client.get(f"/cases/{body['case_id']}").json()
    assert fetched["metadata"]["complaint_ref"] == "REF-1"
    assert fetched["source"] == "complaint"


def test_missing_case_is_404():
    assert client.get("/cases/CT-NOPE").status_code == 404
