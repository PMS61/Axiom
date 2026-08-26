"""FastAPI backend - the second entry point into the same trace engine.

An investigator uses the Streamlit dashboard; a fraud complaint arrives here
over HTTP. Both call `app.trace.trace()`, so there is exactly one engine.

Run:  uvicorn app.api:app --reload --port 8000
"""

from __future__ import annotations

import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import DEFAULT_MAX_DEPTH
from .ml import score_wallet
from .storage import case_from_trace, get_case_store
from .trace import trace

app = FastAPI(
    title="ChainTrace API",
    version="0.1.0",
    description="Trace a suspect wallet to its cash-out point.",
)


class TraceRequest(BaseModel):
    address: str = Field(..., description="Suspect wallet address")
    max_depth: int = Field(DEFAULT_MAX_DEPTH, ge=1, le=10)
    allow_network: bool = Field(True, description="False = bundled sample data only")


class ComplaintRequest(BaseModel):
    """A fraud complaint arriving from an intake system."""

    wallet: str = Field(..., description="Wallet address reported by the victim")
    complaint_ref: str | None = Field(None, description="Reference number from the feed")
    complainant: str | None = None
    amount_reported: float | None = None
    description: str | None = None
    max_depth: int = Field(DEFAULT_MAX_DEPTH, ge=1, le=10)
    allow_network: bool = True


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "chaintrace", "version": app.version}


@app.post("/trace")
def run_trace(request: TraceRequest) -> dict[str, Any]:
    """Trace a wallet and return the result. Does not open a case."""
    if not request.address.strip():
        raise HTTPException(status_code=400, detail="address is required")
    result = trace(
        request.address,
        max_depth=request.max_depth,
        allow_network=request.allow_network,
    )
    return result.to_dict()


@app.post("/complaint")
def intake_complaint(request: ComplaintRequest) -> dict[str, Any]:
    """Real-time mode: a complaint comes in, the same engine runs, a case opens.

    The response carries `time_to_result_ms` - the whole point of the demo is
    that this number is small enough to act on while the money is still moving.
    """
    if not request.wallet.strip():
        raise HTTPException(status_code=400, detail="wallet is required")

    started = time.perf_counter()
    result = trace(
        request.wallet,
        max_depth=request.max_depth,
        allow_network=request.allow_network,
    )
    risk = score_wallet(request.wallet, allow_network=request.allow_network)

    metadata = {
        key: value
        for key, value in {
            "complaint_ref": request.complaint_ref,
            "complainant": request.complainant,
            "amount_reported": request.amount_reported,
            "description": request.description,
        }.items()
        if value is not None
    }
    case = get_case_store().save(case_from_trace(result, source="complaint", metadata=metadata))
    elapsed_ms = (time.perf_counter() - started) * 1000

    return {
        "case_id": case.case_id,
        "received": metadata,
        "time_to_result_ms": round(elapsed_ms, 1),
        "illicit_likelihood": risk.percent,
        "trace": result.to_dict(),
    }


@app.get("/cases")
def list_cases(limit: int = 25) -> dict[str, Any]:
    cases = get_case_store().list(limit=limit)
    return {
        "count": len(cases),
        "cases": [
            {
                "case_id": case.case_id,
                "created_at": case.created_at,
                "address": case.address,
                "status": case.status,
                "entity": case.entity,
                "confidence": case.confidence,
                "hops": case.hops,
                "source": case.source,
                "metadata": case.metadata,
            }
            for case in cases
        ],
    }


@app.get("/cases/{case_id}")
def get_case(case_id: str) -> dict[str, Any]:
    case = get_case_store().get(case_id)
    if case is None:
        raise HTTPException(status_code=404, detail=f"case {case_id} not found")
    return case.to_dict()
