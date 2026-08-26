"""PDF report generation with reportlab.

One page (or two) an investigator can attach to a file: what was traced, where
it went, how confident the engine is and why, plus the flags raised.
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .scoring import confidence_band
from .trace import TraceResult
from .viz import format_amount, short

STATUS_TEXT = {
    "attributed": "Cash-out point identified",
    "trail_broken": "Trail broken at a mixing service",
    "depth_limit": "Hop limit reached - funds still moving",
    "dead_end": "Dead end - funds have not moved on",
    "no_activity": "No outgoing activity found",
}


def _table(rows: list[list[str]], widths: list[float]) -> Table:
    table = Table(rows, colWidths=widths, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def build_pdf(
    result: TraceResult,
    case_id: str | None = None,
    risk: Any | None = None,
    metadata: dict[str, Any] | None = None,
) -> bytes:
    """Render a trace result as a PDF and return the bytes."""
    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=16 * mm,
        bottomMargin=16 * mm,
        title=f"ChainTrace report {case_id or result.address}",
    )

    styles = getSampleStyleSheet()
    small = ParagraphStyle("small", parent=styles["Normal"], fontSize=8, leading=11)
    heading = ParagraphStyle(
        "heading", parent=styles["Heading2"], fontSize=11, spaceBefore=10, spaceAfter=4
    )
    story: list[Any] = []

    story.append(Paragraph("ChainTrace - wallet trace report", styles["Title"]))
    story.append(
        Paragraph(
            f"Generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
            + (f" &middot; Case {case_id}" if case_id else ""),
            small,
        )
    )
    story.append(Spacer(1, 8))

    # --- summary -------------------------------------------------------------
    summary = [
        ["Field", "Value"],
        ["Suspect wallet", result.address],
        ["Chain", result.chain],
        ["Outcome", STATUS_TEXT.get(result.status, result.status)],
        ["Attributed exchange", result.attributed_entity or "-"],
        ["Deposit address", result.attributed_address or "-"],
        [
            "Confidence",
            f"{result.confidence:.0f}% ({confidence_band(result.confidence)})"
            if result.confidence
            else "-",
        ],
        ["Hops", str(result.hops) if result.hops is not None else "-"],
        ["Hop limit", str(result.max_depth)],
        ["Data sources", ", ".join(result.data_sources) or "none"],
    ]
    if risk is not None:
        summary.append(["Illicit likelihood (ML)", f"{risk.percent:.0f}%"])
    for key, value in (metadata or {}).items():
        summary.append([str(key).replace("_", " ").title(), str(value)])
    story.append(_table(summary, [45 * mm, 120 * mm]))

    # --- the path ------------------------------------------------------------
    best = result.best_path
    if best:
        story.append(Paragraph("Fund flow", heading))
        rows = [["Hop", "From", "To", "Amount", "Transaction", "Timestamp"]]
        for index, edge in enumerate(best.edges, start=1):
            rows.append(
                [
                    str(index),
                    short(edge.src, 12, 4),
                    short(edge.dst, 12, 4),
                    format_amount(edge.amount, edge.chain),
                    short(edge.txid, 12, 4),
                    edge.timestamp or "-",
                ]
            )
        story.append(_table(rows, [10 * mm, 33 * mm, 33 * mm, 26 * mm, 30 * mm, 33 * mm]))

        story.append(Paragraph("Confidence factors", heading))
        factor_rows = [["Factor", "Effect", "Why"]]
        factor_rows += [
            [f["factor"], f"{f['delta']:+.0f}", Paragraph(f["detail"], small)]
            for f in best.confidence_factors
        ]
        story.append(_table(factor_rows, [45 * mm, 16 * mm, 104 * mm]))

    # --- flags ---------------------------------------------------------------
    flags = [flag for path in result.paths for flag in path.flags]
    if flags:
        story.append(Paragraph("Typology flags", heading))
        flag_rows = [["Severity", "Flag", "Address", "Detail"]]
        seen = set()
        for flag in flags:
            key = (flag["code"], flag.get("address", ""))
            if key in seen:
                continue
            seen.add(key)
            flag_rows.append(
                [
                    flag["severity"].upper(),
                    flag["title"],
                    short(flag.get("address", ""), 10, 4),
                    Paragraph(flag["detail"], small),
                ]
            )
        story.append(_table(flag_rows, [16 * mm, 32 * mm, 30 * mm, 87 * mm]))

    # --- other branches ------------------------------------------------------
    if len(result.paths) > 1:
        story.append(Paragraph("Other branches followed", heading))
        branch_rows = [["Terminal", "Type", "Hops", "Confidence"]]
        for path in result.paths[1:]:
            branch_rows.append(
                [
                    path.entity or short(path.terminal, 14, 4),
                    path.terminal_kind.replace("_", " "),
                    str(path.hops),
                    f"{path.confidence:.0f}%",
                ]
            )
        story.append(_table(branch_rows, [70 * mm, 35 * mm, 20 * mm, 30 * mm]))

    story.append(Spacer(1, 10))
    story.append(
        Paragraph(
            "Proof of concept. Addresses, labels and transactions in this report come "
            "from bundled sample data unless a live data source is listed above. "
            "Attribution and scores are investigative leads, not evidence.",
            small,
        )
    )

    document.build(story)
    return buffer.getvalue()
