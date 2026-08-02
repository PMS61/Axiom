import React from "react";
import DemoNotice from "@/components/DemoNotice";
import Header from "@/components/Header";
import { slotToTime } from "@/lib/engine";
import { getUserMatrix } from "@/lib/matrix-util";

export const dynamic = "force-dynamic";

export default async function MatrixDemoPage() {
  const result = await getUserMatrix();

  if ("error" in result) {
    return (
      <div className="container" style={{ paddingTop: 100 }}>
        <h1>Error loading matrix</h1>
        <p>{result.error}</p>
      </div>
    );
  }

  const matrix = result;
  const days = ["Today", "Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6"];

  return (
    <div style={{ minHeight: "100vh", background: "var(--bg)" }}>
      <Header />
      <DemoNotice label="matrix engine debugger" />

      <main
        className="container"
        style={{ paddingTop: 60, paddingBottom: 100 }}
      >
        <section style={{ marginBottom: 40 }}>
          <h1 style={{ fontSize: 32, fontWeight: 800, marginBottom: 8 }}>
            Matrix Engine Debugger
          </h1>
          <p className="meta-text" style={{ color: "var(--muted)" }}>
            Visualizing the raw 96×7 output of <code>getUserMatrix()</code>.
            This includes sleep, fixed commitments, energy windows, and
            scheduled load.
          </p>
        </section>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "80px repeat(7, 1fr)",
            gap: "1px",
            background: "var(--rule)",
            border: "0.5px solid var(--rule)",
            borderRadius: 4,
            overflow: "hidden",
          }}
        >
          {/* Header Row */}
          <div
            style={{
              background: "var(--card-bg)",
              padding: "12px 8px",
              fontWeight: 600,
              fontSize: 11,
              textAlign: "center",
            }}
          >
            TIME
          </div>
          {days.map((day, i) => (
            <div
              key={i}
              style={{
                background: "var(--card-bg)",
                padding: "12px 8px",
                fontWeight: 600,
                fontSize: 11,
                textAlign: "center",
              }}
            >
              {day.toUpperCase()}
            </div>
          ))}

          {/* Rows */}
          {Array.from({ length: 96 }).map((_, slotIdx) => (
            <React.Fragment key={slotIdx}>
              {/* Time Column */}
              <div
                style={{
                  background: "var(--card-bg)",
                  padding: "4px 8px",
                  fontSize: 10,
                  color: "var(--muted)",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  borderBottom:
                    slotIdx % 4 === 3 ? "2px solid var(--rule)" : "none",
                }}
              >
                {slotIdx % 4 === 0 ? slotToTime(slotIdx) : ""}
              </div>

              {/* Data Columns */}
              {matrix.map((day, dayIdx) => {
                const slot = day[slotIdx];
                let bg = "var(--bg)";
                let label = "";

                if (slot.isSleep) bg = "rgba(128,128,128,0.2)";
                if (slot.isFixedCommitment) bg = "rgba(100,100,255,0.2)";
                if (slot.isHardExclusion) bg = "rgba(255,100,100,0.2)";

                if (slot.etask > 0) {
                  bg = "var(--vermillion)";
                  label = `Etask ${slot.etask.toFixed(1)}`;
                } else if (slot.isPeak) {
                  bg = "rgba(100,255,100,0.15)";
                } else if (slot.isLow) {
                  bg = "rgba(255,200,100,0.15)";
                }

                return (
                  <div
                    key={`${dayIdx}-${slotIdx}`}
                    style={{
                      background: bg,
                      minHeight: 20,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: 9,
                      fontWeight: 600,
                      color: "var(--fg)",
                      borderBottom:
                        slotIdx % 4 === 3
                          ? "1px solid var(--rule)"
                          : "0.5px solid rgba(255,255,255,0.05)",
                    }}
                    title={`Day ${dayIdx}, Slot ${slotIdx}: ${slot.isAvailable ? "Available" : "Blocked"}`}
                  >
                    {label}
                  </div>
                );
              })}
            </React.Fragment>
          ))}
        </div>

        <div
          style={{ marginTop: 40, display: "flex", gap: 24, flexWrap: "wrap" }}
        >
          <LegendItem color="rgba(128,128,128,0.2)" text="Sleep" />
          <LegendItem color="rgba(100,100,255,0.2)" text="Fixed Commitment" />
          <LegendItem color="rgba(255,100,100,0.2)" text="Hard Exclusion" />
          <LegendItem color="rgba(100,255,100,0.15)" text="Peak Window" />
          <LegendItem color="rgba(255,200,100,0.15)" text="Low Energy" />
          <LegendItem color="var(--vermillion)" text="Scheduled Task" />
        </div>
        <section style={{ marginTop: 80 }}>
          <h2 style={{ fontSize: 24, fontWeight: 700, marginBottom: 16 }}>
            Raw Matrix Data (JSON)
          </h2>
          <div
            style={{
              background: "var(--card-bg)",
              padding: 24,
              borderRadius: 4,
              border: "0.5px solid var(--rule)",
              maxWidth: "100%",
              overflow: "hidden",
            }}
          >
            <pre
              style={{
                fontSize: 11,
                color: "var(--fg)",
                maxHeight: 600,
                overflowY: "auto",
                fontFamily: "var(--mono)",
                background: "rgba(0,0,0,0.2)",
                padding: 16,
              }}
            >
              {JSON.stringify(matrix, null, 2)}
            </pre>
          </div>
        </section>
      </main>
    </div>
  );
}

function LegendItem({ color, text }: { color: string; text: string }) {
  return (
    <div
      style={{ display: "flex", alignItems: "center", gap: 8, fontSize: 12 }}
    >
      <div
        style={{
          width: 16,
          height: 16,
          background: color,
          border: "0.5px solid var(--rule)",
        }}
      ></div>
      <span>{text}</span>
    </div>
  );
}
