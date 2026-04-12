"use client";

import { useMemo, useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

/**
 * @param {{ items: Array<{ id: string; imageKeyword?: string; blogContent: string }>; heading?: string }} props
 */
export default function ShortBitStack({ items = [], heading = "Short Bit" }) {
  const [currentIndex, setCurrentIndex] = useState(0);

  const current = useMemo(
    () => items[currentIndex] ?? null,
    [items, currentIndex],
  );

  const handlePrev = () => {
    setCurrentIndex((prev) => Math.max(prev - 1, 0));
  };

  const handleNext = () => {
    setCurrentIndex((prev) => Math.min(prev + 1, items.length - 1));
  };

  if (!items.length) {
    return (
      <div
        className="container section-rule"
        style={{ paddingTop: 24, paddingBottom: 24 }}
      >
        <div className="meta-text" style={{ color: "var(--muted)" }}>
          No short bits available.
        </div>
      </div>
    );
  }

  return (
    <div
      style={{
        backgroundColor: "var(--bg)",
        paddingTop: 28,
        paddingBottom: 40,
        paddingInline: "clamp(10px, 2.5vw, 28px)",
      }}
    >
      <div className="container" style={{ maxWidth: 920 }}>
        <div style={{ marginBottom: 20 }}>
          <div
            className="meta-text"
            style={{
              color: "var(--muted)",
              marginBottom: 8,
              letterSpacing: "0.14em",
            }}
          >
            MICRO LEARNING STREAM
          </div>
          <h2
            style={{
              fontFamily: "var(--font-playfair)",
              fontSize: 34,
              color: "var(--ink)",
              marginBottom: 6,
              lineHeight: 1.1,
            }}
          >
            {heading}
          </h2>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              gap: 12,
              flexWrap: "wrap",
            }}
          >
            <div className="meta-text" style={{ color: "var(--muted)" }}>
              Bit {currentIndex + 1} of {items.length}
            </div>
            <div
              style={{
                display: "inline-flex",
                alignItems: "center",
                gap: 6,
                padding: "4px 8px",
                border: "0.5px solid var(--rule)",
                background: "var(--card-bg)",
              }}
            >
              {items.slice(0, 10).map((item, idx) => (
                <span
                  key={item.id}
                  style={{
                    width: 7,
                    height: 7,
                    borderRadius: 999,
                    background:
                      idx === currentIndex ? "var(--ink)" : "var(--rule)",
                    opacity: idx === currentIndex ? 1 : 0.6,
                    display: "inline-block",
                  }}
                />
              ))}
            </div>
          </div>
        </div>

        <Card
          style={{
            backgroundColor: "var(--card-bg)",
            border: "0.5px solid var(--rule)",
            boxShadow: "0 18px 42px rgba(0, 0, 0, 0.08)",
            overflow: "hidden",
          }}
        >
          <div
            style={{
              height: 3,
              width: "100%",
              background:
                "linear-gradient(90deg, var(--ink) 0%, var(--watch) 60%, var(--safe) 100%)",
              opacity: 0.9,
            }}
          />
          <CardContent style={{ padding: "clamp(18px, 3vw, 34px)" }}>
            <div style={{ display: "grid", gap: 14 }}>
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  gap: 8,
                  flexWrap: "wrap",
                }}
              >
                <div
                  className="meta-text"
                  style={{ color: "var(--muted)", letterSpacing: "0.12em" }}
                >
                  {current.imageKeyword
                    ? `VISUAL CUE: ${current.imageKeyword.toUpperCase()}`
                    : "VISUAL CUE"}
                </div>
                <div
                  style={{
                    fontFamily: "var(--mono)",
                    fontSize: 10,
                    color: "var(--muted)",
                    padding: "4px 8px",
                    border: "0.5px solid var(--rule)",
                  }}
                >
                  ~30 SEC READ
                </div>
              </div>
              <div
                style={{
                  fontSize: 26,
                  color: "var(--ink)",
                  lineHeight: 0.8,
                  opacity: 0.3,
                }}
              >
                “
              </div>
              <p
                style={{
                  margin: 0,
                  fontSize: 20,
                  lineHeight: 1.75,
                  color: "var(--ink)",
                  fontFamily: "var(--sans, var(--mono))",
                }}
              >
                {current.blogContent}
              </p>
              <div
                className="meta-text"
                style={{
                  color: "var(--muted)",
                  marginTop: 6,
                  paddingTop: 12,
                  borderTop: "0.5px solid var(--rule)",
                }}
              >
                Tip {String(currentIndex + 1).padStart(2, "0")}
              </div>
            </div>
          </CardContent>
        </Card>

        <div
          style={{
            marginTop: 18,
            display: "flex",
            justifyContent: "space-between",
            gap: 10,
          }}
        >
          <Button
            onClick={handlePrev}
            disabled={currentIndex === 0}
            className="btn"
            style={{
              opacity: currentIndex === 0 ? 0.5 : 1,
              minWidth: 110,
              justifyContent: "center",
            }}
          >
            ← Prev
          </Button>
          <div
            className="meta-text"
            style={{
              color: "var(--muted)",
              alignSelf: "center",
              textAlign: "center",
            }}
          >
            Swipe your focus one bit at a time
          </div>
          <Button
            onClick={handleNext}
            disabled={currentIndex >= items.length - 1}
            className="btn"
            style={{
              opacity: currentIndex >= items.length - 1 ? 0.5 : 1,
              minWidth: 110,
              justifyContent: "center",
            }}
          >
            Next →
          </Button>
        </div>
      </div>
    </div>
  );
}
