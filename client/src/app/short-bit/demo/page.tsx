"use client";

import { useState } from "react";
import DemoNotice from "@/components/DemoNotice";
import ShortBitStack from "@/components/ShortBitStack";
import { generateShortBits } from "./actions";

type ShortBitItem = {
  id: string;
  imageKeyword: string;
  blogContent: string;
};

type ShortBitSet = {
  title: string;
  items: ShortBitItem[];
};

export default function ShortBitDemoPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ShortBitSet | null>(null);

  async function handleGenerate(formData: FormData) {
    setLoading(true);
    setError(null);

    try {
      const response = await generateShortBits(formData);
      if (!response.success || !response.data) {
        setResult(null);
        setError(response.error || "Failed to generate short bits.");
        return;
      }

      setResult(response.data);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Unexpected error";
      setError(message);
      setResult(null);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div
      style={{
        minHeight: "100vh",
        backgroundColor: "var(--bg)",
        paddingBottom: 40,
      }}
    >
      <DemoNotice label="short-bit generation sandbox" />
      <section
        className="container section-rule"
        style={{ paddingTop: 40, paddingBottom: 32 }}
      >
        <h1
          style={{
            fontFamily: "var(--font-playfair)",
            color: "var(--ink)",
            marginBottom: 8,
          }}
        >
          Short Bit Agent Demo
        </h1>
        <div
          className="meta-text"
          style={{ color: "var(--muted)", marginBottom: 24 }}
        >
          Generate short visual content objects from title, description, and
          additional instructions.
        </div>

        <form
          action={handleGenerate}
          style={{ display: "grid", gap: 16, maxWidth: 760 }}
        >
          <div>
            <label htmlFor="title">TITLE</label>
            <input
              id="title"
              name="title"
              required
              placeholder="e.g., Learning Systems Design"
            />
          </div>

          <div>
            <label htmlFor="description">DESCRIPTION</label>
            <textarea
              id="description"
              name="description"
              required
              rows={3}
              placeholder="e.g., Practical habits and frameworks for technical decision making."
            />
          </div>

          <div>
            <label htmlFor="additionalInstructions">
              ADDITIONAL INSTRUCTIONS (OPTIONAL)
            </label>
            <textarea
              id="additionalInstructions"
              name="additionalInstructions"
              rows={3}
              placeholder="e.g., Keep it motivational and beginner-friendly."
            />
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
            style={{ width: "fit-content", opacity: loading ? 0.6 : 1 }}
          >
            {loading ? "GENERATING..." : "GENERATE SHORT BITS"}
          </button>
        </form>

        {error && (
          <div
            className="meta-text"
            style={{ color: "var(--vermillion)", marginTop: 16 }}
          >
            ERR: {error}
          </div>
        )}
      </section>

      {result && (
        <section
          className="section-rule"
          style={{ marginTop: 20, paddingBottom: 16 }}
        >
          <ShortBitStack
            items={result.items}
            heading={result.title || "Short Bit"}
          />
        </section>
      )}
    </div>
  );
}
