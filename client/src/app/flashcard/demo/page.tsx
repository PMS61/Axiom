"use client";

import { useState } from "react";
import DemoNotice from "@/components/DemoNotice";
import FlashcardStack from "@/components/FlashcardStack";
import { generateFlashcards } from "./actions";

type Flashcard = {
  id: string;
  front: string;
  back: string;
};

type FlashcardSet = {
  title: string;
  cards: Flashcard[];
};

export default function FlashcardDemoPage() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<FlashcardSet | null>(null);

  async function handleGenerate(formData: FormData) {
    setLoading(true);
    setError(null);

    try {
      const response = await generateFlashcards(formData);
      if (!response.success || !response.data) {
        setResult(null);
        setError(response.error || "Failed to generate flashcards.");
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
      <DemoNotice label="flashcard generation sandbox" />
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
          Flashcard Agent Demo
        </h1>
        <div
          className="meta-text"
          style={{ color: "var(--muted)", marginBottom: 24 }}
        >
          Generate study cards from course title, description, and additional
          instructions.
        </div>

        <form
          action={handleGenerate}
          style={{ display: "grid", gap: 16, maxWidth: 760 }}
        >
          <div>
            <label htmlFor="title">COURSE TITLE</label>
            <input
              id="title"
              name="title"
              required
              placeholder="e.g., Operating Systems"
            />
          </div>

          <div>
            <label htmlFor="description">COURSE DESCRIPTION</label>
            <textarea
              id="description"
              name="description"
              required
              rows={3}
              placeholder="e.g., Processes, scheduling, memory management, concurrency, file systems."
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
              placeholder="e.g., Focus on interview-style conceptual and tricky confusion cards."
            />
          </div>

          <button
            type="submit"
            className="btn btn-primary"
            disabled={loading}
            style={{ width: "fit-content", opacity: loading ? 0.6 : 1 }}
          >
            {loading ? "GENERATING..." : "GENERATE FLASHCARDS"}
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
          <FlashcardStack
            cards={result.cards}
            heading={result.title || "Generated Flashcards"}
          />
        </section>
      )}
    </div>
  );
}
