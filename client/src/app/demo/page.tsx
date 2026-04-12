"use client";

import { useState } from "react";
import { generateDemoCourse } from "./actions";
import Header from "@/components/Header";

export default function DemoCoursePage() {
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [error, setError] = useState<string | null>(null);

  async function handleGenerate(formData: FormData) {
    setLoading(true);
    setResult(null);
    setError(null);
    
    try {
      const response = await generateDemoCourse(formData);
      if (response.success) {
        setResult(response.data);
      } else {
        setError(response.error);
      }
    } catch (e: any) {
      setError(e.message || "An error occurred");
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Header />
      <div style={{ paddingTop: 60, minHeight: "100vh", backgroundColor: "var(--bg)" }}>
        <section className="container section-rule" style={{ paddingTop: 40, paddingBottom: 40 }}>
          <div style={{ marginBottom: 40 }}>
            <h1 style={{ fontFamily: "var(--font-playfair)", fontSize: 36, letterSpacing: -0.5, color: "var(--ink)", marginBottom: 8 }}>
              Course Generation
            </h1>
            <div className="meta-text" style={{ color: "var(--muted)" }}>
              Test the adaptive course generation agent.
            </div>
          </div>

          <form action={handleGenerate} style={{ display: "flex", flexDirection: "column", gap: 32, maxWidth: 600 }}>
            <div>
              <label htmlFor="title" style={{ display: "block", marginBottom: 8, color: "var(--ink)" }}>COURSE TITLE</label>
              <input 
                required
                id="title"
                name="title" 
                type="text" 
                placeholder="e.g., Introduction to React..."
              />
            </div>
            
            <div>
              <label htmlFor="description" style={{ display: "block", marginBottom: 8, color: "var(--ink)" }}>COURSE DESCRIPTION</label>
              <textarea 
                required
                id="description"
                name="description" 
                rows={3}
                placeholder="e.g., A comprehensive guide to building interactive UI with React..."
              />
            </div>

            <div>
              <label htmlFor="userPrompt" style={{ display: "block", marginBottom: 8, color: "var(--ink)" }}>CUSTOM INSTRUCTIONS (OPTIONAL)</label>
              <input 
                id="userPrompt"
                name="userPrompt" 
                type="text" 
                placeholder="e.g., Focus heavily on hooks and state management"
              />
            </div>

            <button 
              type="submit" 
              className="btn btn-primary"
              disabled={loading}
              style={{ opacity: loading ? 0.5 : 1, width: "fit-content" }}
            >
              {loading ? "GENERATING SEQUENCE..." : "GENERATE SKELETON & SLIDES"}
            </button>
          </form>
        </section>

        {error && (
          <section className="container section-rule" style={{ paddingTop: 20, paddingBottom: 20 }}>
            <div className="meta-text" style={{ color: "var(--vermillion)" }}>
              ERR: {error}
            </div>
          </section>
        )}

        {result && (
          <section className="container" style={{ paddingTop: 40, paddingBottom: 80 }}>


            <div>
              <div className="meta-text" style={{ marginBottom: 16 }}>01. GENERATED SLIDES ({result.slides?.length || 0})</div>
              
              <div style={{ display: "flex", flexDirection: "column", gap: 32 }}>
                {(result.slides || []).map((slide: any, index: number) => (
                  <div key={index} style={{ border: "0.5px solid var(--rule)", padding: 24, backgroundColor: "var(--card-bg)" }}>
                    <div style={{ display: "flex", justifyContent: "space-between", borderBottom: "0.5px solid var(--rule)", paddingBottom: 16, marginBottom: 16 }}>
                      <div className="meta-text" style={{ color: "var(--ink)" }}>SLIDE NO. {(index + 1).toString().padStart(2, "0")}</div>
                      <div className="meta-text" style={{ color: "var(--muted)" }}>TYPE: {slide.type}</div>
                    </div>
                    {slide.title && (
                      <div style={{ fontFamily: "var(--font-playfair)", fontSize: 24, marginBottom: 16, color: "var(--ink)" }}>
                        {slide.title}
                      </div>
                    )}
                    <pre className="meta-text" style={{ whiteSpace: "pre-wrap", wordBreak: "break-word", lineHeight: 1.6 }}>
                      {JSON.stringify(slide, null, 2)}
                    </pre>
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
      </div>
    </>
  );
}
