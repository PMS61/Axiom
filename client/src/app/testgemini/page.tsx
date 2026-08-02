"use client";

import { useState } from "react";
import DemoNotice from "@/components/DemoNotice";
import Header from "@/components/Header";
import { sendChatMessage } from "./actions";

export default function TestGeminiPage() {
  const [messages, setMessages] = useState<
    { role: "user" | "bot"; text: string }[]
  >([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSend(e: React.FormEvent) {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput("");
    setMessages((prev) => [...prev, { role: "user", text: userMessage }]);
    setLoading(true);

    try {
      const result = await sendChatMessage(userMessage);
      if (result.success) {
        setMessages((prev) => [
          ...prev,
          { role: "bot", text: result.text || "" },
        ]);
      } else {
        setMessages((prev) => [
          ...prev,
          { role: "bot", text: `Error: ${result.error}` },
        ]);
      }
    } catch (err: any) {
      setMessages((prev) => [
        ...prev,
        { role: "bot", text: "Caught error communicating with backend." },
      ]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <>
      <Header />
      <DemoNotice label="raw Gemma 4 prompt tester" />
      <div
        style={{
          paddingTop: 60,
          minHeight: "100vh",
          backgroundColor: "var(--bg)",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <section
          className="container section-rule"
          style={{ paddingTop: 40, paddingBottom: 20 }}
        >
          <div>
            <h1
              style={{
                fontFamily: "var(--font-playfair)",
                fontSize: 36,
                letterSpacing: -0.5,
                color: "var(--ink)",
                marginBottom: 8,
              }}
            >
              Gemini Test Interface
            </h1>
            <div className="meta-text" style={{ color: "var(--muted)" }}>
              Run raw chat prompts to test your integration.
            </div>
          </div>
        </section>

        <section
          className="container"
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            paddingBottom: 40,
          }}
        >
          <div
            style={{
              flex: 1,
              border: "0.5px solid var(--rule)",
              backgroundColor: "var(--card-bg)",
              padding: 24,
              marginBottom: 20,
              overflowY: "auto",
              display: "flex",
              flexDirection: "column",
              gap: 24,
              minHeight: "50vh",
            }}
          >
            {messages.length === 0 ? (
              <div
                className="meta-text"
                style={{ color: "var(--muted)", textAlign: "left" }}
              >
                NO MESSAGES YET. START TYPING TO TEST.
              </div>
            ) : (
              messages.map((msg, idx) => (
                <div
                  key={idx}
                  style={{
                    alignSelf: msg.role === "user" ? "flex-end" : "flex-start",
                    maxWidth: "80%",
                  }}
                >
                  <div
                    className="meta-text"
                    style={{
                      fontSize: 10,
                      color: "var(--muted)",
                      marginBottom: 4,
                      textAlign: msg.role === "user" ? "right" : "left",
                    }}
                  >
                    {msg.role === "user" ? "USER_PROMPT" : "SYS_GEMINI"}
                  </div>
                  <div
                    style={{
                      padding: 16,
                      backgroundColor:
                        msg.role === "user" ? "var(--ink)" : "var(--bg)",
                      border:
                        msg.role === "user"
                          ? "none"
                          : "0.5px solid var(--rule)",
                      color: msg.role === "user" ? "var(--bg)" : "var(--ink)",
                      fontFamily:
                        msg.role === "bot" ? "var(--mono)" : "inherit",
                      whiteSpace: "pre-wrap",
                    }}
                  >
                    {msg.text}
                  </div>
                </div>
              ))
            )}
            {loading && (
              <div style={{ alignSelf: "flex-start", maxWidth: "80%" }}>
                <div
                  className="meta-text"
                  style={{
                    fontSize: 10,
                    color: "var(--muted)",
                    marginBottom: 4,
                  }}
                >
                  SYS_GEMINI
                </div>
                <div
                  style={{
                    padding: 16,
                    border: "0.5px solid var(--rule)",
                    color: "var(--muted)",
                    display: "flex",
                    gap: 8,
                    alignItems: "center",
                  }}
                >
                  <div
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      backgroundColor: "var(--warning)",
                      animation: "pulse 1.5s infinite",
                    }}
                  />
                  ANALYSING...
                </div>
              </div>
            )}
          </div>

          <form
            onSubmit={handleSend}
            style={{ display: "flex", gap: 16, alignItems: "flex-end" }}
          >
            <div style={{ flex: 1 }}>
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                type="text"
                placeholder="Enter prompt here..."
                disabled={loading}
                style={{ paddingLeft: 16, paddingRight: 16 }}
              />
            </div>
            <button
              type="submit"
              className="btn btn-primary"
              disabled={loading || !input.trim()}
              style={{ padding: "12px 24px", height: "fit-content" }}
            >
              SEND
            </button>
          </form>
        </section>
      </div>
      <style
        dangerouslySetInnerHTML={{
          __html: `
        @keyframes pulse {
          0% { opacity: 0.2; }
          50% { opacity: 1; }
          100% { opacity: 0.2; }
        }
      `,
        }}
      />
    </>
  );
}
