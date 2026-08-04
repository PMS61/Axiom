"use client";

import { useEffect, useRef, useState } from "react";
import {
  askCourseDoubtAction,
  type DoubtChatMessage,
  getCourseDoubtHistoryAction,
} from "@/app/actions/doubt-chat";
import { Button } from "@/components/ui/button";

export interface CourseDoubtChatProps {
  courseId: string;
  heading?: string;
}

export default function CourseDoubtChat({
  courseId,
  heading = "Ask a Doubt",
}: CourseDoubtChatProps) {
  const [messages, setMessages] = useState<DoubtChatMessage[]>([]);
  const [question, setQuestion] = useState("");
  const [loadingHistory, setLoadingHistory] = useState(true);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    setLoadingHistory(true);
    getCourseDoubtHistoryAction(courseId).then((res) => {
      if (cancelled) return;
      if (res.messages) setMessages(res.messages);
      if (res.error) setError(res.error);
      setLoadingHistory(false);
    });
    return () => {
      cancelled = true;
    };
  }, [courseId]);

  useEffect(() => {
    if (messages.length === 0) return;
    scrollRef.current?.scrollTo({
      top: scrollRef.current.scrollHeight,
      behavior: "smooth",
    });
  }, [messages]);

  const handleSend = async () => {
    const trimmed = question.trim();
    if (!trimmed || sending) return;

    setSending(true);
    setError(null);
    setQuestion("");

    // Optimistic append so the student's own message shows immediately.
    const optimisticUser: DoubtChatMessage = {
      role: "user",
      content: trimmed,
      createdAt: new Date().toISOString(),
    };
    setMessages((prev) => [...prev, optimisticUser]);

    const res = await askCourseDoubtAction(courseId, trimmed);
    setSending(false);

    if (res.error) {
      setError(res.error);
      return;
    }
    if (res.messages) setMessages(res.messages);
  };

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        border: "0.5px solid var(--rule)",
        backgroundColor: "var(--card-bg)",
        height: "450px",
      }}
    >
      <div
        style={{
          padding: "16px 20px",
          borderBottom: "0.5px solid var(--rule)",
        }}
      >
        <div className="meta-text" style={{ color: "var(--ink)" }}>
          {heading}
        </div>
        <div
          className="meta-text"
          style={{ color: "var(--muted)", fontSize: "10px", marginTop: 4 }}
        >
          Answers are grounded in this course's content only.
        </div>
      </div>

      <div
        ref={scrollRef}
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "16px 20px",
          display: "flex",
          flexDirection: "column",
          gap: 12,
        }}
      >
        {loadingHistory && (
          <div className="meta-text" style={{ color: "var(--muted)" }}>
            Loading conversation…
          </div>
        )}

        {!loadingHistory && messages.length === 0 && (
          <div className="meta-text" style={{ color: "var(--muted)" }}>
            No questions yet. Ask anything about this course.
          </div>
        )}

        {messages.map((message, index) => (
          <div
            // biome-ignore lint/suspicious/noArrayIndexKey: messages are append-only within a session, index is stable for the render lifetime
            key={index}
            style={{
              alignSelf: message.role === "user" ? "flex-end" : "flex-start",
              maxWidth: "80%",
            }}
          >
            <div
              className="meta-text"
              style={{
                color:
                  message.role === "user"
                    ? "var(--vermillion)"
                    : "var(--muted)",
                fontSize: "10px",
                marginBottom: 4,
                textAlign: message.role === "user" ? "right" : "left",
              }}
            >
              {message.role === "user" ? "You" : "Instructor"}
            </div>
            <div
              style={{
                padding: "10px 14px",
                border: "0.5px solid var(--rule)",
                backgroundColor:
                  message.role === "user" ? "var(--bg)" : "var(--card-bg)",
                fontSize: 14,
                lineHeight: 1.5,
                color: "var(--ink)",
                whiteSpace: "pre-wrap",
              }}
            >
              {message.content}
            </div>
          </div>
        ))}

        {sending && (
          <div
            className="meta-text"
            style={{ color: "var(--muted)", alignSelf: "flex-start" }}
          >
            Instructor is answering…
          </div>
        )}
      </div>

      {error && (
        <div
          className="meta-text"
          style={{ color: "var(--vermillion)", padding: "0 20px 8px" }}
        >
          {error}
        </div>
      )}

      <div
        style={{
          display: "flex",
          gap: 8,
          padding: "12px 20px",
          borderTop: "0.5px solid var(--rule)",
        }}
      >
        <input
          type="text"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              handleSend();
            }
          }}
          placeholder="Ask a question about this course…"
          disabled={sending}
          style={{
            flex: 1,
            border: "0.5px solid var(--rule)",
            backgroundColor: "var(--bg)",
            color: "var(--ink)",
            padding: "8px 12px",
            fontSize: 14,
            fontFamily: "var(--mono)",
          }}
        />
        <Button
          onClick={handleSend}
          disabled={sending || !question.trim()}
          size="sm"
          className="btn"
        >
          Send
        </Button>
      </div>
    </div>
  );
}
