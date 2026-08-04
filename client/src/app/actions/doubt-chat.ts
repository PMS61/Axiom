"use server";

import { sql } from "@vercel/postgres";
import { verify } from "jsonwebtoken";
import { cookies } from "next/headers";
import { getGeminiResponse } from "@/lib/gemini";
import { getCourseAction } from "./courses";

// ── Auth Helper ───────────────────────────────────────────

async function getUserId(): Promise<number | null> {
  const cookieStore = await cookies();
  const token = cookieStore.get("token")?.value;
  if (!token || !process.env.JWT_SECRET) return null;
  try {
    const decoded = verify(token, process.env.JWT_SECRET) as { userId: number };
    return decoded.userId;
  } catch {
    return null;
  }
}

// ── Schema Bootstrap ──────────────────────────────────────

async function createDoubtChatTable(): Promise<void> {
  await sql`
    CREATE TABLE IF NOT EXISTS course_doubt_chats (
      task_id UUID PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
      user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      messages JSONB NOT NULL DEFAULT '[]'::jsonb,
      updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
  `;
}

// ── Types ─────────────────────────────────────────────────

export interface DoubtChatMessage {
  role: "user" | "assistant";
  content: string;
  createdAt: string;
}

// ── Prompt construction ──────────────────────────────────

// CourseSlide (course-type.ts) is an untyped bag of fields that varies by
// slide kind (bullets, code, table, quote, ...) — a JSON dump is the only
// representation that's always accurate regardless of which slide type
// produced it, without maintaining a per-type formatter here.
function buildCourseGroundingContext(courseData: unknown): string {
  const course = (courseData ?? {}) as {
    metadata?: { title?: string; description?: string };
    slides?: Array<Record<string, unknown>>;
  };

  const slides = Array.isArray(course.slides) ? course.slides : [];
  const slideSummaries = slides
    .map((slide, index) => `Slide ${index + 1}: ${JSON.stringify(slide)}`)
    .join("\n\n");

  return `
Course title: ${course.metadata?.title ?? "Untitled"}
Course description: ${course.metadata?.description ?? ""}

Full slide content:
${slideSummaries || "(no slides generated yet)"}
`.trim();
}

function buildDoubtPrompt(
  groundingContext: string,
  history: DoubtChatMessage[],
  question: string,
): string {
  const historyText = history
    .map((m) => `${m.role === "user" ? "Student" : "Instructor"}: ${m.content}`)
    .join("\n");

  return `
You are the AI instructor for this course. Answer the student's question using ONLY the course content below as grounding. If the question falls outside what this course actually covers, say so explicitly and redirect the student back to the course material instead of answering from general knowledge.

=== COURSE CONTENT ===
${groundingContext}
=== END COURSE CONTENT ===
${historyText ? `\nConversation so far:\n${historyText}\n` : ""}
Student: ${question}
Instructor:
`.trim();
}

// ── Actions ───────────────────────────────────────────────

export async function getCourseDoubtHistoryAction(
  courseId: string,
): Promise<{ messages?: DoubtChatMessage[]; error?: string }> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    await createDoubtChatTable();
    const result = await sql`
      SELECT messages
      FROM course_doubt_chats
      WHERE user_id = ${userId} AND task_id = ${courseId}
      LIMIT 1
    `;

    if (result.rows.length === 0) return { messages: [] };
    return { messages: (result.rows[0].messages as DoubtChatMessage[]) ?? [] };
  } catch (err) {
    console.error("[getCourseDoubtHistoryAction] Error:", err);
    return { error: "Failed to fetch chat history." };
  }
}

export async function askCourseDoubtAction(
  courseId: string,
  question: string,
): Promise<{ answer?: string; messages?: DoubtChatMessage[]; error?: string }> {
  const trimmedQuestion = question.trim();
  if (!trimmedQuestion) return { error: "empty_question" };

  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    const { courseData, error: courseError } = await getCourseAction(courseId);
    if (courseError) return { error: courseError };
    if (!courseData) return { error: "Course not found." };

    await createDoubtChatTable();
    const historyResult = await sql`
      SELECT messages
      FROM course_doubt_chats
      WHERE user_id = ${userId} AND task_id = ${courseId}
      LIMIT 1
    `;
    const history: DoubtChatMessage[] =
      historyResult.rows.length > 0
        ? ((historyResult.rows[0].messages as DoubtChatMessage[]) ?? [])
        : [];

    const groundingContext = buildCourseGroundingContext(courseData);
    const prompt = buildDoubtPrompt(groundingContext, history, trimmedQuestion);
    const answer = await getGeminiResponse(prompt, false);

    const now = new Date().toISOString();
    const updatedMessages: DoubtChatMessage[] = [
      ...history,
      { role: "user", content: trimmedQuestion, createdAt: now },
      { role: "assistant", content: answer, createdAt: now },
    ];

    await sql`
      INSERT INTO course_doubt_chats (task_id, user_id, messages)
      VALUES (${courseId}, ${userId}, ${JSON.stringify(updatedMessages)}::jsonb)
      ON CONFLICT (task_id) DO UPDATE SET
        messages = EXCLUDED.messages,
        updated_at = CURRENT_TIMESTAMP
    `;

    return { answer, messages: updatedMessages };
  } catch (err) {
    console.error("[askCourseDoubtAction] Error:", err);
    return { error: "Failed to get an answer." };
  }
}
