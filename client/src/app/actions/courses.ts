"use server";

import { sql } from "@vercel/postgres";
import { cookies } from "next/headers";
import { verify } from "jsonwebtoken";

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

async function createCoursesTable(): Promise<void> {
  await sql`
    CREATE TABLE IF NOT EXISTS courses (
      task_id UUID PRIMARY KEY REFERENCES tasks(id) ON DELETE CASCADE,
      user_id INT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
      content JSONB NOT NULL,
      created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    )
  `;
}

type StoredContentType =
  | "course"
  | "story"
  | "flashcards"
  | "short_bits"
  | "cheatsheet";

export type StoredContentKind = StoredContentType;

interface StoredContentMap {
  __contentMap: true;
  items: Partial<Record<StoredContentType, unknown>>;
}

function isContentMap(value: unknown): value is StoredContentMap {
  if (!value || typeof value !== "object") return false;
  const record = value as Record<string, unknown>;
  return record.__contentMap === true && !!record.items && typeof record.items === "object";
}

function normalizeToContentMap(existing: unknown): StoredContentMap {
  if (isContentMap(existing)) return existing;
  if (existing == null) {
    return { __contentMap: true, items: {} };
  }
  // Legacy rows stored the course payload directly.
  return {
    __contentMap: true,
    items: {
      course: existing,
    },
  };
}

// ── Actions ───────────────────────────────────────────────

export async function saveCourseAction(
  taskId: string,
  content: any,
  contentType: StoredContentType = "course",
): Promise<{ success?: boolean; error?: string }> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    await createCoursesTable();

    const existing = await sql`
      SELECT content
      FROM courses
      WHERE user_id = ${userId} AND task_id = ${taskId}
      LIMIT 1
    `;

    const baseContent = existing.rows.length > 0 ? existing.rows[0].content : null;
    const mergedMap = normalizeToContentMap(baseContent);
    mergedMap.items[contentType] = content;

    await sql`
      INSERT INTO courses (task_id, user_id, content)
      VALUES (${taskId}, ${userId}, ${JSON.stringify(mergedMap)}::jsonb)
      ON CONFLICT (task_id) DO UPDATE SET
        content = EXCLUDED.content,
        created_at = CURRENT_TIMESTAMP
    `;
    return { success: true };
  } catch (err) {
    console.error("[saveCourseAction] Error:", err);
    return { error: "Failed to persist course." };
  }
}

export async function getCourseAction(taskId: string): Promise<{
  courseData?: any | null;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    await createCoursesTable();
    const result = await sql`SELECT content FROM courses WHERE user_id = ${userId} AND task_id = ${taskId} LIMIT 1`;
      
    if (result.rows.length === 0) return { courseData: null };
    const stored = result.rows[0].content;
    const contentMap = normalizeToContentMap(stored);
    return {
      courseData: contentMap.items.course ?? null,
    };
  } catch (err) {
    console.error("[getCourseAction] Error:", err);
    return { error: "Failed to fetch course." };
  }
}

export async function getStoredContentByTypeAction(
  taskId: string,
  contentType: StoredContentType,
): Promise<{
  contentData?: any | null;
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    await createCoursesTable();
    const result = await sql`
      SELECT content
      FROM courses
      WHERE user_id = ${userId} AND task_id = ${taskId}
      LIMIT 1
    `;

    if (result.rows.length === 0) return { contentData: null };

    const stored = result.rows[0].content;
    const contentMap = normalizeToContentMap(stored);
    return {
      contentData: contentMap.items[contentType] ?? null,
    };
  } catch (err) {
    console.error("[getStoredContentByTypeAction] Error:", err);
    return { error: "Failed to fetch stored content." };
  }
}

export async function getStoredContentTypesAction(
  taskId: string,
): Promise<{
  types?: StoredContentType[];
  error?: string;
}> {
  const userId = await getUserId();
  if (!userId) return { error: "Unauthorized" };

  try {
    await createCoursesTable();
    const result = await sql`
      SELECT content
      FROM courses
      WHERE user_id = ${userId} AND task_id = ${taskId}
      LIMIT 1
    `;

    if (result.rows.length === 0) return { types: [] };

    const stored = result.rows[0].content;
    const contentMap = normalizeToContentMap(stored);
    const types = Object.entries(contentMap.items)
      .filter(([, value]) => value != null)
      .map(([key]) => key as StoredContentType);

    return { types };
  } catch (err) {
    console.error("[getStoredContentTypesAction] Error:", err);
    return { error: "Failed to fetch stored content types." };
  }
}
