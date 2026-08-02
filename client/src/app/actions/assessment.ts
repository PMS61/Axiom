"use server";

import { verify } from "jsonwebtoken";
import { cookies } from "next/headers";
import { evaluateAssessment, generateAssessment } from "@/lib/assessment-agent";
import type { Assessment, AssessmentResult } from "@/lib/assessment-type";
import { normalizeTopicKey } from "@/lib/personalization";
import { upsertMastery } from "./personalization";

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

export async function generateAssessmentAction(
  title: string,
  description: string,
  goal: string,
): Promise<{ assessment?: Assessment; error?: string }> {
  try {
    const assessment = await generateAssessment(title, description, goal);
    return { assessment };
  } catch (error: any) {
    return { error: error.message };
  }
}

export async function evaluateAssessmentAction(
  assessment: Assessment,
  userAnswers: Record<string, any>,
  topic?: string,
): Promise<{ result?: AssessmentResult; error?: string }> {
  try {
    const result = await evaluateAssessment(assessment, userAnswers);

    // Cross-pillar mastery signal (plans/04) — a real graded score, so this
    // is the authoritative source: it always overwrites, per upsertMastery's
    // documented merge policy.
    if (topic) {
      const userId = await getUserId();
      if (userId) {
        await upsertMastery(
          userId,
          normalizeTopicKey(topic),
          result.score,
          "assessment",
        );
      }
    }

    return { result };
  } catch (error: any) {
    return { error: error.message };
  }
}
