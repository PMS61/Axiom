"use server";

import { generateAssessment, evaluateAssessment } from "@/lib/assessment-agent";
import type { Assessment, AssessmentResult } from "@/lib/assessment-type";

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
): Promise<{ result?: AssessmentResult; error?: string }> {
  try {
    const result = await evaluateAssessment(assessment, userAnswers);
    return { result };
  } catch (error: any) {
    return { error: error.message };
  }
}
