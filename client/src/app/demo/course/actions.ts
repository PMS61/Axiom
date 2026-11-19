"use server";

import { generateCourseComplete } from "@/lib/course-agent";

export async function generateDemoCourse(formData: FormData) {
  const title = formData.get("title") as string;
  const description = formData.get("description") as string;
  const userPrompt = (formData.get("userPrompt") as string) || "";

  try {
    const result = await generateCourseComplete(title, description, userPrompt);
    return { success: true, data: result };
  } catch (error: any) {
    console.error("Demo Generation Error:", error);
    return {
      success: false,
      error: error.message || "Failed to generate course",
    };
  }
}
