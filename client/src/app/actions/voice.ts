"use server";
import { getGeminiResponse } from "@/lib/gemini";
import type { TaskState } from "@/lib/types";

export type VoiceAction =
  | { type: "SET_ENERGY"; payload: number }
  | { type: "UPDATE_TASK_STATE"; payload: { taskId: string; state: TaskState } }
  | { type: "ADAPTIVE_RESCHEDULE"; payload: { reason: string } }
  | { type: "ASK_FOCUS"; payload: null }
  | { type: "UNKNOWN"; payload: string };

export async function parseVoiceCommand(
  transcript: string,
  tasks: Array<{ id: string; name: string }>,
): Promise<VoiceAction> {
  const prompt = `
    You are the Voice Command Parser for Axiom, an AI productivity dashboard.
    User Transcript: "${transcript}"
    
    Existing Tasks: ${JSON.stringify(tasks.map((t) => ({ id: t.id, name: t.name })))}
    
    Map the transcript to one of the following actions:
    1. SET_ENERGY: Update energy level (-2 to 2). Keywords: "tired", "energetic", "high energy", "low energy".
    2. UPDATE_TASK_STATE: Mark a task as completed or skipped. Find the closest matching task name from the list.
    3. ADAPTIVE_RESCHEDULE: Trigger a reschedule. Keywords: "reschedule", "replan", "fix my day".
    4. ASK_FOCUS: User is asking what to do next. Keywords: "focus", "next", "what's next".
    
    Return a JSON object:
    {
      "type": "ACTION_TYPE",
      "payload": { ... }
    }
    
    Example: "I finished the algorithms study" -> { "type": "UPDATE_TASK_STATE", "payload": { "taskId": "task-101", "state": "completed" } }
    Example: "I'm feeling very energetic" -> { "type": "SET_ENERGY", "payload": 2 }
    Example: "What should I focus on now?" -> { "type": "ASK_FOCUS", "payload": null }
    
    If no match is found, return { "type": "UNKNOWN", "payload": "Original transcript" }.
  `;

  try {
    const response = await getGeminiResponse(prompt, true);
    return JSON.parse(response);
  } catch (error) {
    console.error("Failed to parse voice command:", error);
    return { type: "UNKNOWN", payload: transcript };
  }
}
