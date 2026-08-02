import { getGeminiResponse } from "./gemini";
import type { Assessment, AssessmentQuestion } from "./assessment-type";

export async function generateAssessment(
  title: string,
  description: string,
  goal: string,
): Promise<Assessment> {
  const prompt = `
You are an expert educational assessment designer.
Create a comprehensive assessment for the following topic:
Title: ${title}
Description: ${description}
Overarching Goal: ${goal}

Hard rules:
- Return ONLY valid JSON.
- No markdown formatting (no \`\`\`json blocks).
- Generate a mix of 4-6 questions.
- Include exactly:
  - At least 1 MCQ (type: "mcq")
  - At least 1 Short Answer (type: "short_answer")
  - At least 1 Long Answer (type: "long_answer")
  - At least 1 Coding Question (type: "coding") if applicable to the topic, otherwise replace with another MCQ or Short Answer.

JSON Schema:
{
  "title": "string",
  "description": "string",
  "questions": [
    {
      "id": "q1",
      "type": "mcq",
      "question": "string",
      "options": ["a", "b", "c", "d"],
      "correctOptionIndex": number,
      "explanation": "string"
    },
    {
      "id": "q2",
      "type": "short_answer",
      "question": "string",
      "sampleAnswer": "string",
      "keywords": ["key1", "key2"]
    },
    {
      "id": "q3",
      "type": "long_answer",
      "question": "string",
      "rubric": "string"
    },
    {
      "id": "q4",
      "type": "coding",
      "question": "string",
      "starterCode": "string",
      "language": "python|javascript|typescript|cpp|java",
      "testCases": [
        { "input": "string", "expectedOutput": "string", "description": "string", "isPublic": true }
      ]
    }
  ]
}
`;

  try {
    const raw = await getGeminiResponse(prompt, true);
    const parsed = JSON.parse(raw);
    
    if (parsed?.error) {
      throw new Error(parsed.message || parsed.error);
    }

    return parsed as Assessment;
  } catch (error) {
    console.error("[AssessmentAgent] Generation failed:", error);
    throw error;
  }
}

export async function evaluateAssessment(
  assessment: Assessment,
  userAnswers: Record<string, any>
): Promise<any> {
  // We can use Gemini to evaluate long answers and overall performance
  const prompt = `
You are an expert evaluator. Evaluate the following assessment results.
Assessment: ${JSON.stringify(assessment)}
User Answers: ${JSON.stringify(userAnswers)}

Evaluate each answer. For MCQ and Coding, check against correct options/test cases.
For Short and Long answers, use your expertise to score them.

Return a JSON object:
{
  "score": number (0.0 - 1.0),
  "feedback": "overall feedback string",
  "questionResults": [
    {
      "questionId": "string",
      "isCorrect": boolean,
      "score": number (0.0 - 1.0),
      "feedback": "feedback for this specific question"
    }
  ]
}
`;

  try {
    const raw = await getGeminiResponse(prompt, true);
    return JSON.parse(raw);
  } catch (error) {
    console.error("[AssessmentAgent] Evaluation failed:", error);
    throw error;
  }
}
