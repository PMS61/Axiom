import { getGeminiResponse } from "./gemini";

export type CheatSheetSection = {
  heading: string;
  summary: string;
  keyPoints: string[];
  formulasOrSyntax: string[];
  examples: string[];
  pitfalls: string[];
  quickChecks: string[];
};

export type CheatSheetPayload = {
  cheatSheetTitle: string;
  examChecklist: string[];
  rapidRevision: string[];
  sections: CheatSheetSection[];
};

function normalizePayload(
  payload: unknown,
  title: string,
  description: string,
): CheatSheetPayload {
  if (!payload || typeof payload !== "object") {
    return fallbackPayload(title, description);
  }

  const candidate = payload as Partial<CheatSheetPayload>;
  const sections = Array.isArray(candidate.sections) ? candidate.sections : [];

  const safeSections: CheatSheetSection[] = sections
    .map((section) => {
      if (!section || typeof section !== "object") return null;
      const data = section as Partial<CheatSheetSection>;

      const heading =
        typeof data.heading === "string" && data.heading.trim().length > 0
          ? data.heading.trim()
          : "Core Topic";

      const summary =
        typeof data.summary === "string" && data.summary.trim().length > 0
          ? data.summary.trim()
          : "Review this topic's essentials.";

      const asStringArray = (value: unknown): string[] =>
        Array.isArray(value)
          ? value
              .map((item) => (typeof item === "string" ? item.trim() : ""))
              .filter(Boolean)
          : [];

      return {
        heading,
        summary,
        keyPoints: asStringArray(data.keyPoints),
        formulasOrSyntax: asStringArray(data.formulasOrSyntax),
        examples: asStringArray(data.examples),
        pitfalls: asStringArray(data.pitfalls),
        quickChecks: asStringArray(data.quickChecks),
      };
    })
    .filter((section): section is CheatSheetSection => Boolean(section));

  return {
    cheatSheetTitle:
      typeof candidate.cheatSheetTitle === "string" &&
      candidate.cheatSheetTitle.trim().length > 0
        ? candidate.cheatSheetTitle.trim()
        : `${title} Cheat Sheet`,
    examChecklist: Array.isArray(candidate.examChecklist)
      ? candidate.examChecklist.filter(
          (item): item is string =>
            typeof item === "string" && item.trim().length > 0,
        )
      : [],
    rapidRevision: Array.isArray(candidate.rapidRevision)
      ? candidate.rapidRevision.filter(
          (item): item is string =>
            typeof item === "string" && item.trim().length > 0,
        )
      : [],
    sections:
      safeSections.length > 0
        ? safeSections
        : fallbackPayload(title, description).sections,
  };
}

function fallbackPayload(
  title: string,
  description: string,
): CheatSheetPayload {
  return {
    cheatSheetTitle: `${title} Cheat Sheet`,
    sections: [
      {
        heading: "High-Level Overview",
        summary: description || `A concise revision guide for ${title}.`,
        keyPoints: [
          `Understand the foundational concepts of ${title}.`,
          "Prioritize core definitions and relationships.",
          "Practice retrieval using active recall prompts.",
        ],
        formulasOrSyntax: [
          "No specific formulas provided. Add course-specific equations here.",
        ],
        examples: ["Build one small worked example from your class notes."],
        pitfalls: ["Memorizing without understanding application scenarios."],
        quickChecks: ["Can you explain this topic in 60 seconds from memory?"],
      },
    ],
    examChecklist: [
      "Review all section summaries once.",
      "Test yourself with blank-paper recall.",
      "Rehearse two exam-style problems per major topic.",
    ],
    rapidRevision: [
      "Focus on definitions, formulas, and common traps.",
      "Prioritize weak topics first.",
      "Do a final 10-minute active recall sprint.",
    ],
  };
}

function toMarkdown(
  payload: CheatSheetPayload,
  title: string,
  additionalInstructions: string,
): string {
  const lines: string[] = [];

  lines.push(`# ${payload.cheatSheetTitle || `${title} Cheat Sheet`}`);
  lines.push("");
  lines.push("## Snapshot");
  lines.push(`- **Course:** ${title}`);
  lines.push(
    `- **Instructions Context:** ${additionalInstructions || "None provided"}`,
  );
  lines.push("");

  payload.sections.forEach((section, index) => {
    lines.push(`## ${index + 1}. ${section.heading}`);
    lines.push("");
    lines.push(section.summary);
    lines.push("");

    if (section.keyPoints.length > 0) {
      lines.push("### Key Points");
      section.keyPoints.forEach((item) => {
        lines.push(`- ${item}`);
      });
      lines.push("");
    }

    if (section.formulasOrSyntax.length > 0) {
      lines.push("### Formulas / Syntax");
      section.formulasOrSyntax.forEach((item) => {
        lines.push(`- ${item}`);
      });
      lines.push("");
    }

    if (section.examples.length > 0) {
      lines.push("### Quick Examples");
      section.examples.forEach((item) => {
        lines.push(`- ${item}`);
      });
      lines.push("");
    }

    if (section.pitfalls.length > 0) {
      lines.push("### Common Pitfalls");
      section.pitfalls.forEach((item) => {
        lines.push(`- ${item}`);
      });
      lines.push("");
    }

    if (section.quickChecks.length > 0) {
      lines.push("### Quick Checks");
      section.quickChecks.forEach((item) => {
        lines.push(`- ${item}`);
      });
      lines.push("");
    }
  });

  if (payload.rapidRevision.length > 0) {
    lines.push("## Rapid Revision (Last 15 Minutes)");
    payload.rapidRevision.forEach((item) => {
      lines.push(`- ${item}`);
    });
    lines.push("");
  }

  if (payload.examChecklist.length > 0) {
    lines.push("## Exam Checklist");
    payload.examChecklist.forEach((item) => {
      lines.push(`- ${item}`);
    });
    lines.push("");
  }

  return lines.join("\n").trim();
}

export async function generateCheatSheetMarkdown(
  title: string,
  description: string,
  additionalInstructions: string = "",
): Promise<string> {
  const prompt = `
You are an expert academic revision coach.
Generate a COMPLETE course cheat sheet in one pass.

Input course title: ${title}
Input course description: ${description}
Additional instructions: ${additionalInstructions || "None"}

Hard requirements:
- Generate the entire cheat sheet in ONE response.
- Keep it concise, exam-focused, and memory-oriented.
- Include high-yield concepts, formulas/syntax, examples, pitfalls, and quick checks.
- Return ONLY valid JSON. No markdown. No code fences.

JSON schema to follow exactly:
{
  "cheatSheetTitle": "string",
  "sections": [
    {
      "heading": "string",
      "summary": "string",
      "keyPoints": ["string"],
      "formulasOrSyntax": ["string"],
      "examples": ["string"],
      "pitfalls": ["string"],
      "quickChecks": ["string"]
    }
  ],
  "rapidRevision": ["string"],
  "examChecklist": ["string"]
}
`;

  try {
    const raw = await getGeminiResponse(prompt, true);
    const parsed = JSON.parse(raw);
    if (parsed?.error) {
      throw new Error(parsed.message || parsed.error);
    }
    const payload = normalizePayload(parsed, title, description);
    return toMarkdown(payload, title, additionalInstructions);
  } catch (error) {
    console.error("[CheatSheetAgent] generation failed:", error);
    throw error;
  }
}
