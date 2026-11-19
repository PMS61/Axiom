import type {
  CompleteCourse,
  CourseSkeleton,
  CourseSlide,
} from "./course-type";
import { getGeminiResponse } from "./gemini";
import { slideSchema } from "./learning-content";
import { slidePrompts } from "./slide-prompts";

export async function generateCourseSkeleton(
  title: string,
  description: string,
  userPrompt: string = "",
  compact = false,
): Promise<CourseSkeleton[]> {
  const commonRules = `
  - Only respond with the course skeleton in JSON format.
  - Do not include any additional text or explanations.
  - Do not include any markdown formatting.
  - Ensure the JSON is valid and properly formatted.
  - The "type" field should be an array of slide types.
  - Use single type arrays for simple slides (e.g., ["title"], ["section"], ["content"]).
  - Limit to maximum 3 slides per topic. If a topic needs more than 3 slides, divide it into 2 separate topic titles.
  `;

  const jsonFormat = compact
    ? `[
    {"title":"Core concept","description":"Explain the objective and essential idea.","type":["title","content"]},
    {"title":"Worked example","description":"Walk through a specific example and explain the result.","type":["table","content"]},
    {"title":"Practice and takeaway","description":"Give a retrieval task and explain its answer.","type":["list","conclusion"]}
  ]`
    : `
  [
    {
      "title": "Slide Title 1",
      "description": "Slide Description 1",
      "type": ["content"]
    },
    {
      "title": "Slide Title 2",
      "description": "Slide Description 2",
      "type": ["section", "content", "content"]
    }
  ]
  `;

  const importantNotes = `
  - When a topic requires more than 3 slides, break it into multiple topic titles (e.g., "JavaScript Basics Part 1" and "JavaScript Basics Part 2").
  - Descriptions will be used to generate the content later, so keep them detailed and specific.
  - Multiple slide types are if [content, code, mermaid] then slide 1 is content, slide 2 is code, slide 3 is mermaid we are just clubbing together things with same title.
  - Descriptions should give entire picture of the slide content, because when generating only this description will be used ignoring the other descriptions.
  `;

  const compactTypes = [
    "title",
    "content",
    "table",
    "content-and-code",
    "mermaid",
    "list",
    "conclusion",
  ];
  const types = Object.entries(slidePrompts)
    .filter(([key, value]) =>
      compact ? compactTypes.includes(key) : value.include,
    )
    .map(([key, value]) => `${key}: ${value.shortDescription}`)
    .join("\n");

  const fullPrompt = `
You are an expert AI course designer. Your task is to generate a detailed course skeleton based on the provided title, description, and user prompt. Keep the course engaging and interactive by using various slide types. This is a crash course, so ensure the content is concise yet informative.

Title: ${title}
Description: ${description}
${userPrompt ? `User Prompt: ${userPrompt}` : ""}

Here are the available slide types and their descriptions:
${types}

Common Rules:
${commonRules}

Important Notes:
${importantNotes}

${
  compact
    ? `For this self-contained lesson, return exactly these three groups in order:
1. ["title", "content"]: introduce a specific learning objective and explain the core concept.
2. ["mermaid", "content"], ["table", "content"], or ["content-and-code", "content"]: choose the best way to show a concrete worked example; give the example's key steps and result in the description.
3. ["list", "conclusion"]: provide a short retrieval exercise and a recap that answers it.
Supply a distinct title and specific teaching brief for every group. Use a single small visual, no external URLs, and avoid hello/thank-you filler.`
    : ""
}

Response Format:
${jsonFormat}
`;

  try {
    const response = await getGeminiResponse(
      fullPrompt,
      true,
      undefined,
      compact,
    );
    const parsed = JSON.parse(response);
    if (
      !Array.isArray(parsed) &&
      parsed?.error &&
      (!compact || parsed.error !== "invalid_output")
    ) {
      throw new Error(String(parsed.message || parsed.error));
    }
    const courseSkeleton: CourseSkeleton[] =
      compact && !Array.isArray(parsed) ? [] : parsed;

    if (Array.isArray(courseSkeleton) && courseSkeleton.length > 0) {
      const firstItem = courseSkeleton[0] as Record<string, unknown>;
      if (firstItem?.error) {
        throw new Error(
          `Generation error: ${firstItem.message || firstItem.error}`,
        );
      }
    }

    if (!Array.isArray(courseSkeleton)) {
      throw new Error("Invalid course skeleton format. Expected an array.");
    }

    if (compact) {
      const brief = [
        [
          "Core concept",
          `Introduce ${title} and explain its key idea in context. ${description}`,
        ],
        [
          "Worked example",
          `Work through one specific example of ${title}. Explain every step and what the result shows.`,
        ],
        [
          "Practice and takeaway",
          `Set a short recall or application task on ${title}, then explain its answer and recap the core idea.`,
        ],
      ];
      const visualTypes = ["table", "content-and-code", "mermaid"] as const;
      const proposed = courseSkeleton[1]?.type;
      const requestedVisual = Array.isArray(proposed)
        ? proposed.find((type) =>
            visualTypes.some((candidate) => candidate === type),
          )
        : undefined;
      const formats: CourseSkeleton["type"][] = [
        ["title", "content"],
        [requestedVisual ?? "table", "content"],
        ["list", "conclusion"],
      ];
      return brief.map(([fallbackTitle, fallbackDescription], index) => {
        const planned = courseSkeleton[index];
        return {
          title:
            typeof planned?.title === "string" && planned.title.trim()
              ? planned.title
              : fallbackTitle,
          description:
            typeof planned?.description === "string" &&
            planned.description.trim()
              ? planned.description
              : fallbackDescription,
          type: formats[index],
        };
      });
    }
    return courseSkeleton;
  } catch (error) {
    console.error("[CourseAgent] Error generating course skeleton:", error);
    throw error;
  }
}

export async function generateCourseSlide(
  title: string,
  description: string,
  type: Array<keyof typeof slidePrompts>,
  userPrompt: string = "",
  fast = false,
): Promise<any> {
  const commonRules = `
  - Only respond with the course slide in JSON format.
  - Do not include any additional text or explanations.
  - Do not wrap the JSON in Markdown fences. Markdown inside slide text is allowed.
  - Ensure the JSON is valid and properly formatted.
  - Create a balanced mix of content and interactive elements.
  - Don't add extra elements in the output json array
  - Number of elements in the output json array should be equal to the number of types in the type array
  - Make sure to use proper JSON formatting and use escaping where necessary.
  - Make sure to close the JSON array with a closing square bracket.
  `;

  const jsonFormat = `
  [
  ${type
    .map((t) => slidePrompts[t]?.format || "{}")
    .join(",\n")
    .split("\n")
    .map((line) => `  ${line}`)
    .join("\n")}
  ]`;

  const purposes = type
    .map((t) => `${t}: ${slidePrompts[t]?.longDescription || ""}`)
    .join("\n");

  const fullPrompt = `
  You are an expert AI course designer. Your task is to generate a detailed course slide based on the provided title, description, and user prompt. Keep the slide engaging and interactive by using various slide types. This is a crash course, so ensure the content is concise yet informative.

  Title: ${title}
  Description: ${description}
  ${userPrompt ? `User Prompt: ${userPrompt}` : ""}

  Type of Slide[i]: ${type.join(", ")}

  Common Rules:
${commonRules}

  Purpose of the slide:
${purposes}

  Response Format:
${jsonFormat}
  `;

  try {
    const response = await getGeminiResponse(fullPrompt, true, undefined, fast);
    const courseSlide = JSON.parse(response);

    if (courseSlide.error) {
      throw new Error(
        `Generation error: ${courseSlide.message || courseSlide.error}`,
      );
    }

    if (!courseSlide || typeof courseSlide !== "object") {
      throw new Error("Invalid course slide format. Expected an object.");
    }

    return courseSlide;
  } catch (error) {
    console.error("[CourseAgent] Error generating course slide:", error);
    throw error;
  }
}

type SlideType = keyof typeof slidePrompts;

function compactSlide(
  value: unknown,
  type: SlideType,
  sectionTitle: string,
  courseTitle: string,
) {
  if (!value || typeof value !== "object") return undefined;
  const raw = value as Record<string, unknown>;
  const slide = {
    ...raw,
    type,
    title:
      typeof raw.title === "string" && raw.title.trim()
        ? raw.title
        : sectionTitle,
  };

  if (type === "title") {
    Object.assign(slide, {
      title: courseTitle,
      author: "Axiom",
      date: new Date().toISOString().slice(0, 10),
    });
  }
  if (type === "mermaid" && typeof raw.diagram === "string") {
    Object.assign(slide, {
      diagram: raw.diagram
        .trim()
        .replace(/^```(?:mermaid)?\s*([\s\S]*?)\s*```$/i, "$1")
        .replace(/^\s*%%(?!\s*\{)[^\n]*$/gm, "")
        .trim()
        .replace(/^graph(?=\s)/i, "flowchart"),
    });
  }

  const result = slideSchema.safeParse(slide);
  return result.success ? result.data : undefined;
}

async function generateCompactSection(
  section: CourseSkeleton,
  courseTitle: string,
  userPrompt: string,
) {
  const generated: unknown = await generateCourseSlide(
    section.title,
    section.description,
    section.type,
    userPrompt,
    true,
  );
  const values = Array.isArray(generated) ? generated : [];

  const slides = await Promise.all(
    section.type.map(async (type, index) => {
      const original = compactSlide(
        values[index],
        type,
        section.title,
        courseTitle,
      );
      if (original) return original;

      const visual =
        type === "mermaid" || type === "table" || type === "content-and-code";
      const replacement = visual ? "table" : type;
      console.warn(`[CourseAgent] Repairing an invalid ${type} slide.`);
      try {
        const revised: unknown = await generateCourseSlide(
          section.title,
          section.description,
          [replacement],
          `${userPrompt} Return just this one slide. ${visual ? "Summarize the worked example in a table. Use the same number of cells in each row as there are headers." : ""}`,
          true,
        );
        const repaired = Array.isArray(revised)
          ? compactSlide(revised[0], replacement, section.title, courseTitle)
          : undefined;
        if (repaired) return repaired;
      } catch (error) {
        if (!visual) throw error;
      }
      // The companion explanation still teaches the worked example.
      if (visual) return undefined;
      throw new Error(`Gemma 4 returned an incomplete ${type} slide.`);
    }),
  );

  return slides.filter((slide) => slide !== undefined);
}

/**
 * Main entry point for generating a complete course.
 * Replaces the old generateCourseComplete.
 */
export async function generateCourseComplete(
  title: string,
  description: string,
  userPrompt: string = "",
  compact = false,
): Promise<CompleteCourse> {
  console.log("[CourseAgent] Generating complete course limit...");

  const skeleton = await generateCourseSkeleton(
    title,
    description,
    userPrompt,
    compact,
  );
  console.log(
    "[CourseAgent] Generated course skeleton items:",
    skeleton.length,
  );

  if (compact) {
    const sections = await Promise.all(
      skeleton.map((section) =>
        generateCompactSection(section, title, userPrompt),
      ),
    );
    return {
      skeleton,
      slides: sections.flat(),
      metadata: { title, description, userPrompt },
    };
  }

  const slidePromises = skeleton.map(async (slide) => {
    console.log("[CourseAgent] Generating slide for:", slide.title);

    // We only pass known valid slide types
    const validTypes = (slide.type || []).filter(
      (t) => slidePrompts[t as keyof typeof slidePrompts],
    );
    if (validTypes.length === 0) return [];

    const courseSlideArray = await generateCourseSlide(
      slide.title,
      slide.description,
      validTypes as Array<keyof typeof slidePrompts>,
      userPrompt,
    );

    if (
      !courseSlideArray ||
      !Array.isArray(courseSlideArray) ||
      courseSlideArray.length === 0
    ) {
      console.warn(`[CourseAgent] No slides generated for: ${slide.title}`);
      return [];
    }

    const slideSet: CourseSlide[] = [];
    courseSlideArray.forEach((s: Record<string, unknown>, idx: number) => {
      if (typeof s === "object" && s !== null) {
        s.type = validTypes[idx] || validTypes[0];
        s.title = slide.title;
        slideSet.push(s as CourseSlide);
      }
    });

    return slideSet;
  });

  const slidesArrays = await Promise.all(slidePromises);
  const slides = slidesArrays.flat();

  return {
    skeleton,
    slides,
    metadata: {
      title,
      description,
      userPrompt,
    },
  };
}
