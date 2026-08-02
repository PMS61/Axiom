import type {
  CompleteCourse,
  CourseSkeleton,
  CourseSlide,
} from "./course-type";
import { getGeminiResponse } from "./gemini";
import { slidePrompts } from "./slide-prompts";

export async function generateCourseSkeleton(
  title: string,
  description: string,
  userPrompt: string = "",
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

  const jsonFormat = `
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

  const types = Object.entries(slidePrompts)
    .filter(([, value]) => value.include)
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

Response Format:
${jsonFormat}
`;

  try {
    const response = await getGeminiResponse(fullPrompt, true);
    const courseSkeleton: CourseSkeleton[] = JSON.parse(response);

    if (
      !Array.isArray(courseSkeleton) &&
      (courseSkeleton as Record<string, unknown>)?.error
    ) {
      throw new Error(
        String(
          (courseSkeleton as Record<string, unknown>).message ||
            (courseSkeleton as Record<string, unknown>).error,
        ),
      );
    }

    if (Array.isArray(courseSkeleton) && courseSkeleton.length > 0) {
      const firstItem = courseSkeleton[0] as Record<string, unknown>;
      if (firstItem.error) {
        throw new Error(
          `Generation error: ${firstItem.message || firstItem.error}`,
        );
      }
    }

    if (!Array.isArray(courseSkeleton)) {
      throw new Error("Invalid course skeleton format. Expected an array.");
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
): Promise<any> {
  const commonRules = `
  - Only respond with the course slide in JSON format.
  - Do not include any additional text or explanations.
  - Do not include any markdown formatting.
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
    const response = await getGeminiResponse(fullPrompt, true);
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

/**
 * Main entry point for generating a complete course.
 * Replaces the old generateCourseComplete.
 */
export async function generateCourseComplete(
  title: string,
  description: string,
  userPrompt: string = "",
): Promise<CompleteCourse> {
  console.log("[CourseAgent] Generating complete course limit...");

  const skeleton = await generateCourseSkeleton(title, description, userPrompt);
  console.log(
    "[CourseAgent] Generated course skeleton items:",
    skeleton.length,
  );

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
