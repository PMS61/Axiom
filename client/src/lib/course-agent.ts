import { getGeminiResponse } from "./gemini";
import { slidePrompts } from "./slide-prompts";
import { getRagChunks, formatRagContext } from "./rag-utils";
import type { CompleteCourse, CourseSkeleton, CourseSlide } from "./course-type";

export async function generateCourseSkeleton(
  title: string, 
  description: string, 
  userPrompt: string = "", 
  clusterIds: string[] = []
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
    .join('\n');

  let ragContext = "";
  if (clusterIds.length > 0) {
    try {
      const searchPrompt = `${title} ${description} ${userPrompt}`.trim();
      const ragClusters = clusterIds.map(id => ({ id, prompt: searchPrompt, k: 3 }));
      const ragChunks = await getRagChunks(ragClusters, 3);
      ragContext = await formatRagContext(ragChunks, "Relevant Course Material");
    } catch (error) {
      console.warn("Failed to fetch RAG context:", error);
    }
  }

  const fullPrompt = `
You are an expert AI course designer. Your task is to generate a detailed course skeleton based on the provided title, description, and user prompt. Keep the course engaging and interactive by using various slide types. This is a crash course, so ensure the content is concise yet informative.

Title: ${title}
Description: ${description}
${userPrompt ? `User Prompt: ${userPrompt}` : ""}
${ragContext}

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

    if (Array.isArray(courseSkeleton) && courseSkeleton.length > 0) {
      const firstItem = courseSkeleton[0] as Record<string, unknown>;
      if (firstItem.error) {
        throw new Error(`Generation error: ${firstItem.message || firstItem.error}`);
      }
    }

    if (!Array.isArray(courseSkeleton)) {
      throw new Error("Invalid course skeleton format. Expected an array.");
    }

    return courseSkeleton;
  } catch (error) {
    console.error("[CourseAgent] Error generating course skeleton:", error);
    return generateFallbackSkeleton(title, description);
  }
}

function generateFallbackSkeleton(title: string, description: string): CourseSkeleton[] {
  return [
    {
      title: `Introduction to ${title}`,
      description: `Overview and fundamental concepts of ${title}. ${description}`,
      type: ["title", "content"]
    },
    {
      title: `Core Concepts of ${title}`,
      description: `Deep dive into the main principles and key learning points.`,
      type: ["section", "content", "list"]
    },
    {
      title: `Practical Applications`,
      description: `Real-world examples and use cases demonstrating the practical value.`,
      type: ["content", "list"]
    },
    {
      title: `Summary and Next Steps`,
      description: `Key takeaways and recommendations for continued learning.`,
      type: ["content", "quote"]
    }
  ];
}

export async function generateCourseSlide(
  title: string, 
  description: string, 
  type: Array<keyof typeof slidePrompts>, 
  userPrompt: string = "", 
  clusterIds: string[] = []
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
  ${type.map(t => slidePrompts[t]?.format || "{}").join(',\n').split('\n').map(line => `  ${line}`).join('\n')}
  ]`;

  const purposes = type.map(t => `${t}: ${slidePrompts[t]?.longDescription || ""}`).join('\n');

  let ragContext = "";
  if (clusterIds.length > 0) {
    try {
      const searchPrompt = `${title} ${description} ${userPrompt}`.trim();
      const ragClusters = clusterIds.map(id => ({ id, prompt: searchPrompt, k: 5 }));
      const ragChunks = await getRagChunks(ragClusters, 5);
      ragContext = await formatRagContext(ragChunks, "Relevant Course Material");
    } catch (error) {
      console.warn("Failed to fetch RAG context:", error);
    }
  }

  const fullPrompt = `
  You are an expert AI course designer. Your task is to generate a detailed course slide based on the provided title, description, and user prompt. Keep the slide engaging and interactive by using various slide types. This is a crash course, so ensure the content is concise yet informative.

  Title: ${title}
  Description: ${description}
  ${userPrompt ? `User Prompt: ${userPrompt}` : ""}
  ${ragContext}

  Type of Slide[i]: ${type.join(', ')}

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
      throw new Error(`Generation error: ${courseSlide.message || courseSlide.error}`);
    }

    if (!courseSlide || typeof courseSlide !== "object") {
      throw new Error("Invalid course slide format. Expected an object.");
    }

    return courseSlide;
  } catch (error) {
    console.error("[CourseAgent] Error generating course slide:", error);
    return generateFallbackSlides(title, description, type);
  }
}

function generateFallbackSlides(title: string, description: string, type: Array<keyof typeof slidePrompts>) {
  const fallbackSlides: any[] = [];
  
  for (const slideType of type) {
    switch (slideType) {
      case 'title':
        fallbackSlides.push({
          title: title,
          subtitle: description,
          backgroundImage: "",
          backgroundColor: "#4f46e5"
        });
        break;
      case 'content':
        fallbackSlides.push({
          content: `# ${title}\n\n${description}\n\nThis section covers the fundamental concepts.`,
          bullets: ["Core concepts", "Key objectives"]
        });
        break;
      case 'section':
        fallbackSlides.push({
          title: title,
          subtitle: description,
          icon: "📚"
        });
        break;
      case 'list':
        fallbackSlides.push({
          title: title,
          items: [
            { title: "Key Concept 1", description: "Understanding fundamentals" },
            { title: "Summary", description: description }
          ]
        });
        break;
      case 'code':
        fallbackSlides.push({
          title: title,
          description: description,
          code: `// Example for ${title}\nconsole.log("Learning ${title}");`,
          language: "javascript"
        });
        break;
      case 'quote':
        fallbackSlides.push({
          quote: `"The journey of learning ${title} begins with core principles."`,
          author: "Learning Framework",
          context: description
        });
        break;
      default:
        fallbackSlides.push({
          content: `# ${title}\n\n${description}`,
          bullets: ["Essential concepts covered"]
        });
    }
  }
  
  return fallbackSlides;
}

/**
 * Main entry point for generating a complete course.
 * Replaces the old generateCourseComplete.
 */
export async function generateCourseComplete(
  title: string, 
  description: string, 
  userPrompt: string = "", 
  clusterIds: string[] = []
): Promise<CompleteCourse> {
  console.log("[CourseAgent] Generating complete course limit...");

  const skeleton = await generateCourseSkeleton(title, description, userPrompt, clusterIds);
  console.log("[CourseAgent] Generated course skeleton items:", skeleton.length);

  const slidePromises = skeleton.map(async (slide) => {
    console.log("[CourseAgent] Generating slide for:", slide.title);
    
    // We only pass known valid slide types
    const validTypes = (slide.type || []).filter(t => slidePrompts[t as keyof typeof slidePrompts]);
    if (validTypes.length === 0) return [];

    const courseSlideArray = await generateCourseSlide(
      slide.title, 
      slide.description, 
      validTypes as Array<keyof typeof slidePrompts>, 
      userPrompt, 
      clusterIds
    );
    
    if (!courseSlideArray || !Array.isArray(courseSlideArray) || courseSlideArray.length === 0) {
      console.warn(`[CourseAgent] No slides generated for: ${slide.title}`);
      return [];
    }

    const slideSet: CourseSlide[] = [];
    courseSlideArray.forEach((s: Record<string, unknown>, idx: number) => {
      if (typeof s === 'object' && s !== null) {
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
    }
  };
}