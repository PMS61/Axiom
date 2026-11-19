import { z } from "zod";
import { slidePrompts } from "./slide-prompts";

export const topicSchema = z.string().trim().min(2).max(100);
const text = z.string().trim().min(1);
export const slideSchema = z
  .object({
    type: z.enum([
      "title",
      "content",
      "table",
      "content-and-code",
      "mermaid",
      "list",
      "conclusion",
    ]),
    title: text.max(180),
    script: text.max(1700).optional(),
    content: text.max(2500).optional(),
    bullets: z.array(text.max(440)).max(8).optional(),
    subtitle: text.max(300).optional(),
    author: text.max(100).optional(),
    date: z.string().max(30).optional(),
    headers: z.array(text.max(100)).min(1).max(6).optional(),
    rows: z
      .array(z.array(z.string().max(280)).max(6))
      .max(10)
      .optional(),
    code: text.max(2600).optional(),
    language: text.max(45).optional(),
    diagram: text.max(1500).optional(),
    caption: text.max(420).optional(),
    items: z
      .array(z.object({ text: text.max(440), checked: z.boolean() }))
      .max(8)
      .optional(),
  })
  .superRefine((slide, ctx) => {
    // Axiom's earlier saved lessons contain plain slides without an instructor script.
    if (slide.type === "content" && slide.script === undefined && slide.content)
      return;
    const result = slidePrompts[slide.type].schema.safeParse(slide);
    if (
      !result.success ||
      (slide.type === "table" &&
        slide.rows?.some((row) => row.length !== slide.headers?.length)) ||
      (slide.type === "mermaid" &&
        (!/^flowchart\s+(?:TD|TB|LR|RL|BT)\b/i.test(slide.diagram ?? "") ||
          /%%|<|>|^\s*(click|link)\b/im.test(
            (slide.diagram ?? "").replace(/-->/g, ""),
          )))
    ) {
      ctx.addIssue({ code: "custom", message: `Invalid ${slide.type} slide` });
    }
  });

export const slidesSchema = z.array(slideSchema).min(3).max(10);
export const flashcardSchema = z.object({
  id: z.string().min(1).max(80),
  front: z.string().min(1).max(350),
  back: z.string().min(1).max(1200),
});
export const deckSchema = z.array(flashcardSchema).min(3).max(12);
export type LearningSlide = z.infer<typeof slideSchema>;
export type LearningCard = z.infer<typeof flashcardSchema>;

export const roadmapGoalSchema = z.string().trim().min(3).max(300);
const roadmapNode = z.object({
  id: text.max(100),
  title: text.max(100),
  description: text.max(1200),
  type: z.enum(["course", "assessment"]),
  priority: z.enum(["low", "normal", "high"]),
  sequence: z.number().int().min(0),
  difficulty: z.number().int().min(1).max(10),
  estimatedMinutes: z.number().int().min(15).max(180),
  prerequisites: z.array(text.max(100)).max(10),
  masteryScore: z.number().min(0).max(1),
  subject: text.max(120),
  nodeState: z.enum(["locked", "available", "in_progress", "mastered"]),
});
export const roadmapSchema = z.object({
  id: text.max(64),
  goal: roadmapGoalSchema,
  nodes: z.array(roadmapNode).min(4).max(12),
});
export type LearningRoadmap = z.infer<typeof roadmapSchema>;

export const learningRequestSchema = z.discriminatedUnion("kind", [
  z.object({
    kind: z.literal("lesson"),
    topic: topicSchema,
    context: z.string().trim().max(1200).optional(),
  }),
  z.object({
    kind: z.literal("cards"),
    topic: topicSchema,
    slides: slidesSchema,
  }),
  z.object({
    kind: z.literal("roadmap"),
    topic: roadmapGoalSchema,
    syllabus: z.string().trim().max(2500).optional(),
    deadline: z.iso.date().optional(),
    interests: z.array(topicSchema).max(12).optional(),
    studied: z.array(topicSchema).max(12).optional(),
  }),
]);
