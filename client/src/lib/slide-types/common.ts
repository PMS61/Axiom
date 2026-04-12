import { z } from 'zod';

/**
 * Common elements shared across all slide types
 */

// Script generation instructions for AI
export const scriptInstructions = "Generate a concise and engaging script for the AI instructor to present this slide. The script should be informative, easy to understand, and suitable for a crash course format. Use a conversational tone and include any necessary explanations or context to help learners grasp the key concepts.";

// Common attributes format for slides
export const commonAttributesFormat = `"script": "${scriptInstructions}"`;

// Common fields that appear in most slide schemas
export const commonSlideFields = {
  script: z.string().min(1, "AI instructor script is required")
} as const;

// Base slide type interface
export interface BaseSlideType {
  include: boolean;
  shortDescription: string;
  longDescription: string;
  format: string;
  schema: z.ZodSchema;
}
