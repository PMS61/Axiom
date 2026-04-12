import { z } from 'zod';
import { commonSlideFields, commonAttributesFormat, BaseSlideType } from './common';

export const contentSlide: BaseSlideType = {
  include: true,
  shortDescription: "Primary information delivery slide structure with text and bullet points",
  longDescription: `This is the main content slide type for delivering information. It includes a title, main content with markdown formatting support, and optional bullet points for key takeaways.`,
  format:
`{
  "type": "content",
  "title": "Slide Title",
  "content": "Main content with **markdown** formatting support",
  "bullets": [
    "First key point",
    "Second key point with **bold** text",
    "Third key point with _italic_ text"
  ],
  ${commonAttributesFormat}
}`,
  schema: z.object({
    type: z.literal("content"),
    title: z.string().min(1, "Title is required"),
    content: z.string().min(1, "Content is required"),
    bullets: z.array(z.string().min(1)).optional(),
    ...commonSlideFields
  }),
};
