import { z } from 'zod';
import { commonSlideFields, commonAttributesFormat, BaseSlideType } from './common';

export const titleSlide: BaseSlideType = {
  include: true,
  shortDescription: "Course opening slide structure for title, subtitle, and metadata - use as first slide",
  longDescription: `This slide should contain the main title of the course, a subtitle that provides additional context, the author's name, and the date of creation. It sets the stage for the entire course.`,
  format: 
`{
  "type": "title",
  "title": "Course Title",
  "subtitle": "Subtitle providing additional context",
  "author": "Author Name",
  "date": "YYYY-MM-DD",
  ${commonAttributesFormat}
}`,
  schema: z.object({
    type: z.literal("title"),
    title: z.string().min(1, "Title is required"),
    subtitle: z.string().min(1, "Subtitle is required"),
    author: z.string().min(1, "Author is required"),
    date: z.string().regex(/^\d{4}-\d{2}-\d{2}$/, "Date must be in YYYY-MM-DD format"),
    ...commonSlideFields
  }),
};
