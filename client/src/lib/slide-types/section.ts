import { z } from 'zod';
import { commonSlideFields, commonAttributesFormat, BaseSlideType } from './common';

export const sectionSlide: BaseSlideType = {
  include: true,
  shortDescription: "Section divider slide structure for organizing course into logical segments",
  longDescription: `This slide serves as a section divider to organize the course into logical segments. It contains a section title and a brief description of what will be covered in this section.`,
  format:
`{
  "type": "section",
  "title": "Section Title",
  "description": "A brief description of this section",
  ${commonAttributesFormat}
}`,
  schema: z.object({
    type: z.literal("section"),
    title: z.string().min(1, "Section title is required"),
    description: z.string().min(1, "Section description is required"),
    ...commonSlideFields
  }),
};
