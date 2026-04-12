import { z } from 'zod';
import { commonSlideFields, commonAttributesFormat, BaseSlideType } from './common';

export const listSlide: BaseSlideType = {
  include: true,
  shortDescription: "Interactive checklist slide structure for tracking progress and objectives",
  longDescription: `This slide creates an interactive checklist where items can be marked as checked or unchecked. Useful for learning objectives, prerequisites, or step-by-step processes.`,
  format:
`{
  "type": "list",
  "title": "Checklist Title",
  "items": [
    { "text": "First checklist item", "checked": true },
    { "text": "Second checklist item", "checked": false },
    { "text": "Third checklist item", "checked": true }
  ],
  ${commonAttributesFormat}
}`,
  schema: z.object({
    type: z.literal("list"),
    title: z.string().min(1, "Title is required"),
    items: z.array(z.object({
      text: z.string().min(1, "Item text is required"),
      checked: z.boolean()
    })).min(1, "At least one item is required"),
    ...commonSlideFields
  }),
};
