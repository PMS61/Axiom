import { z } from 'zod';
import { commonSlideFields, commonAttributesFormat, BaseSlideType } from './common';

export const conclusionSlide: BaseSlideType = {
  include: true,
  shortDescription: "Course ending slide structure for summary and next steps - use as final slide",
  longDescription: `This slide serves as the conclusion of the course or section. It typically includes a summary, thank you message, call to action, or invitation for questions.`,
  format:
`{
  "content": "Thank you for your attention! Summary of key takeaways and call to action.",
  ${commonAttributesFormat}
}`,
  schema: z.object({
    content: z.string().min(1, "Conclusion content is required"),
    ...commonSlideFields
  }),
};
