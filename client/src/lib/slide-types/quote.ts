import { z } from 'zod';
import { commonSlideFields, commonAttributesFormat, BaseSlideType } from './common';

export const quoteSlide: BaseSlideType = {
  include: true,
  shortDescription: "Motivational or expert citation slide structure for inspiration and credibility",
  longDescription: `This slide presents an inspirational quote or relevant citation with proper attribution. Use this to add motivation or expert opinions to your course.`,
  format:
`{
  "quote": "The inspirational or relevant quote text",
  "author": "Quote Author Name",
  ${commonAttributesFormat}
}`,
  schema: z.object({
    quote: z.string().min(1, "Quote text is required"),
    author: z.string().min(1, "Quote author is required"),
    ...commonSlideFields
  }),
};
