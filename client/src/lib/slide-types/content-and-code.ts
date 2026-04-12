import { z } from 'zod';
import { commonSlideFields, commonAttributesFormat, BaseSlideType } from './common';

export const contentAndCodeSlide: BaseSlideType = {
  include: false,
  shortDescription: "Combined explanation and implementation slide structure for technical concepts",
  longDescription: `This slide combines explanatory content with a code snippet, providing context and usage examples. Perfect for technical tutorials where you need both explanation and implementation.`,
  format:
`{
  "content": "Explanation of the code concept with context and usage examples",
  "language": "python",
  "code": "def example_function(param):\\n    \\"\\"\\"Docstring explaining the function.\\"\\"\\"\\n    return f\\"Result: {param}\\"",
  ${commonAttributesFormat}
}`,
  schema: z.object({
    content: z.string().min(1, "Content is required"),
    language: z.string().min(1, "Programming language is required"),
    code: z.string().min(1, "Code content is required"),
    ...commonSlideFields
  }),
};
