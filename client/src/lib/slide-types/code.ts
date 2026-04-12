import { z } from 'zod';
import { commonSlideFields, commonAttributesFormat, BaseSlideType } from './common';

export const codeSlide: BaseSlideType = {
  include: false,
  shortDescription: "Syntax-highlighted code demonstration slide structure for programming examples",
  longDescription: `This slide displays syntax-highlighted code examples. Specify the programming language for proper syntax highlighting and include clean, well-commented code.`,
  format:
`{
  "language": "javascript",
  "code": "function example() {\\n  console.log('Hello, world!');\\n}",
  ${commonAttributesFormat}
}`,
  schema: z.object({
    language: z.string().min(1, "Programming language is required"),
    code: z.string().min(1, "Code content is required"),
    ...commonSlideFields
  }),
};
