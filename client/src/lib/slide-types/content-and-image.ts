import { z } from 'zod';
import { commonSlideFields, commonAttributesFormat, BaseSlideType } from './common';

export const contentAndImageSlide: BaseSlideType = {
  include: false,
  shortDescription: "Hybrid slide combining content with supporting image",
  longDescription: `This slide combines textual content with a supporting visual element. It includes content, bullet points, and an image with caption to enhance understanding and break up text-heavy sections.`,
  format:
`{
  "content": "Main content explaining the concept",
  "bullets": [
    "First supporting point",
    "Second supporting point",
    "Third supporting point"
  ],
  "imageUrl": "https://example.com/supporting-image.jpg",
  "imageCaption": "Caption explaining how the image relates to the content",
  ${commonAttributesFormat}
}`,
  schema: z.object({
    content: z.string().min(1, "Content is required"),
    bullets: z.array(z.string().min(1)).optional(),
    imageUrl: z.string().url("Must be a valid URL").or(z.string().startsWith("/", "Must be a valid path")),
    imageCaption: z.string().min(1, "Image caption is required"),
    ...commonSlideFields
  }),
};
