import { z } from 'zod';
import { commonSlideFields, commonAttributesFormat, BaseSlideType } from './common';

export const imageSlide: BaseSlideType = {
  include: false,
  shortDescription: "Image slide with title, image URL, and caption",
  longDescription: `This slide displays an image with supporting text. The image can be from a CDN or local path, and includes a caption for context.`,
  format:
`{
  "imageUrl": "https://example.com/image.jpg or /local-image.svg",
  "caption": "Descriptive caption for the image",
  ${commonAttributesFormat}
}`,
  schema: z.object({
    imageUrl: z.string().url("Must be a valid URL").or(z.string().startsWith("/", "Must be a valid path")),
    caption: z.string().min(1, "Caption is required"),
    ...commonSlideFields
  }),
};
