import { z } from 'zod';
import { commonSlideFields, commonAttributesFormat, BaseSlideType } from './common';

export const tableSlide: BaseSlideType = {
  include: true,
  shortDescription: "Structured data presentation slide for comparisons and organized information",
  longDescription: `This slide presents structured information in a tabular format. Define headers and rows to organize comparative data, features, specifications, or any structured information.`,
  format:
`{
  "headers": ["Column 1", "Column 2", "Column 3"],
  "rows": [
    ["Row 1 Col 1", "Row 1 Col 2", "Row 1 Col 3"],
    ["Row 2 Col 1", "Row 2 Col 2", "Row 2 Col 3"],
    ["Row 3 Col 1", "Row 3 Col 2", "Row 3 Col 3"]
  ],
  ${commonAttributesFormat}
}`,
  schema: z.object({
    headers: z.array(z.string().min(1)).min(1, "At least one header is required"),
    rows: z.array(z.array(z.string())).min(1, "At least one row is required"),
    ...commonSlideFields
  }),
};
