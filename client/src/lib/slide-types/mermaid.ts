import { z } from 'zod';
import { commonSlideFields, commonAttributesFormat, BaseSlideType } from './common';

export const mermaidSlide: BaseSlideType = {
  include: true,
  shortDescription: "Visual diagram slide structure for flowcharts, processes, and relationships",
  longDescription: `This slide renders Mermaid diagrams for flowcharts, graphs, and other visual representations. Include the Mermaid diagram syntax and a caption explaining the diagram.`,
  format:
`{
  "diagram": "flowchart TD\\n    A[Start] --> B{Decision?}\\n    B -->|Yes| C[Action]\\n    B -->|No| D[Alternative]",
  "caption": "Description of what the diagram represents",
  ${commonAttributesFormat}
}`,
  schema: z.object({
    diagram: z.string().min(1, "Mermaid diagram syntax is required"),
    caption: z.string().min(1, "Diagram caption is required"),
    ...commonSlideFields
  }),
};
