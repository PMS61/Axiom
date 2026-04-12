import {
  titleSlide,
  sectionSlide,
  contentSlide,
  imageSlide,
  mermaidSlide,
  quoteSlide,
  codeSlide,
  contentAndImageSlide,
  contentAndCodeSlide,
  tableSlide,
  listSlide,
  conclusionSlide
} from './slide-types';

// what are slide?
// Slides are a single unit in our interactive course.

export const slidePrompts = {
  title: titleSlide,
  section: sectionSlide,
  content: contentSlide,
  image: imageSlide,
  mermaid: mermaidSlide,
  quote: quoteSlide,
  code: codeSlide,
  "content-and-image": contentAndImageSlide,
  "content-and-code": contentAndCodeSlide,
  table: tableSlide,
  list: listSlide,
  conclusion: conclusionSlide
};
