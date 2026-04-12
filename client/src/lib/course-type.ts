import { slidePrompts } from "./slide-prompts";

// Roadmap Types
export type RoadmapNode = {
  title: string;
  description: string;
  type: 'course' | 'test';
}

export type CourseSkeleton = {
  title: string;
  description: string;
  type: Array<keyof typeof slidePrompts>;
};

// Course Slide Types
export type CourseSlide = {
  [key: string]: string | string[] | object | boolean | number;
};

// Complete Course Types
export type CompleteCourse = {
  skeleton: CourseSkeleton[];
  slides: CourseSlide[];
  metadata: {
    title: string;
    description: string;
    userPrompt: string;
  };
};

export type assement = {
  question: string;
  options: string[];
  answer: number[];
}