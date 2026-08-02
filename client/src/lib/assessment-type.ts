export type AssessmentQuestionType = "mcq" | "short_answer" | "long_answer" | "coding";

export interface MCQQuestion {
  id: string;
  type: "mcq";
  question: string;
  options: string[];
  correctOptionIndex: number;
  explanation: string;
}

export interface ShortAnswerQuestion {
  id: string;
  type: "short_answer";
  question: string;
  sampleAnswer: string;
  keywords: string[];
}

export interface LongAnswerQuestion {
  id: string;
  type: "long_answer";
  question: string;
  rubric: string; // Evaluation criteria
}

export interface CodingQuestion {
  id: string;
  type: "coding";
  question: string;
  starterCode: string;
  language: string;
  testCases: Array<{
    input: string;
    expectedOutput: string;
    description?: string;
    isPublic: boolean;
  }>;
}

export type AssessmentQuestion = MCQQuestion | ShortAnswerQuestion | LongAnswerQuestion | CodingQuestion;

export interface Assessment {
  title: string;
  description: string;
  questions: AssessmentQuestion[];
}

export interface AssessmentResult {
  nodeId: string;
  score: number; // 0.0 - 1.0
  feedback: string;
  questionResults: Array<{
    questionId: string;
    isCorrect: boolean;
    score: number;
    feedback: string;
  }>;
}
