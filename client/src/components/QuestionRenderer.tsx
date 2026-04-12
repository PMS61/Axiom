import type { ReactNode } from "react";

type QuestionRendererProps = {
  children: ReactNode;
};

export default function QuestionRenderer({ children }: QuestionRendererProps) {
  return <div style={{ whiteSpace: "pre-wrap" }}>{children}</div>;
}
