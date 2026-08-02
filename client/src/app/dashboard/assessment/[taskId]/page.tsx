"use client";

import { use } from "react";
import Header from "@/components/Header";
import AssessmentView from "@/components/AssessmentView";

export default function AssessmentPage({
  params,
}: {
  params: Promise<{ taskId: string }>;
}) {
  const { taskId } = use(params);

  return (
    <>
      <Header />
      <div style={{ paddingTop: 60, minHeight: "100vh", backgroundColor: "var(--bg)" }}>
        <AssessmentView taskId={taskId} />
      </div>
    </>
  );
}
