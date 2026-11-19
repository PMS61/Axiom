import Header from "@/components/Header";
import StudyWorkspace from "./StudyWorkspace";

export const metadata = { title: "Study a topic | Axiom" };

export default async function StudyPage({
  searchParams,
}: {
  searchParams: Promise<{
    topic?: string | string[];
    context?: string | string[];
  }>;
}) {
  const { topic, context } = await searchParams;
  return (
    <>
      <Header dashboardDemo />
      <StudyWorkspace
        startingTopic={typeof topic === "string" ? topic.slice(0, 100) : ""}
        context={
          typeof context === "string" ? context.slice(0, 1200) : undefined
        }
      />
    </>
  );
}
