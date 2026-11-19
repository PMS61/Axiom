import Header from "@/components/Header";
import RoadmapWorkspace from "./RoadmapWorkspace";

export const metadata = { title: "Learning roadmap | Axiom" };

export default async function GuestRoadmapPage({
  searchParams,
}: {
  searchParams: Promise<{ goal?: string | string[] }>;
}) {
  const { goal } = await searchParams;
  return (
    <>
      <Header dashboardDemo />
      <RoadmapWorkspace
        prefill={typeof goal === "string" ? goal.slice(0, 300) : undefined}
      />
    </>
  );
}
