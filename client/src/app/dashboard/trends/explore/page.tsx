/* ═══════════════════════════════════════════════════════════
   Trend Explore — Server Entry
   Reads ?topic query param and passes to the client component.
   ═══════════════════════════════════════════════════════════ */

import { redirect } from "next/navigation";
import ExploreClient from "@/components/trends/ExploreClient";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export default async function ExplorePage({
  searchParams,
}: {
  searchParams: Promise<{ topic?: string }>;
}) {
  const params = await searchParams;
  const topic = (params.topic ?? "").trim();

  if (!topic) redirect("/dashboard/trends");

  return <ExploreClient topic={topic} />;
}
