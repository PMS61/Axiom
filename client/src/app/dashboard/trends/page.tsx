/* ═══════════════════════════════════════════════════════════
   Trend Intelligence Dashboard — Server Component
   Pre-fetches trends on the server for fast first paint.
   Client component handles interactivity.
   ═══════════════════════════════════════════════════════════ */

import { fetchTopTrends } from "@/app/actions/trends";
import TrendsDashboard from "@/components/trends/TrendsDashboard";
import type { TrendResult } from "@/lib/trend-engine/types";

export const dynamic = "force-dynamic"; // never cache this page at the Next.js level
export const revalidate = 0;

export default async function TrendsPage() {
  const res = await fetchTopTrends();

  const trends: TrendResult[] = res.trends ?? [];
  const cached = res.cached ?? false;
  const fetchedAt = new Date().toISOString();

  return (
    <TrendsDashboard
      initialTrends={trends}
      cached={cached}
      fetchedAt={fetchedAt}
    />
  );
}
