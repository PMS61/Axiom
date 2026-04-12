/* ═══════════════════════════════════════════════════════════
   THE AXIOM — Dashboard Page
   Server component: pre-fetches trend signals so the
   TrendsSummary section renders on first paint.
   ═══════════════════════════════════════════════════════════ */

import Dashboard from "@/components/Dashboard";
import { fetchTopTrends } from "@/app/actions/trends";
import type { TrendResult } from "@/lib/trend-engine/types";

export const dynamic = "force-dynamic";
export const revalidate = 0;

export default async function DashboardPage() {
  const res = await fetchTopTrends();
  const trends: TrendResult[] = res.trends ?? [];

  return <Dashboard initialTrends={trends} />;
}
