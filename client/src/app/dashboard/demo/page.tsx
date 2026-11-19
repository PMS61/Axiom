import GuestStudyDashboard from "./GuestStudyDashboard";

export const dynamic = "force-dynamic";

export const metadata = {
  title: "Study dashboard | Axiom",
  description:
    "A returning learner’s roadmap, study history, and adaptive reviews.",
};

export default function StudyDashboardPage() {
  const today = new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Kolkata",
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
  }).format(new Date());

  return <GuestStudyDashboard today={today} />;
}
