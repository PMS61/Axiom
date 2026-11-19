import Header from "@/components/Header";
import DiscoverTopics from "./DiscoverTopics";

export const metadata = { title: "Discover topics | Axiom" };

export default function DiscoverPage() {
  return (
    <>
      <Header dashboardDemo />
      <DiscoverTopics />
    </>
  );
}
