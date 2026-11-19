import { fetchHackerNewsSignals } from "@/lib/trend-engine/fetchHackerNews";
import { aggregateByTopic, scoreTopics } from "@/lib/trend-engine/scoring";
import { mapTitleToTopic } from "@/lib/trend-engine/topicMapper";
import type { RawSignal } from "@/lib/trend-engine/types";

const spaceTopics: [RegExp, string][] = [
  [/\b(artemis|moon|lunar)\b/i, "Moon exploration"],
  [/\b(mars|martian)\b/i, "Mars exploration"],
  [/\b(rocket|rocket[s]? propulsion|space launch)\b/i, "Rocket propulsion"],
  [
    /\b(jupiter|saturn|uranus|neptune|venus|mercury|planetary)\b/i,
    "Planetary science",
  ],
  [/\b(exoplanet|webb|black hole|astronom|galax|cosmo)/i, "Astronomy"],
  [
    /\b(space station|spaceflight|spacecraft|astronaut|space mission)\b/i,
    "Space exploration",
  ],
];

function plainTitle(text: string) {
  const entities: Record<string, string> = {
    amp: "&",
    quot: '"',
    apos: "'",
    nbsp: " ",
    "#8211": "–",
    "#8212": "—",
    "#8216": "‘",
    "#8217": "’",
    "#8220": "“",
    "#8221": "”",
  };
  return text
    .replace(/<[^>]+>/g, "")
    .replace(/&([a-z]+|#\d+);/gi, (match, code) => entities[code] ?? match);
}

async function fetchNasaSignals(): Promise<RawSignal[]> {
  try {
    const response = await fetch(
      "https://www.nasa.gov/wp-json/wp/v2/posts?per_page=30&_fields=link,date_gmt,title",
      { next: { revalidate: 1800 }, signal: AbortSignal.timeout(7000) },
    );
    if (!response.ok) return [];
    const posts = (await response.json()) as {
      date_gmt: string;
      link: string;
      title: { rendered: string };
    }[];
    return posts.flatMap((post) => {
      if (
        typeof post.title?.rendered !== "string" ||
        typeof post.date_gmt !== "string"
      )
        return [];
      const url = new URL(post.link);
      const timestamp = Date.parse(`${post.date_gmt}Z`);
      if (
        url.protocol !== "https:" ||
        !Number.isFinite(timestamp) ||
        Date.now() - timestamp > 14 * 24 * 60 * 60 * 1000
      )
        return [];
      return [
        {
          title: plainTitle(post.title.rendered),
          source: "news",
          url: url.href,
          engagement: 0,
          publishedAt: new Date(timestamp).toISOString(),
        },
      ];
    });
  } catch {
    return [];
  }
}

export async function GET() {
  const [nasa, hackerNews] = await Promise.all([
    fetchNasaSignals(),
    fetchHackerNewsSignals(),
  ]);
  const mapped = [...nasa, ...hackerNews].flatMap((signal) => {
    const topic =
      spaceTopics.find(([pattern]) => pattern.test(signal.title))?.[1] ??
      (signal.source === "hackernews" ? mapTitleToTopic(signal.title) : null);
    return topic ? [{ signal, topic }] : [];
  });
  const trends = scoreTopics(aggregateByTopic(mapped)).map((entry) => ({
    topic: entry.topic,
    score: entry.trend_score,
    articles: [...entry.raw]
      .sort((a, b) => b.publishedAt.localeCompare(a.publishedAt))
      .filter((signal) => /^https?:\/\//i.test(signal.url))
      .slice(0, 3)
      .map(({ title, url, source, publishedAt }) => ({
        title,
        url,
        source: source === "news" ? "NASA" : "Hacker News",
        publishedAt,
      })),
  }));
  return Response.json(
    { trends },
    {
      headers: {
        "Cache-Control": "public, s-maxage=1800, stale-while-revalidate=3600",
      },
    },
  );
}
