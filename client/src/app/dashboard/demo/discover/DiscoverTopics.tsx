"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { updateLibrary, useLibrary } from "@/lib/learning-library";
import s from "../learning.module.css";

type Trend = {
  topic: string;
  score: number;
  articles: {
    title: string;
    url: string;
    source: string;
    publishedAt: string;
  }[];
};

const ignored = new Set(["the", "and", "for", "new", "with", "from"]);
function words(value: string) {
  return new Set(
    value
      .toLowerCase()
      .match(/[a-z0-9]+/g)
      ?.filter((word) => word.length > 2 && !ignored.has(word)) ?? [],
  );
}

function match(topic: Trend, reference: string) {
  const candidate = words(
    [topic.topic, ...topic.articles.map(({ title }) => title)].join(" "),
  );
  return [...words(reference)].filter((word) => candidate.has(word)).length;
}

export default function DiscoverTopics() {
  const library = useLibrary();
  const [trends, setTrends] = useState<Trend[]>();
  const [retry, setRetry] = useState(0);
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState("");

  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/learning/trends?retry=${retry}`, { signal: controller.signal })
      .then((response) => (response.ok ? response.json() : Promise.reject()))
      .then((data: { trends: Trend[] }) => setTrends(data.trends))
      .catch(() => {
        if (!controller.signal.aborted) setTrends([]);
      });
    return () => controller.abort();
  }, [retry]);

  const ranked = trends
    ?.map((trend) => {
      const current = match(trend, library.activeTopic);
      const interest = library.interests.find((item) => match(trend, item) > 0);
      return {
        ...trend,
        priority:
          trend.score / 100 +
          3 * current +
          (interest ? 2 * match(trend, interest) : 0),
        reason: current
          ? `Related to ${library.activeTopic}`
          : interest
            ? `Matches your interest in ${interest}`
            : "",
      };
    })
    .sort((a, b) => b.priority - a.priority)
    .slice(0, 8);

  return (
    <main className={`container ${s.learningPage}`}>
      <header className={s.pageHead}>
        <h1>Discover what to learn</h1>
        <p>
          Recent topics ranked around what you’re studying and what you care
          about.
        </p>
      </header>
      <section className={s.panel} aria-label="Learning interests">
        <p>
          Currently studying <strong>{library.activeTopic}</strong>
        </p>
        <p>
          Your interests · {library.interests.join(", ") || "None selected"}
        </p>
        {editing ? (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              const interests = [
                ...new Set(
                  draft
                    .split(",")
                    .map((item) => item.trim())
                    .filter((item) => item.length >= 2 && item.length <= 100),
                ),
              ].slice(0, 12);
              updateLibrary((previous) => ({ ...previous, interests }));
              setEditing(false);
            }}
          >
            <label htmlFor="interests">Interests, separated by commas</label>
            <input
              id="interests"
              maxLength={600}
              value={draft}
              onChange={(event) => setDraft(event.target.value)}
            />
            <button className="btn btn-primary" type="submit">
              Save interests
            </button>
          </form>
        ) : (
          <button
            className="btn btn-sm"
            type="button"
            onClick={() => {
              setDraft(library.interests.join(", "));
              setEditing(true);
            }}
          >
            Edit interests
          </button>
        )}
      </section>

      {!ranked && <output>Finding recent topics…</output>}
      {ranked?.length === 0 && (
        <div className={s.panel}>
          <p>Live sources are temporarily unavailable.</p>
          <button
            className="btn"
            type="button"
            onClick={() => {
              setTrends(undefined);
              setRetry(retry + 1);
            }}
          >
            Retry
          </button>
        </div>
      )}
      {ranked?.map((trend) => (
        <article key={trend.topic} className={s.trend}>
          <div>
            <h2>{trend.topic}</h2>
            {trend.reason && <p className="meta-text">{trend.reason}</p>}
            {trend.articles.map((article) => (
              <p key={article.url}>
                <a href={article.url} target="_blank" rel="noreferrer">
                  {article.title}
                </a>
                {" · "}
                {article.source}
              </p>
            ))}
          </div>
          <Link
            href={`/dashboard/demo/study?topic=${encodeURIComponent(trend.topic)}`}
            className="btn btn-primary"
          >
            Study this topic
          </Link>
        </article>
      ))}
      <Link className="btn" href="/dashboard/demo/study">
        Study a topic of my own
      </Link>
    </main>
  );
}
