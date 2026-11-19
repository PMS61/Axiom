"use client";

import { scoreTopic, TOPICS, type Topic } from "@/lib/demo/story";
import styles from "./demo.module.css";
import { useDrawn } from "./useDrawn";

interface Props {
  topicId: string;
  onSelect: (id: string) => void;
}

const SPARK_W = 240;
const SPARK_H = 34;

function sparkPoints(values: number[]): string {
  const n = values.length - 1;
  return values
    .map((v, i) => {
      const x = (i / n) * SPARK_W;
      const y = SPARK_H - 2 - v * (SPARK_H - 4);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(" ");
}

function TopicCard({
  topic,
  selected,
  drawn,
  onSelect,
}: {
  topic: Topic;
  selected: boolean;
  drawn: boolean;
  onSelect: () => void;
}) {
  const bars: { key: string; label: string; value: number }[] = [
    { key: "F", label: "Frequency", value: topic.F },
    { key: "E", label: "Engagement", value: topic.E },
    { key: "R", label: "Recency", value: topic.R },
  ];
  return (
    <button
      type="button"
      className={`${styles.card} ${selected ? styles.cardOn : ""}`}
      aria-pressed={selected}
      onClick={onSelect}
    >
      <div className={styles.cardTop}>
        <span className={styles.tag}>{topic.source}</span>
        <span className={styles.score}>{scoreTopic(topic)}</span>
      </div>
      <p className={styles.topicName}>{topic.name}</p>
      <div className={styles.bars}>
        {bars.map((b) => (
          <div key={b.key} className={styles.barRow}>
            <span>{b.label}</span>
            <span className={styles.barTrack}>
              <span
                className={styles.barFill}
                style={
                  {
                    "--w": `${Math.round(b.value * 100)}%`,
                  } as React.CSSProperties
                }
              />
            </span>
            <span className={styles.barVal}>{b.value.toFixed(2)}</span>
          </div>
        ))}
      </div>
      <svg
        className={styles.spark}
        viewBox={`0 0 ${SPARK_W} ${SPARK_H}`}
        preserveAspectRatio="none"
        role="img"
        aria-label={`Twelve-week trend for ${topic.name}`}
      >
        <polyline
          pathLength={1}
          points={sparkPoints(topic.spark)}
          className={`${styles.sparkLine} ${styles.draw} ${drawn ? styles.on : ""} ${selected ? styles.sparkLineOn : ""}`}
        />
      </svg>
    </button>
  );
}

export default function StepTrends({ topicId, onSelect }: Props) {
  const drawn = useDrawn();
  const ranked = [...TOPICS].sort((a, b) => scoreTopic(b) - scoreTopic(a));
  return (
    <section
      className={`${styles.step} ${drawn ? styles.drawn : ""}`}
      aria-labelledby="step-trends-title"
    >
      <header>
        <p className={styles.kicker}>Step 1 · Trends</p>
        <h1 id="step-trends-title" className={styles.headline}>
          What to learn next
        </h1>
        <p className={styles.caption}>
          Six topics are rising in Riya's feeds. Transformers &amp; attention
          leads on frequency, engagement and recency.
        </p>
      </header>
      <div className={styles.cardGrid}>
        {ranked.map((t) => (
          <TopicCard
            key={t.id}
            topic={t}
            selected={t.id === topicId}
            drawn={drawn}
            onSelect={() => onSelect(t.id)}
          />
        ))}
      </div>
      <p className={styles.formula}>
        score = round(100 × (0.4·F + 0.4·E + 0.2·R)), where F = frequency, E =
        engagement, R = recency. Click a card to select it.
      </p>
    </section>
  );
}
