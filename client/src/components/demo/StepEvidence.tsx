"use client";

import online from "@/lib/demo/seed/online.json";
import results from "@/lib/demo/seed/results.json";
import styles from "./evidence.module.css";
import { useDrawn } from "./useDrawn";

type Row = { name: string; ll: number; ours?: boolean; muted?: boolean };

const T = online.final.table;
const BOARD: Row[] = [
  { name: "FSRS-6 (ships in Anki)", ll: T["FSRS-6"].weighted, muted: true },
  { name: "RWKV-Curve", ll: T["RWKV-Curve"].weighted },
  { name: "GBDT alone", ll: T["GBDT (stage 1)"].weighted },
  {
    name: "Ours: GBDT + online layer",
    ll: T["Online layer (ours, final)"].weighted,
    ours: true,
  },
  { name: "RWKV-Instant (2.76M params)", ll: T["RWKV-Instant"].weighted },
];

const fmt = (n: number, d = 3) => n.toFixed(d);

function Leaderboard({ drawn }: { drawn: boolean }) {
  const lo = 0.26;
  const hi = 0.345;
  const W = 560;
  const label = 230;
  const rowH = 38;
  const scale = (v: number) => ((v - lo) / (hi - lo)) * (W - label - 60);
  return (
    <svg
      viewBox={`0 0 ${W} ${BOARD.length * rowH + 26}`}
      className={styles.svg}
      role="img"
      aria-label="Log loss on 1,000 unseen srs-benchmark users"
    >
      {BOARD.map((r, i) => {
        const y = i * rowH + 6;
        const w = scale(r.ll);
        const cls = r.ours
          ? styles.barOurs
          : r.muted
            ? styles.barFsrs
            : styles.barOther;
        return (
          <g key={r.name}>
            <text
              x={label - 10}
              y={y + 19}
              textAnchor="end"
              className={r.ours ? styles.labelOurs : styles.label}
            >
              {r.name}
            </text>
            <rect
              x={label}
              y={y + 4}
              height={rowH - 14}
              width={drawn ? w : 0}
              className={cls}
              style={{ transitionDelay: `${i * 90}ms` }}
            />
            <text
              x={label + (drawn ? w : 0) + 8}
              y={y + 21}
              className={r.ours ? styles.valOurs : styles.val}
              style={{ transitionDelay: `${i * 90}ms` }}
            >
              {fmt(r.ll, 4)}
            </text>
          </g>
        );
      })}
      <text x={label} y={BOARD.length * rowH + 20} className={styles.axis}>
        log loss, lower is better (axis starts at {lo})
      </text>
    </svg>
  );
}

function UserScatter({ drawn }: { drawn: boolean }) {
  const S = 300;
  const pad = 34;
  const lo = 0;
  const hi = 0.7;
  const t = (v: number) => (Math.min(Math.max(v, lo), hi) - lo) / (hi - lo);
  const span = S - pad - 8;
  const X = (v: number) => pad + t(v) * span;
  const Y = (v: number) => S - pad - t(v) * span;
  const users = online.perUser;
  return (
    <svg
      viewBox={`0 0 ${S} ${S}`}
      className={styles.svgSquare}
      role="img"
      aria-label={`Per-user log loss: ours beats FSRS-6 for ${online.winsVsFsrs6} of ${users.length} users`}
    >
      <rect
        x={pad}
        y={8}
        width={S - pad - 8}
        height={S - pad - 8}
        className={styles.frame}
      />
      {/* the win region: below the diagonal (ours < FSRS-6) */}
      <polygon
        points={`${X(lo)},${Y(lo)} ${X(hi)},${Y(hi)} ${X(hi)},${Y(lo)}`}
        className={styles.winZone}
      />
      <text
        x={X(hi) - 6}
        y={Y(lo) - 8}
        textAnchor="end"
        className={styles.zoneLabel}
      >
        ours better
      </text>
      <text x={X(lo) + 6} y={Y(hi) + 16} className={styles.axis}>
        FSRS better
      </text>
      <line
        x1={X(lo)}
        y1={Y(lo)}
        x2={X(hi)}
        y2={Y(hi)}
        className={styles.diag}
      />
      <g className={drawn ? styles.dotsOn : styles.dots}>
        {users.map((u) => (
          <circle
            key={u.user}
            cx={X(u.fsrs6)}
            cy={Y(u.ours)}
            r={1.8}
            className={u.ours < u.fsrs6 ? styles.dotWin : styles.dotLose}
          />
        ))}
      </g>
      <text
        x={S / 2 + pad / 2}
        y={S - 6}
        textAnchor="middle"
        className={styles.axis}
      >
        FSRS-6 log loss →
      </text>
      <text
        x={12}
        y={S / 2}
        textAnchor="middle"
        transform={`rotate(-90 12 ${S / 2})`}
        className={styles.axis}
      >
        ours →
      </text>
    </svg>
  );
}

function Reliability({ drawn }: { drawn: boolean }) {
  const S = 220;
  const pad = 28;
  const x = (v: number) => pad + v * (S - pad - 6);
  const y = (v: number) => S - pad - v * (S - pad - 6);
  const line = (bins: { meanPred: number; meanObs: number }[]) =>
    bins
      .map((b) => `${x(b.meanPred).toFixed(1)},${y(b.meanObs).toFixed(1)}`)
      .join(" ");
  const rel = results.reliability;
  return (
    <svg
      viewBox={`0 0 ${S} ${S}`}
      className={styles.svgSmall}
      role="img"
      aria-label="Calibration"
    >
      <rect
        x={pad}
        y={6}
        width={S - pad - 6}
        height={S - pad - 6}
        className={styles.frame}
      />
      <line x1={x(0)} y1={y(0)} x2={x(1)} y2={y(1)} className={styles.diag} />
      <polyline
        points={line(rel.fsrs_opt)}
        className={`${styles.relFsrs} ${drawn ? styles.relOn : ""}`}
      />
      <polyline
        points={line(rel.gbdt)}
        className={`${styles.relOurs} ${drawn ? styles.relOn : ""}`}
      />
      {rel.gbdt.map((b) => (
        <circle
          key={b.binLow}
          cx={x(b.meanPred)}
          cy={y(b.meanObs)}
          r={2.6}
          className={styles.relDot}
        />
      ))}
      <text
        x={S / 2 + pad / 2}
        y={S - 8}
        textAnchor="middle"
        className={styles.axis}
      >
        predicted →
      </text>
      <text
        x={10}
        y={S / 2}
        textAnchor="middle"
        transform={`rotate(-90 10 ${S / 2})`}
        className={styles.axis}
      >
        actual →
      </text>
    </svg>
  );
}

export default function StepEvidence() {
  const drawn = useDrawn();
  const b = online.final.bootstrap;
  const vsFsrs = b["online - FSRS-6"];
  const vsGbdt = b["online - GBDT (stage 1)"];
  const n = online.perUser.length;
  return (
    <section className={styles.step} aria-labelledby="step-evidence-title">
      <header>
        <p className={styles.kicker}>Step 5 · Evidence</p>
        <h1 id="step-evidence-title" className={styles.headline}>
          Better predictions, on people it has never seen
        </h1>
        <p className={styles.caption}>
          srs-benchmark final test: {n.toLocaleString()} unseen Anki users,
          32.7M reviews, model frozen before it was scored once.
        </p>
      </header>

      <div className={styles.grid}>
        <div className={styles.panel}>
          <p className={styles.panelKicker}>Leaderboard · log loss</p>
          <Leaderboard drawn={drawn} />
          <p className={styles.note}>
            <strong>2nd of all published models.</strong> Beats FSRS-6 by{" "}
            {fmt(-vsFsrs.mean)} (99% CI {fmt(-vsFsrs.hi)}–{fmt(-vsFsrs.lo)}).
          </p>
        </div>

        <div className={styles.panel}>
          <p className={styles.panelKicker}>Every user is a dot</p>
          <div className={styles.scatterRow}>
            <UserScatter drawn={drawn} />
            <div className={styles.bigStat}>
              <span className={styles.bigNum}>{online.winsVsFsrs6}</span>
              <span className={styles.bigOf}>/ {n.toLocaleString()}</span>
              <p>users where ours predicts better than FSRS-6</p>
              <p className={styles.small}>
                Online layer alone helps {online.winsVsGbdt} / {n}: −
                {fmt(-vsGbdt.mean, 4)} log loss
              </p>
            </div>
          </div>
        </div>

        <div className={styles.panel}>
          <p className={styles.panelKicker}>At a fraction of the cost</p>
          <div className={styles.tiles}>
            <div className={styles.tile}>
              <span className={styles.tileNum}>0.09 ms</span>
              <span className={styles.tileVs}>vs 96 ms</span>
              <span className={styles.tileLabel}>per prediction, CPU</span>
            </div>
            <div className={styles.tile}>
              <span className={styles.tileNum}>196k</span>
              <span className={styles.tileVs}>vs 2.76M</span>
              <span className={styles.tileLabel}>stored numbers</span>
            </div>
            <div className={styles.tile}>
              <span className={styles.tileNum}>6</span>
              <span className={styles.tileVs}>weights / learner</span>
              <span className={styles.tileLabel}>
                online layer, starts at GBDT
              </span>
            </div>
          </div>
          <p className={styles.small}>
            Ours vs RWKV-Instant, the only model ahead of us.
          </p>
        </div>

        <div className={styles.panel}>
          <p className={styles.panelKicker}>
            Honest probabilities · our Anki split
          </p>
          <div className={styles.scatterRow}>
            <Reliability drawn={drawn} />
            <div>
              <ul className={styles.legend}>
                <li>
                  <span className={styles.swOurs} /> GBDT (stage 1), ECE{" "}
                  {fmt(results.overall.gbdt.ece)}
                </li>
                <li>
                  <span className={styles.swFsrs} /> FSRS-6 fitted, ECE{" "}
                  {fmt(results.overall.fsrs_opt.ece)}
                </li>
              </ul>
              <p className={styles.small}>
                When it says 70%, about 70% are recalled. Wins for{" "}
                {results.userWins.count} of {results.userWins.total} users on
                this split.
              </p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
