"use client";

import CardReplay from "./CardReplay";
import LiveQueue from "./LiveQueue";
import styles from "./memory.module.css";

export default function MemoryStep() {
  return (
    <div className={styles.step}>
      <p className={styles.kicker}>Step 4 · Memory engine</p>
      <h2 className={styles.headline}>The memory engine</h2>
      <p className={styles.caption}>
        Real reviews from an unseen srs-benchmark user. Grey is FSRS-6, red is
        ours (GBDT + online layer).
      </p>
      <div className={styles.grid}>
        <CardReplay />
        <LiveQueue />
      </div>
    </div>
  );
}
