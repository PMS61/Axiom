# Demo backlog (LOW findings)

## Task 1.1 (export_demo_data.py)
- Honesty card ("fsrs_wins") gap sign is not asserted in pick_cards; assert it.
- Card probabilities clipped to 1e-4 / 4 dp vs overall EPS 1e-6; client tallies differ slightly. Document or align.
- Cryptic names `r4`, `pr`, `honest` → round4, card_preds, honesty_key.
- test_card_predictions_match_parquet re-reads events.parquet in a loop; read once.
- results/full_u100_c1000_final/metrics.csv has duplicate const_global/const_user overall rows (upstream, unused).

## Task 1.1b (export_online_demo.py)
- replayCheck carries raw srs-benchmark user ids (994/574/203); drop `user` or keep it test-only.
- pick_cards has no sign check on the "fsrs_wins" card; assert the gap is negative.
- load_jsonl silently overwrites duplicate users; raise instead.
- build_online_seed calls os.chdir(REPO) as a side effect; remove it.
- Card pFsrsDefault = default-weight FSRS-6 vs per-user-fitted headline; the UI must label it (fsrsNote).
