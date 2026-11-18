"""Probe candidate NEW past-only attributes on top of the current GBDT (Phase 6, task 6.10).

For each validation user in the selection half (8001-8050) we compute, for every scored row, a set of candidate
attributes from the raw review stream (past-only; the current row contributes only what is known before the
answer). Then, cross-fitted BY USER (2 folds within 8001-8050), a small LightGBM with init_score = the global
GBDT margin and ONLY that attribute (plus the margin) is fitted on one fold and scored on the other. The log-loss
improvement over the plain GBDT tells how much signal the attribute adds. Users 8051-8100 are not used here.

Also prints permutation importance of the existing features on the same rows.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from fsrs_benchmark.srsb import run_gbdt as R
from fsrs_benchmark.srsb import val_plus as V

SEL = list(range(8001, 8051))


def candidate_attrs(st, m_all):
    """st: review stream (time order, index = review_th). m_all: GBDT margin per stream row (NaN where unknown).
    Returns a DataFrame of candidate attributes for every stream row, using rows < t only (plus row t's
    known-before-answer fields: elapsed time, state, ids)."""
    n = len(st)
    card = st["card_id"].to_numpy()
    y = st["y"].to_numpy(np.float64)
    rating = st["rating"].to_numpy()
    day = st["day_offset"].to_numpy()
    first = st["is_first"].to_numpy()
    p_model = 1 / (1 + np.exp(-m_all))
    out = {k: np.full(n, np.nan) for k in [
        "elo_user", "elo_card", "elo_p", "card_resid_ewm", "user_resid_ewm20", "user_resid_ewm200",
        "lastday_n", "lastday_again", "lastday_last_rating", "card_first_rating", "card_relapses",
        "again_before_grad", "user_again_last20", "today_pos", "card_last_state"]}
    theta = 0.0
    dcard, ncard = {}, {}
    cres, ures20, ures200 = {}, np.nan, np.nan
    hist = {}            # card -> list of (day, rating, state)
    first_rating, relapses, again_grad, graduated = {}, {}, {}, {}
    last20 = []
    today, today_pos = None, 0
    state = st["state"].to_numpy()
    for i in range(n):
        c, d = card[i], day[i]
        if d != today:
            today, today_pos = d, 0
        # features of row i from the past only
        dc = dcard.get(c, 0.0)
        out["elo_user"][i] = theta
        out["elo_card"][i] = dc
        out["elo_p"][i] = 1 / (1 + np.exp(-(theta - dc)))
        out["card_resid_ewm"][i] = cres.get(c, np.nan)
        out["user_resid_ewm20"][i] = ures20
        out["user_resid_ewm200"][i] = ures200
        h = hist.get(c)
        if h:
            prev_days = [x for x in h if x[0] < d]
            if prev_days:
                ld = prev_days[-1][0]
                same = [x for x in prev_days if x[0] == ld]
                out["lastday_n"][i] = len(same)
                out["lastday_again"][i] = sum(1 for x in same if x[1] == 1)
                out["lastday_last_rating"][i] = same[-1][1]
            out["card_last_state"][i] = h[-1][2]
        out["card_first_rating"][i] = first_rating.get(c, np.nan)
        out["card_relapses"][i] = relapses.get(c, 0) if c in first_rating else np.nan
        out["again_before_grad"][i] = again_grad.get(c, np.nan)
        out["user_again_last20"][i] = (np.mean(last20) if last20 else np.nan)
        out["today_pos"][i] = today_pos
        # update with row i's outcome
        r = rating[i]
        if first[i]:
            first_rating[c] = r
            relapses[c] = 0
            again_grad[c] = 0
            graduated[c] = r > 1
        else:
            if not graduated.get(c, False):
                if r == 1:
                    again_grad[c] = again_grad.get(c, 0) + 1
                else:
                    graduated[c] = True
            elif r == 1:
                relapses[c] = relapses.get(c, 0) + 1
            k = 0.4 / (1 + 0.05 * ncard.get(c, 0))
            p = 1 / (1 + np.exp(-(theta - dc)))
            theta += 0.05 * (y[i] - p)
            dcard[c] = dc - k * (y[i] - p)
            ncard[c] = ncard.get(c, 0) + 1
            if not np.isnan(m_all[i]):
                res = y[i] - p_model[i]
                cres[c] = res if np.isnan(cres.get(c, np.nan)) else 0.6 * cres[c] + 0.4 * res
                ures20 = res if np.isnan(ures20) else 0.9 * ures20 + 0.1 * res
                ures200 = res if np.isnan(ures200) else 0.99 * ures200 + 0.01 * res
            last20.append(1.0 if r == 1 else 0.0)
            if len(last20) > 20:
                last20.pop(0)
        hist.setdefault(c, []).append((d, r, state[i]))
        today_pos += 1
    return pd.DataFrame(out, index=st.index)


def main():
    data = Path("data/anki10k")
    model = lgb.Booster(model_file="results/srsb_beat/sweep_a_l63_side.jsonl.model.txt")
    info = json.loads(Path("results/srsb_beat/sweep_a_l63_side.jsonl.run_info.json").read_text())
    feats = info.get("features") or info.get("feature_cols") or list(V.ALL_COLS)
    cols_idx = [V.ALL_COLS.index(c) for c in feats]
    V._init(str(data), V.ALL_COLS, None)
    cfg, loader = V._W["cfg"], V._W["loader"]
    cache = V.cache_dir_for(data)
    frames = []
    for k, u in enumerate(SEL):
        try:
            ds = loader.load_user_data(u)
            raw = pd.read_parquet(data / "revlogs" / f"user_id={u}")
            st = R.review_stream(raw, cfg.max_seq_len * 2)
            full = V._full_table(u, loader, cfg, data, cache)
        except Exception as e:
            print("skip", u, e, flush=True)
            continue
        keys = ds["review_th"].to_numpy()
        X = full["X"][:, cols_idx]
        m_sc = model.predict(X, raw_score=True)
        m_all = pd.Series(np.nan, index=st.index)
        m_all.loc[keys] = m_sc
        cand = candidate_attrs(st, m_all.to_numpy()).reindex(keys)
        test = full["test"]
        df = pd.DataFrame({"uid": u, "y": full["y"][test], "m": m_sc[test]})
        for c in cand.columns:
            df[c] = cand[c].to_numpy()[test]
        for j, c in enumerate(V.ALL_COLS):
            df["x_" + c] = full["X"][test, j]
        frames.append(df)
        print(f"probe user {u} ({k + 1}/{len(SEL)}), test rows {len(df)}", flush=True)
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet("results/srsb_beat/feature_probe_rows.parquet", index=False)
    users = df["uid"].to_numpy()
    folds = [(np.isin(users, SEL[:25]), np.isin(users, SEL[25:])), (np.isin(users, SEL[25:]), np.isin(users, SEL[:25]))]
    y, m = df["y"].to_numpy(np.float64), df["m"].to_numpy()

    def ll(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))

    base = ll(1 / (1 + np.exp(-m)))
    print(f"\nrows {len(df)}, users {df.uid.nunique()}, plain GBDT log loss (row-weighted) {base:.5f}")
    params = dict(objective="binary", learning_rate=0.05, num_leaves=8, min_data_in_leaf=500, verbose=-1, num_threads=6)
    cands = [c for c in df.columns if c not in ("uid", "y", "m") and not c.startswith("x_")]
    res = {}
    for c in cands + ["ALL_NEW"]:
        Xc = df[cands].to_numpy() if c == "ALL_NEW" else df[[c]].to_numpy()
        p = np.empty(len(df))
        for tr, te in folds:
            b = lgb.train(params, lgb.Dataset(Xc[tr], y[tr], init_score=m[tr]), 150)
            p[te] = 1 / (1 + np.exp(-(m[te] + b.predict(Xc[te], raw_score=True))))
        res[c] = base - ll(p)
    print("\nlog-loss GAIN over plain GBDT when adding the attribute (cross-fitted by user; positive = helps):")
    for c, g in sorted(res.items(), key=lambda kv: -kv[1]):
        print(f"  {c:22s} {g:+.5f}")
    # permutation importance of existing features (on all rows, using the global model)
    Xall = df[["x_" + c for c in V.ALL_COLS]].to_numpy()
    Xm = Xall[:, cols_idx]
    rng = np.random.default_rng(0)
    imp = {}
    for j, c in enumerate(feats):
        Xp = Xm.copy()
        Xp[:, j] = Xp[rng.permutation(len(Xp)), j]
        imp[c] = ll(1 / (1 + np.exp(-model.predict(Xp, raw_score=True)))) - base
    print("\npermutation importance of existing features (log-loss increase when shuffled; top 25):")
    for c, g in sorted(imp.items(), key=lambda kv: -kv[1])[:25]:
        print(f"  {c:22s} {g:+.5f}")
    Path("results/srsb_beat/feature_probe.json").write_text(json.dumps(dict(base=base, gain=res, perm=imp), indent=1))


if __name__ == "__main__":
    sys.exit(main())
