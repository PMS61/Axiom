"""
gbdt.py -- Phase 7 task 7.3: a gradient-boosted-trees recall model (LightGBM).

Same interface as the other models: fit(events) / predict(events) -> Series indexed by event_id
for scored rows, clipped to (1e-6, 1-1e-6).

Features of review t use only reviews of the same card and the same user BEFORE t
(`base_features`, numba kernels over the card order and the user order). Statistics aggregated
across rows (the user prior and the lexeme recall rate) come from FIT rows only (is_train and
non-first and elapsed_days >= 1), and for TRAIN rows they are out-of-fold (card folds within a user
for the user prior, user folds for the lexeme), so a row's own label never feeds its own feature.
At predict time the stored full fit-row statistics are used.

Feature groups (ablated on validation): gap, dt, card, fsrs, user, userprior, lexeme.
Not used: history_seen/history_correct, the Anki `state` column.
"""
import math
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from numba import njit

from fsrs.scheduler import DEFAULT_PARAMETERS

P_CLIP = 1e-6
ALL_GROUPS = ("gap", "dt", "card", "fsrs", "user", "userprior", "lexeme")
N_FOLDS = 5
W = np.array(DEFAULT_PARAMETERS, dtype=np.float64)
S_MIN, S_MAX = 0.001, 36500.0

CARD_COLS = ["card_n", "card_lapses", "card_rate", "g1", "g2", "g3", "y1", "y2", "y3",
             "gap1", "gap2", "gap3", "since_first", "f_S", "f_D", "f_R"]
USER_COLS = ["u_n", "u_rate", "u_rate_inf", "u_recent", "u_day_n", "u_since_last"]

GROUP_FEATURES = {
    "gap": ["el", "log_el", "same_day"],
    "dt": ["dt_days"],
    "card": ["card_n", "card_lapses", "card_rate", "g1", "g2", "g3", "y1", "y2", "y3",
             "gap1", "gap2", "gap3", "since_first"],
    "fsrs": ["f_S", "f_D", "f_R", "f_logitR", "f_el_over_S"],
    "user": USER_COLS,
    "userprior": ["up_rate", "up_n"],
    "lexeme": ["lex_rate", "lex_n"],
}


@njit(cache=True)
def _card_kernel(key, gkey, el, rating, yv, day, w):
    """Rows sorted by (card key, time). Past-only per-card features and the FSRS-6 default state.

    Rows of one card with the same `gkey` (raw timestamp; unique per row when there is none) are
    simultaneous: all of them get features from the card state BEFORE the group, and the state is then
    updated with the group's rows in order.
    """
    n = len(key)
    out = np.full((n, 16), np.nan)
    decay = w[20]
    factor = 0.9 ** (-1.0 / decay) - 1.0
    S = 0.0
    D = 0.0
    cnt = 0
    lapses = 0
    ysum = 0.0
    first_day = 0
    g1 = g2 = g3 = -1.0
    y1 = y2 = y3 = -1.0
    e1 = e2 = e3 = -1.0
    a = 0
    while a < n:
        b = a + 1
        while b < n and key[b] == key[a] and gkey[b] == gkey[a]:
            b += 1
        if a == 0 or key[a] != key[a - 1]:
            cnt = 0
            lapses = 0
            ysum = 0.0
            first_day = day[a]
            g1 = g2 = g3 = -1.0
            y1 = y2 = y3 = -1.0
            e1 = e2 = e3 = -1.0
        # features of every row of the group, from the state before the group
        for i in range(a, b):
            out[i, 0] = cnt
            out[i, 1] = lapses
            if cnt > 0:
                out[i, 2] = ysum / cnt
            if g1 >= 0:
                out[i, 3] = g1
                out[i, 6] = y1
                out[i, 9] = e1
            if g2 >= 0:
                out[i, 4] = g2
                out[i, 7] = y2
                out[i, 10] = e2
            if g3 >= 0:
                out[i, 5] = g3
                out[i, 8] = y3
                out[i, 11] = e3
            out[i, 12] = day[i] - first_day
            if cnt > 0:
                t = max(float(el[i]), 0.0)
                out[i, 13] = S
                out[i, 14] = D
                out[i, 15] = (1.0 + factor * t / S) ** (-decay)
        # then update the state with the group's rows, in order
        for i in range(a, b):
            g = rating[i]
            gf = float(g)
            t = float(el[i])
            if cnt == 0:
                S_new = min(max(w[g - 1], S_MIN), S_MAX)
                D_new = min(max(w[4] - math.exp(w[5] * (gf - 1.0)) + 1.0, 1.0), 10.0)
            else:
                if t < 0.0:
                    t = 0.0
                R = (1.0 + factor * t / S) ** (-decay)
                Rc = min(max(R, 1e-4), 1.0)
                hard = w[15] if g == 2 else 1.0
                easy = w[16] if g == 4 else 1.0
                inc = (math.exp(w[8]) * (11.0 - D) * S ** (-w[9]) * (math.exp(w[10] * (1.0 - Rc)) - 1.0)
                       * hard * easy)
                S_rec = S * (1.0 + inc)
                S_fl = (w[11] * D ** (-w[12]) * ((S + 1.0) ** w[13] - 1.0) * math.exp((1.0 - Rc) * w[14]))
                S_fs = S / math.exp(w[17] * w[18])
                S_fail = min(S_fl, S_fs)
                S_long = S_fail if g == 1 else S_rec
                sti = math.exp(w[17] * (gf - 3.0 + w[18])) * S ** (-w[19])
                if g >= 2 and sti < 1.0:
                    sti = 1.0
                S_short = S * sti
                S_new = S_short if t < 1.0 else S_long
                S_new = min(max(S_new, S_MIN), S_MAX)
                d0_easy = w[4] - math.exp(w[5] * 3.0) + 1.0
                delta = -(w[6] * (gf - 3.0))
                d_tmp = D + (10.0 - D) * delta / 9.0
                D_new = min(max(w[7] * d0_easy + (1.0 - w[7]) * d_tmp, 1.0), 10.0)
            S = S_new
            D = D_new
            g3 = g2
            y3 = y2
            e3 = e2
            g2 = g1
            y2 = y1
            e2 = e1
            g1 = gf
            y1 = float(yv[i])
            e1 = float(el[i])
            cnt += 1
            ysum += float(yv[i])
            if g == 1:
                lapses += 1
        a = b
    return out


@njit(cache=True)
def _user_kernel(user, gkey, day, yv, inform, span):
    """Rows sorted by (user, time). Past-only running user statistics.

    Rows of one user with the same `gkey` (the raw timestamp, or day_offset when the dataset has none)
    are simultaneous: every row of the group gets features from the state BEFORE the group, and the
    state is updated with the whole group afterwards.
    """
    n = len(user)
    out = np.full((n, 6), np.nan)
    alpha = 2.0 / (span + 1.0)
    cnt = 0
    ysum = 0.0
    icnt = 0
    isum = 0.0
    ew = 0.0
    ew_w = 0.0
    day_n = 0
    last_day = 0
    a = 0
    while a < n:
        b = a + 1
        while b < n and user[b] == user[a] and gkey[b] == gkey[a]:
            b += 1
        if a == 0 or user[a] != user[a - 1]:
            cnt = 0
            ysum = 0.0
            icnt = 0
            isum = 0.0
            ew = 0.0
            ew_w = 0.0
            day_n = 0
            last_day = day[a]
        if day[a] != last_day:
            day_n = 0
        for i in range(a, b):
            out[i, 0] = cnt
            if cnt > 0:
                out[i, 1] = ysum / cnt
                out[i, 3] = ew / ew_w
                out[i, 5] = day[i] - last_day
            if icnt > 0:
                out[i, 2] = isum / icnt
            out[i, 4] = day_n
        for i in range(a, b):
            cnt += 1
            ysum += yv[i]
            if inform[i]:
                icnt += 1
                isum += yv[i]
            ew = (1.0 - alpha) * ew + alpha * yv[i]
            ew_w = (1.0 - alpha) * ew_w + alpha
            day_n += 1
            last_day = day[i]
        a = b
    return out


def _fold_of(values, n_folds=N_FOLDS):
    return (pd.util.hash_array(np.asarray(values, dtype=np.int64)) % np.uint64(n_folds)).astype(np.int64)


def base_features(events, span=20, user_group="auto"):
    """Past-only features for EVERY row of `events`, a DataFrame aligned with events.index.

    Row t's features depend only on reviews of its card / its user strictly before t (in event_id
    order, which is chronological). Includes the helper columns `_ufold`, `_lfold` used by the
    fit-row aggregates.
    """
    ev = events.sort_values("event_id", kind="stable")
    pos = ev.index
    uid = ev["user_id"].to_numpy(np.int64)
    cid = ev["card_id"].to_numpy(np.int64)
    el = ev["elapsed_days"].to_numpy(np.int64)
    rating = ev["rating"].to_numpy(np.int64)
    yv = ev["y"].to_numpy(np.int64)
    day = ev["day_offset"].to_numpy(np.int64)
    first = ev["is_first"].to_numpy(bool)
    n = len(ev)

    key = uid * (int(cid.max()) + 1) + cid
    order = np.argsort(key, kind="stable")
    # Which rows of a user are simultaneous? "ts": equal raw timestamp (Duolingo logs a whole session at one
    # timestamp). "row": none (Anki: every review is individually timed, and the file order is
    # chronological: day_offset never decreases, and 2.0M same-day consecutive card pairs in 100 users all
    # have elapsed_seconds >= 0 in file order). "day": only strictly earlier days count (safe fallback for
    # batched data without a timestamp). "auto": ts if present, else day (safe default; pass
    # user_group="row" explicitly for Anki, which run.py and the validation runner do).
    mode = user_group
    if mode == "auto":
        mode = "ts" if "ts" in ev.columns else "day"
    gkey = {"ts": lambda: ev["ts"].to_numpy(np.int64), "day": lambda: day,
            "row": lambda: np.arange(n, dtype=np.int64)}[mode]()
    # Card rows are simultaneous only when they share a raw timestamp (a duplicate row of the same user and
    # lexeme in the full Duolingo file); without ts every row is its own group (Anki: per-row).
    ckey = ev["ts"].to_numpy(np.int64) if "ts" in ev.columns else np.arange(n, dtype=np.int64)
    card_out = np.empty((n, 16))
    card_out[order] = _card_kernel(key[order], ckey[order], el[order], rating[order], yv[order], day[order], W)

    inform = (~first) & (el >= 1)
    user_out = _user_kernel(uid, gkey, day, yv.astype(np.float64), inform, float(span))

    f = pd.DataFrame(card_out, columns=CARD_COLS, index=pos)
    f[USER_COLS] = user_out
    f["el"] = np.where(first, np.nan, el.astype(float))
    f["log_el"] = np.log1p(np.clip(el, 0, None)).astype(float)
    f.loc[first, "log_el"] = np.nan
    f["same_day"] = np.where(first, np.nan, (el < 1).astype(float))
    f["dt_days"] = ev["dt_days"].to_numpy(float)
    f["f_logitR"] = np.log(np.clip(f["f_R"], 1e-6, 1 - 1e-6) / (1 - np.clip(f["f_R"], 1e-6, 1 - 1e-6)))
    f["f_el_over_S"] = f["el"].clip(lower=0) / f["f_S"]
    f["_ufold"] = _fold_of(uid)            # user fold (for the lexeme out-of-fold statistics)
    f["_cfold"] = _fold_of(uid * 1000003 + cid)   # card fold within user (for the user prior)
    return f.loc[events.index]


def _sl_mask(df):
    """Scored-like rows: not a first review and elapsed_days >= 1."""
    return (~df["is_first"].to_numpy(bool)) & (df["elapsed_days"].to_numpy() >= 1)


class Aggregates:
    """Fit-row statistics (user prior, lexeme rate); built from FIT rows only."""

    def __init__(self, events, base, m_user=10.0, m_lex=10.0):
        fit = events["is_train"].to_numpy(bool) & _sl_mask(events)
        self.m_user, self.m_lex = m_user, m_lex
        y = events["y"].to_numpy(float)
        self.prior = float(y[fit].mean()) if fit.any() else 0.5
        d = pd.DataFrame({"u": events["user_id"].to_numpy(), "y": y, "f": fit,
                          "uf": base["_cfold"].to_numpy()})
        d = d[d["f"]]
        self.u_tot = d.groupby("u")["y"].agg(["sum", "count"])
        self.u_fold = d.groupby(["u", "uf"])["y"].agg(["sum", "count"])
        self.has_lex = "lexeme" in events.columns
        if self.has_lex:
            d = pd.DataFrame({"l": events["lexeme"].to_numpy(), "y": y, "f": fit,
                              "lf": base["_ufold"].to_numpy()})
            d = d[d["f"]]
            self.l_tot = d.groupby("l")["y"].agg(["sum", "count"])
            self.l_fold = d.groupby(["l", "lf"])["y"].agg(["sum", "count"])

    @staticmethod
    def _lookup(tot, fold, k1, k2, oof):
        s = tot["sum"].reindex(k1).to_numpy(float)
        c = tot["count"].reindex(k1).to_numpy(float)
        s, c = np.nan_to_num(s), np.nan_to_num(c)
        if oof is not None:
            idx = pd.MultiIndex.from_arrays([k1, k2])
            fs = fold["sum"].reindex(idx).to_numpy(float)
            fc = fold["count"].reindex(idx).to_numpy(float)
            fs, fc = np.nan_to_num(fs), np.nan_to_num(fc)
            s = np.where(oof, s - fs, s)
            c = np.where(oof, c - fc, c)
        return s, c

    def attach(self, events, base):
        """Return (up_rate, up_n, lex_rate, lex_n) arrays. Train rows get out-of-fold values."""
        oof = events["is_train"].to_numpy(bool)
        s, c = self._lookup(self.u_tot, self.u_fold, events["user_id"].to_numpy(), base["_cfold"].to_numpy(), oof)
        out = {"up_rate": (s + self.m_user * self.prior) / (c + self.m_user), "up_n": c}
        if self.has_lex and "lexeme" in events.columns:
            s, c = self._lookup(self.l_tot, self.l_fold, events["lexeme"].to_numpy(), base["_ufold"].to_numpy(), oof)
            out["lex_rate"] = (s + self.m_lex * self.prior) / (c + self.m_lex)
            out["lex_n"] = c
        return out


# variant nf_l15_lr02_sm3_nup (lowest mean validation log loss over both datasets among variants that do
# not lose to FSRS-opt on either; see docs/reports/phase-7-experiments.md, Task 7.3)
FROZEN_KWARGS = dict(groups=tuple(g for g in ALL_GROUPS if g != "userprior"), train_rows="nonfirst",
                     num_leaves=15, learning_rate=0.02, m_user=3.0, m_lex=3.0)


class GBDT:
    def __init__(self, groups=ALL_GROUPS, num_leaves=31, learning_rate=0.05, min_data_in_leaf=100,
                 feature_fraction=0.8, lambda_l2=1.0, train_rows="scored_like", holdout_frac=0.15,
                 max_rounds=2000, patience=50, m_user=10.0, m_lex=10.0, span=20, user_group="auto", seed=0, num_threads=4):
        self.user_group = user_group
        self.groups = tuple(groups)
        self.num_leaves, self.learning_rate, self.min_data_in_leaf = num_leaves, learning_rate, min_data_in_leaf
        self.feature_fraction, self.lambda_l2 = feature_fraction, lambda_l2
        self.train_rows, self.holdout_frac = train_rows, holdout_frac
        self.max_rounds, self.patience = max_rounds, patience
        self.m_user, self.m_lex, self.span = m_user, m_lex, span
        self.seed, self.num_threads = seed, num_threads
        self.booster_ = None
        self.agg_ = None

    def hyperparams(self):
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in dict(
            groups=self.groups, num_leaves=self.num_leaves, learning_rate=self.learning_rate,
            min_data_in_leaf=self.min_data_in_leaf, feature_fraction=self.feature_fraction,
            lambda_l2=self.lambda_l2, train_rows=self.train_rows, holdout_frac=self.holdout_frac,
            max_rounds=self.max_rounds, patience=self.patience, m_user=self.m_user, m_lex=self.m_lex,
            span=self.span, user_group=self.user_group, seed=self.seed).items()}

    def feature_names(self, events):
        names = []
        for g in self.groups:
            if g == "lexeme" and "lexeme" not in events.columns:
                continue
            names += GROUP_FEATURES[g]
        return names

    def design(self, events, base, agg):
        """Feature matrix (DataFrame) for the chosen groups; `base` must be base_features(events)."""
        X = base.copy()
        for k, v in agg.attach(events, base).items():
            X[k] = v
        return X[self.feature_names(events)]

    def fit(self, events, base=None):
        self.names_ = self.feature_names(events)
        base = base_features(events, self.span, self.user_group) if base is None else base
        self.agg_ = Aggregates(events, base, self.m_user, self.m_lex)
        X = self.design(events, base, self.agg_)
        is_tr = events["is_train"].to_numpy(bool)
        rows = is_tr & (_sl_mask(events) if self.train_rows == "scored_like" else ~events["is_first"].to_numpy(bool))
        if not rows.any():
            raise ValueError("no TRAIN rows to fit on")
        y = events["y"].to_numpy(float)
        users = events["user_id"].to_numpy(np.int64)
        hold = (pd.util.hash_array(users + 7919 * (self.seed + 1)) % np.uint64(1000)) < np.uint64(int(self.holdout_frac * 1000))
        tr_m, ho_m = rows & ~hold, rows & hold
        if ho_m.sum() < 50 or tr_m.sum() < 50:     # tiny data: no inner holdout
            tr_m, ho_m = rows, None
        params = dict(objective="binary", metric="binary_logloss", num_leaves=self.num_leaves,
                      learning_rate=self.learning_rate, min_data_in_leaf=self.min_data_in_leaf,
                      feature_fraction=self.feature_fraction, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=self.lambda_l2, num_threads=self.num_threads, seed=self.seed,
                      deterministic=True, force_row_wise=True, verbose=-1)
        dtr = lgb.Dataset(X[tr_m], y[tr_m])
        if ho_m is not None:
            dho = lgb.Dataset(X[ho_m], y[ho_m], reference=dtr)
            self.booster_ = lgb.train(params, dtr, self.max_rounds, valid_sets=[dho],
                                      callbacks=[lgb.early_stopping(self.patience, verbose=False)])
        else:
            self.booster_ = lgb.train(params, dtr, min(self.max_rounds, 200))
        self.best_iter_ = self.booster_.best_iteration or self.booster_.current_iteration()
        self.n_train_ = int(tr_m.sum())
        return self

    def predict(self, events, base=None):
        if self.booster_ is None:
            raise RuntimeError("call fit() before predict()")
        base = base_features(events, self.span, self.user_group) if base is None else base
        sc = events["scored"].to_numpy(bool)
        if not sc.any():
            return pd.Series([], index=pd.Index([], name="event_id", dtype=np.int64), name="p", dtype=float)
        sub, sub_base = events[sc], base[sc]
        X = self.design(sub, sub_base, self.agg_)
        p = self.booster_.predict(X[self.names_], num_iteration=self.best_iter_)
        out = pd.Series(p, index=pd.Index(sub["event_id"].to_numpy(), name="event_id"), name="p")
        return out.sort_index().clip(P_CLIP, 1 - P_CLIP)
