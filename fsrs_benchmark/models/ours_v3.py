"""
ours_v3.py -- Phase 7 model: ours_v2 (a separate copy, v2 is untouched) plus optional components.

NEW in v3 (each is a constructor flag, off = identical to v2):
  lex_effects  tuple of {"s0","d0","growth","bias"}: random effects of the LEXEME (item shared across
               users; Duolingo only, needs the optional `lexeme` column; absent on Anki -> no effect).
               Each is tau_l * z_lexeme with a learned scale, prior z ~ N(0,1).
  lag          logit += beta * (2*y_prev - 1), y_prev = outcome of the card's previous review (global beta).
  card_bias    per-card logit offset, learned scale, prior N(0,1) (v1's per-card random effect).
  lex_tau/lex_lr fixed lexeme scale and its learning rate; tau_pen L2 penalty on the user-effect
               scales (the one component that helped: it stops tau growing and overfitting);
               card_tau fixed scale of the per-card logit effect.
  use_dt       forgetting-curve time = dt_days (fractional) when available, else elapsed_days.
Everything below is the v2 docstring.

ours_v2.py -- Phase 6 experimental model: hierarchical Bayesian FSRS-style memory model.

Common interface: fit(events) on rows with is_train True; predict(events) returns a
Series of recall probabilities indexed by event_id for the scored rows only, clipped to
[1e-6, 1-1e-6]. Predictions are one-step-ahead (the prediction at an event uses only
earlier reviews of that card) and all parameters are frozen after fit().

WHAT IS FSRS-INSPIRED (borrowed functional forms, not py-fsrs code):
  - power forgetting curve R(t, S) = (1 + F t / S)^(-decay), F = 0.9^(-1/decay) - 1
  - per-card memory state (stability S, difficulty D), initial values per first grade
  - stability update on recall: S' = S (1 + e^w8 (11-D) S^-w9 (e^{w10 (1-R)} - 1) hard*easy)
    and on lapse: S' = min(S, w11 D^-w12 ((S+1)^w13 - 1) e^{w14 (1-R)}); short-term
    (same-day) update S' = S e^{w17 (g-3+w18)} S^-w19; difficulty update with linear
    damping and mean reversion. The 21 global weights are INITIALISED at the FSRS-6
    defaults but then LEARNED from all users jointly.

WHAT IS BAYESIAN / NEW:
  - hierarchical user random effects (non-centred): each user u has K effects
    e_u = tau * z_u with z_u ~ N(0, I) a priori and a learned global scale tau (per
    effect). Effects shift: initial stability (log), recall-growth (log w8), decay,
    lapse stability (log w11) and a logit bias on the predicted recall. Users with few
    reviews stay near the global (pooled) model; users with lots of data move away.
  - fit by MAP in torch (Adam), sequence replay vectorised over cards: the loop
    is over the review index k (fit replays only the first fit_steps reviews of each card;
    predict replays full histories), with batched tensors over all cards still active at k.
  - optional (variant flags) process noise on log-stability in the predictive (Gaussian
    log-S noise, marginalised by Gauss-Hermite quadrature), and a Platt calibration
    layer fitted on the fit part only.
Cards are keyed by (user_id, card_id).
"""
import math
import time

import numpy as np
import pandas as pd
import torch

from fsrs.scheduler import DEFAULT_PARAMETERS, LOWER_BOUNDS_PARAMETERS, UPPER_BOUNDS_PARAMETERS

P_CLIP = 1e-6
LEX_ALL = ("s0", "d0", "growth", "bias")
EFFECTS_ALL = ("s0", "growth", "decay", "fail", "bias", "d0")
S_MIN, S_MAX = 0.001, 36500.0
GH_X, GH_W = np.polynomial.hermite_e.hermegauss(5)  # probabilists' Hermite
GH_W = GH_W / GH_W.sum()


class _Seq:
    """Cards of a set of users laid out for a loop over the review index.

    Cards are sorted by length (descending), so the cards still active at step k are the
    prefix [:n_k]. Event rows live in one array sorted by (card, review_idx); the k-th
    review of sorted card i is at row start[i] + k.
    """

    def __init__(self, frame, user_index, loss_mask, max_steps=None, card_index=None, lex_index=None):
        df = frame.sort_values(["user_id", "card_id", "review_idx"], kind="stable")
        card_key = df.groupby(["user_id", "card_id"], sort=False).ngroup().to_numpy()
        # groupby with sort=False after sort gives contiguous ids in order of appearance
        n_cards = int(card_key.max()) + 1
        lens = np.bincount(card_key, minlength=n_cards)
        orig_start = np.concatenate([[0], np.cumsum(lens)[:-1]])
        order = np.argsort(-lens, kind="stable")
        self.start = torch.as_tensor(orig_start[order], dtype=torch.long)
        self.lens = lens[order]
        user = df["user_id"].to_numpy()[orig_start]
        self.card_user = torch.as_tensor(user_index(user[order]), dtype=torch.long)
        if card_index is None:
            self.card_id = torch.zeros(len(self.start), dtype=torch.long)
        else:
            cid = df["card_id"].to_numpy()[orig_start]
            self.card_id = torch.as_tensor(card_index(user[order], cid[order]), dtype=torch.long)
        if lex_index is None or "lexeme" not in df.columns:
            self.card_lex = torch.full((len(self.start),), -1, dtype=torch.long)  # -1 -> unseen row
        else:
            self.card_lex = torch.as_tensor(lex_index(df["lexeme"].to_numpy()[orig_start][order]), dtype=torch.long)
        self.gap = torch.as_tensor(np.maximum(df["elapsed_days"].to_numpy(float), 0.0), dtype=torch.float32)
        dt = df["dt_days"].to_numpy(float) if "dt_days" in df.columns else np.full(len(df), np.nan)
        self.dt = torch.as_tensor(np.where(np.isfinite(dt), np.maximum(dt, 0.0), np.maximum(df["elapsed_days"].to_numpy(float), 0.0)), dtype=torch.float32)
        self.grade = torch.as_tensor(df["rating"].to_numpy(np.int64).copy(), dtype=torch.long)
        self.y = torch.as_tensor(df["y"].to_numpy(float), dtype=torch.float32)
        self.loss_flag = torch.as_tensor(loss_mask(df), dtype=torch.bool)
        self.event_id = df["event_id"].to_numpy(np.int64)
        self.scored = df["scored"].to_numpy(bool)
        self.n_steps = int(lens.max()) if max_steps is None else min(int(lens.max()), max_steps)
        # n_active[k] = number of cards with more than k reviews
        self.n_active = [int((self.lens > k).sum()) for k in range(self.n_steps)]
        self.n_rows = len(df)
        self.n_flag = int(self.loss_flag.sum())


# Frozen Phase 7 choice (see docs/reports/phase-7-experiments.md), used by run.py key `ours_v3`.
FROZEN_KWARGS = dict(epochs=20, tau_pen=300.0, lag=True, card_bias=True, card_tau=0.2)   # variant q2


class OursV3:
    def __init__(self, effects=EFFECTS_ALL, epochs=6, lr=0.03, fit_steps=64, chunk_cards=15000,
                 seed=0, process_noise=False, calibrate=False, card_effect=False, z_lr=None, prior_w=0.0, tau_init=0.3,
                 z_prior=1.0, verbose=False,
                 lex_effects=(), lag=False, card_bias=False, use_dt=False, lex_lr=None, lex_tau=None, tau_pen=0.0, card_tau=None, lex_pen=0.0):
        self.effects = tuple(effects)
        self.epochs, self.lr, self.fit_steps = int(epochs), float(lr), int(fit_steps)
        self.chunk_cards, self.seed = int(chunk_cards), int(seed)
        self.process_noise, self.calibrate = bool(process_noise), bool(calibrate)
        self.card_effect = bool(card_effect)
        self.z_lr = None if z_lr is None else float(z_lr)
        self.prior_w, self.tau_init, self.z_prior = float(prior_w), float(tau_init), float(z_prior)
        self.verbose = verbose
        self.lex_effects, self.lag, self.card_bias, self.use_dt = tuple(lex_effects), bool(lag), bool(card_bias), bool(use_dt)
        self.lex_lr = None if lex_lr is None else float(lex_lr)
        self.tau_pen = float(tau_pen)   # L2 penalty (in nats of the whole fit set) on the user-effect scales
        self.card_tau = None if card_tau is None else float(card_tau)   # fixed per-card logit scale (None = learned)
        self.lex_pen = float(lex_pen)   # L2 penalty (nats of the fit set) on the learned lexeme scales
        self.lex_tau = None if lex_tau is None else float(lex_tau)   # fixed lexeme scale (None = learned)
        self.fitted_ = False
        self.fit_seconds_ = None
        self.history_ = []

    def hyperparams(self):
        return dict(effects=list(self.effects), epochs=self.epochs, lr=self.lr, z_lr=self.z_lr,
                    fit_steps=self.fit_steps, chunk_cards=self.chunk_cards, seed=self.seed,
                    process_noise=self.process_noise, calibrate=self.calibrate,
                    card_effect=self.card_effect, prior_w=self.prior_w, tau_init=self.tau_init,
                    z_prior=self.z_prior, lex_effects=list(self.lex_effects), lag=self.lag,
                    card_bias=self.card_bias, use_dt=self.use_dt, lex_lr=self.lex_lr, lex_tau=self.lex_tau,
                    tau_pen=self.tau_pen, card_tau=self.card_tau, lex_pen=self.lex_pen)

    # ----- parameters -------------------------------------------------------------
    def _init_params(self, n_users, n_cards=0, n_lex=0):
        torch.manual_seed(self.seed)
        w0 = np.array(DEFAULT_PARAMETERS, dtype=np.float64)
        self._w0 = torch.as_tensor(w0, dtype=torch.float32)
        self._lo = torch.as_tensor(LOWER_BOUNDS_PARAMETERS, dtype=torch.float32)
        self._hi = torch.as_tensor(UPPER_BOUNDS_PARAMETERS, dtype=torch.float32)
        self.theta = torch.zeros(21, requires_grad=True)          # w = w0 * exp(theta)
        k = len(self.effects)
        self.tau_raw = torch.full((k,), math.log(math.expm1(self.tau_init)), requires_grad=True)
        self.z = torch.zeros(n_users + 1, k, requires_grad=True)   # last row = unseen users (kept 0)
        self.noise_raw = torch.tensor(math.log(math.expm1(0.1)), requires_grad=True)
        self.platt = torch.tensor([1.0, 0.0], requires_grad=True)  # slope, intercept on logit
        self.zc = torch.zeros(n_cards + 1, requires_grad=True)     # per-card difficulty effect (last = unseen)
        self.tauc_raw = torch.tensor(math.log(math.expm1(0.3)), requires_grad=True)
        self.zl = torch.zeros(n_lex + 1, len(LEX_ALL), requires_grad=True)   # lexeme effects (last = unseen)
        self.taul_raw = torch.full((len(LEX_ALL),), math.log(math.expm1(0.3)), requires_grad=True)
        self.zb = torch.zeros(n_cards + 1, requires_grad=True)     # per-card logit offset (last = unseen)
        self.taub_raw = torch.tensor(math.log(math.expm1(0.3)), requires_grad=True)
        self.beta_lag = torch.zeros((), requires_grad=True)

    def _w(self):
        return (self._w0 * torch.exp(self.theta)).clamp(self._lo, self._hi)

    def _effects(self, users, zmask=None):
        """[n_cards, 5] effect values (zeros for effects not enabled)."""
        out = torch.zeros(len(users), 6)
        if not self.effects:
            return out
        tau = torch.nn.functional.softplus(self.tau_raw)
        z = self.z[users]
        e = z * tau
        cols = [EFFECTS_ALL.index(n) for n in self.effects]
        out[:, cols] = e
        return out

    # ----- forward ----------------------------------------------------------------
    def _run(self, seq, grad, want_pred):
        """Loop over the review index. Returns (loss_sum, n_flag, preds[row]) ."""
        w = self._w()
        eff = self._effects(seq.card_user)
        e_s, e_g, e_d, e_f, e_b, e_d0 = eff.unbind(1)
        lx = torch.zeros(len(seq.card_lex), len(LEX_ALL))
        if self.lex_effects:
            cols = [LEX_ALL.index(n) for n in self.lex_effects]
            lidx = torch.where(seq.card_lex < 0, torch.full_like(seq.card_lex, self.zl.shape[0] - 1), seq.card_lex)
            lx[:, cols] = (self.zl[lidx] * (torch.full((len(LEX_ALL),), self.lex_tau) if self.lex_tau is not None else torch.nn.functional.softplus(self.taul_raw)))[:, cols]
        l_s, l_d0, l_g, l_b = lx.unbind(1)
        cb = ((self.card_tau if self.card_tau is not None else torch.nn.functional.softplus(self.taub_raw)) * self.zb[seq.card_id]) if self.card_bias else torch.zeros(len(seq.card_id))
        ce = (torch.nn.functional.softplus(self.tauc_raw) * self.zc[seq.card_id]) if self.card_effect else torch.zeros(len(seq.card_id))
        loss = torch.zeros(())
        preds = torch.full((seq.n_rows,), float("nan")) if want_pred else None
        S = D = None
        sig_noise = torch.nn.functional.softplus(self.noise_raw) if self.process_noise else None
        for k in range(seq.n_steps):
            n = seq.n_active[k]
            idx = seq.start[:n] + k
            g = seq.grade[idx]
            gf = g.float()
            if k == 0:
                S = w[g - 1] * torch.exp(e_s[:n] + l_s[:n])
                S = S.clamp(S_MIN, S_MAX)
                D = (w[4] - torch.exp(w[5] * (gf - 1)) + 1 + e_d0[:n] + l_d0[:n] + ce[:n]).clamp(1.0, 10.0)
                continue
            S, D = S[:n], D[:n]
            t = seq.gap[idx]
            tf = seq.dt[idx] if self.use_dt else t   # time for the forgetting curve; t (whole days) still picks the branch
            decay = (w[20] * torch.exp(e_d[:n])).clamp(0.1, 0.8)
            factor = 0.9 ** (-1.0 / decay) - 1.0
            R = (1 + factor * tf / S) ** (-decay)
            flag = seq.loss_flag[idx]
            if want_pred or flag.any():
                bias = e_b[:n] + l_b[:n] + cb[:n]
                if self.lag:
                    bias = bias + self.beta_lag * (2.0 * seq.y[idx - 1] - 1.0)
                p = self._prob(R, tf, S, decay, factor, bias, sig_noise)
                if want_pred:
                    preds[idx] = p.detach()
                if flag.any():
                    yv = seq.y[idx][flag]
                    pf = p[flag].clamp(1e-6, 1 - 1e-6)
                    loss = loss - (yv * torch.log(pf) + (1 - yv) * torch.log1p(-pf)).sum()
            # ---- memory update ----
            Rc = R.clamp(1e-4, 1.0)
            hard = torch.where(g == 2, w[15], torch.ones_like(w[15]))
            easy = torch.where(g == 4, w[16], torch.ones_like(w[16]))
            inc = (torch.exp(w[8] + e_g[:n] + l_g[:n]) * (11 - D) * S ** (-w[9]) * (torch.exp(w[10] * (1 - Rc)) - 1) * hard * easy)
            S_rec = S * (1 + inc)
            S_fail_long = w[11] * torch.exp(e_f[:n]) * D ** (-w[12]) * ((S + 1) ** w[13] - 1) * torch.exp((1 - Rc) * w[14])
            S_fail_short = S / torch.exp(w[17] * w[18])
            S_fail = torch.minimum(S_fail_long, S_fail_short)
            S_long = torch.where(g == 1, S_fail, S_rec)
            sti = torch.exp(w[17] * (gf - 3 + w[18])) * S ** (-w[19])
            sti = torch.where(g >= 2, sti.clamp(min=1.0), sti)
            S_short = S * sti
            Snew = torch.where(t < 1, S_short, S_long).clamp(S_MIN, S_MAX)
            d0_easy = w[4] - torch.exp(w[5] * 3) + 1
            delta = -(w[6] * (gf - 3))
            d_tmp = D + (10.0 - D) * delta / 9.0
            Dnew = (w[7] * d0_easy + (1 - w[7]) * d_tmp).clamp(1.0, 10.0)
            S, D = Snew, Dnew
        return loss, preds

    def _prob(self, R, t, S, decay, factor, e_b, sig_noise):
        lg = torch.logit(R.clamp(1e-6, 1 - 1e-6))
        if self.process_noise:
            # Gaussian noise on log-S, scaled by how far ahead we predict: logS ~ N(logS, sig^2)
            ps = []
            for x, wq in zip(GH_X, GH_W):
                Sq = S * torch.exp(sig_noise * float(x))
                Rq = (1 + factor * t / Sq) ** (-decay)
                ps.append(float(wq) * torch.sigmoid(torch.logit(Rq.clamp(1e-6, 1 - 1e-6)) + e_b))
            p = torch.stack(ps).sum(0)
            if self.calibrate:
                p = torch.sigmoid(self.platt[0] * torch.logit(p.clamp(1e-6, 1 - 1e-6)) + self.platt[1])
            return p
        lg = lg + e_b
        if self.calibrate:
            lg = self.platt[0] * lg + self.platt[1]
        return torch.sigmoid(lg)

    # ----- fit / predict ----------------------------------------------------------
    @staticmethod
    def _loss_mask(df):
        return ((~df["is_first"].to_numpy(bool)) & (df["elapsed_days"].to_numpy() >= 1)
                & df["is_train"].to_numpy(bool))

    def fit(self, events):
        started = time.perf_counter()
        train = events[events["is_train"].astype(bool)]
        users = np.sort(train["user_id"].unique())
        self.user_ids_ = users
        self._uidx = {int(u): i for i, u in enumerate(users)}
        cards = pd.MultiIndex.from_frame(train[["user_id", "card_id"]].drop_duplicates())
        lex_known = pd.Index(np.sort(train["lexeme"].unique())) if "lexeme" in train.columns else pd.Index([])
        self._init_params(len(users), len(cards), len(lex_known))
        n_lex = len(lex_known)
        self._lex_index = lambda v: np.where(lex_known.get_indexer(v) < 0, n_lex, lex_known.get_indexer(v))
        n_known = len(cards)
        def card_index(u, c):
            idx = cards.get_indexer(pd.MultiIndex.from_arrays([u, c]))
            return np.where(idx < 0, n_known, idx)
        self._card_index = card_index
        unseen = len(users)
        lookup = lambda arr: np.fromiter((self._uidx.get(int(u), unseen) for u in arr), dtype=np.int64, count=len(arr))
        self._lookup = lookup

        # chunks of whole users, fixed composition
        ncards = train.groupby("user_id")["card_id"].nunique()
        chunks, cur, cnt = [], [], 0
        for u, c in ncards.items():
            if cur and cnt + c > self.chunk_cards:
                chunks.append(cur); cur, cnt = [], 0
            cur.append(u); cnt += c
        if cur:
            chunks.append(cur)
        seqs = []
        for us in chunks:
            sub = train[train["user_id"].isin(us)]
            s = _Seq(sub, lookup, self._loss_mask, max_steps=self.fit_steps, card_index=card_index, lex_index=self._lex_index)
            if s.n_flag > 0:
                seqs.append(s)

        zl = self.lr if self.z_lr is None else self.z_lr
        groups = [dict(params=[self.theta], lr=self.lr), dict(params=[self.tau_raw, self.z], lr=zl)]
        if self.card_effect:
            groups.append(dict(params=[self.zc, self.tauc_raw], lr=zl))
        if self.process_noise:
            groups.append(dict(params=[self.noise_raw], lr=self.lr))
        ll = self.lr if self.lex_lr is None else self.lex_lr
        if self.lex_effects and n_lex > 0:
            groups.append(dict(params=[self.zl] + ([] if self.lex_tau is not None else [self.taul_raw]), lr=ll))
        if self.card_bias:
            groups.append(dict(params=[self.zb] + ([] if self.card_tau is not None else [self.taub_raw]), lr=zl))
        if self.lag:
            groups.append(dict(params=[self.beta_lag], lr=self.lr))
        total_flag = sum(s.n_flag for s in seqs)
        opt = torch.optim.Adam(groups)
        rng = np.random.default_rng(self.seed)
        for ep in range(self.epochs):
            tot, nflag = 0.0, 0
            for j in rng.permutation(len(seqs)):
                s = seqs[j]
                opt.zero_grad()
                loss, _ = self._run(s, True, False)
                users_in = torch.unique(s.card_user)
                prior = 0.5 * self.z_prior * (self.z[users_in] ** 2).sum()
                if self.card_effect:
                    prior = prior + 0.5 * self.z_prior * (self.zc[torch.unique(s.card_id)] ** 2).sum()
                if self.card_bias:
                    prior = prior + 0.5 * self.z_prior * (self.zb[torch.unique(s.card_id)] ** 2).sum()
                if self.lex_effects and n_lex > 0:
                    # lexeme prior applied once per epoch overall: each chunk carries its share of the data
                    lcols = [LEX_ALL.index(n) for n in self.lex_effects]
                    zlx = self.zl[torch.unique(s.card_lex[s.card_lex >= 0])][:, lcols]
                    prior = prior + 0.5 * (s.n_flag / total_flag) * (zlx ** 2).sum()
                    if self.lex_pen > 0 and self.lex_tau is None:
                        prior = prior + 0.5 * self.lex_pen * (s.n_flag / total_flag) * (torch.nn.functional.softplus(self.taul_raw)[lcols] ** 2).sum()
                if self.tau_pen > 0:
                    prior = prior + 0.5 * self.tau_pen * (s.n_flag / total_flag) * (torch.nn.functional.softplus(self.tau_raw) ** 2).sum()
                if self.prior_w > 0:
                    prior = prior + 0.5 * self.prior_w * (self.theta ** 2).sum()
                obj = (loss + prior) / s.n_flag
                obj.backward()
                opt.step()
                with torch.no_grad():
                    self.z[-1].zero_()
                    self.zc[-1].zero_()
                    self.zb[-1].zero_()
                    self.zl[-1].zero_()
                tot += float(loss.detach()); nflag += s.n_flag
            self.history_.append(tot / max(nflag, 1))
            if self.verbose:
                print(f"  epoch {ep}: fit log loss {tot / max(nflag, 1):.4f}", flush=True)
        if self.calibrate:
            self._fit_platt(seqs)
        self.fitted_ = True
        self.fit_seconds_ = time.perf_counter() - started
        return self

    def _fit_platt(self, seqs):
        """Calibration layer fitted on FIT rows only (never on validation or test)."""
        self.calibrate = False
        P, Y = [], []
        with torch.no_grad():
            for s in seqs:
                _, preds = self._run(s, False, True)
                m = s.loss_flag & ~torch.isnan(preds)
                P.append(preds[m]); Y.append(s.y[m])
        self.calibrate = True
        P, Y = torch.cat(P), torch.cat(Y)
        lg = torch.logit(P.clamp(1e-6, 1 - 1e-6))
        opt = torch.optim.LBFGS([self.platt], lr=0.5, max_iter=50)
        def closure():
            opt.zero_grad()
            q = torch.sigmoid(self.platt[0] * lg + self.platt[1]).clamp(1e-6, 1 - 1e-6)
            l = -(Y * torch.log(q) + (1 - Y) * torch.log1p(-q)).mean()
            l.backward()
            return l
        opt.step(closure)
        self.platt = self.platt.detach()

    def predict(self, events):
        if not self.fitted_:
            raise RuntimeError("call fit() before predict()")
        out_ids, out_p = [], []
        ncards = events.groupby("user_id")["card_id"].nunique()
        chunks, cur, cnt = [], [], 0
        for u, c in ncards.items():
            if cur and cnt + c > 4 * self.chunk_cards:
                chunks.append(cur); cur, cnt = [], 0
            cur.append(u); cnt += c
        if cur:
            chunks.append(cur)
        with torch.no_grad():
            for us in chunks:
                sub = events[events["user_id"].isin(us)]
                s = _Seq(sub, self._lookup, lambda d: np.zeros(len(d), bool), card_index=self._card_index, lex_index=self._lex_index)
                _, preds = self._run(s, False, True)
                m = s.scored
                out_ids.append(s.event_id[m]); out_p.append(preds.numpy()[m])
        ids = np.concatenate(out_ids); p = np.concatenate(out_p).astype(float)
        if np.isnan(p).any():
            raise RuntimeError("some scored events were not predicted")
        series = pd.Series(p, index=pd.Index(ids, name="event_id"), name="p").sort_index()
        return series.clip(P_CLIP, 1 - P_CLIP)
