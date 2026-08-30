"""
Evaluation harness for Experiment 1 (Section 5.1 of the report): computes
MAE, AUC, and "late-sequence AUC" for the dynamic model vs. the static
baseline on held-out review events.

All metrics are implemented from scratch (no sklearn), as requested.
"""
import numpy as np


def compute_auc(y_true, y_score):
    """
    From-scratch AUC via the Mann-Whitney U statistic:
    AUC = P(score of random positive > score of random negative),
    computed efficiently via rank-sum rather than O(n^2) pairwise comparison.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    if n_pos == 0 or n_neg == 0:
        return np.nan  # undefined AUC with only one class present

    # Rank scores (average rank for ties), 1-indexed
    order = np.argsort(y_score)
    ranks = np.empty(len(y_score))
    ranks[order] = np.arange(1, len(y_score) + 1)

    # Handle ties: average rank within each group of equal scores
    _, inv, counts = np.unique(y_score, return_inverse=True, return_counts=True)
    sum_ranks_per_val = np.zeros(len(counts))
    np.add.at(sum_ranks_per_val, inv, ranks)
    avg_rank_per_val = sum_ranks_per_val / counts
    ranks = avg_rank_per_val[inv]

    rank_sum_pos = ranks[y_true == 1].sum()
    auc = (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return auc


def compute_mae(y_true, y_prob):
    return np.mean(np.abs(np.asarray(y_true) - np.asarray(y_prob)))


def dynamic_model_predict(model, test_seqs, train_seqs):
    """
    For the dynamic model: predicts each held-out review's recall
    probability by propagating the LAST posterior-mean state from the
    matching training prefix forward under the AR(1) transition (no new
    observations used -- a genuine forward forecast, not a fit-in-hindsight
    prediction).
    """
    beta_mean = model.samples['beta'].mean(axis=0)
    gamma_mean = model.samples['gamma'].mean(axis=0)
    phi_mean = model.samples['phi'].mean()

    preds, truths = [], []
    late_preds, late_truths = [], []

    # Map (user,item) -> index in train_seqs / model.psi for last fitted state
    train_index = {(s['user'], s['item']): idx for idx, s in enumerate(train_seqs)}

    for test_s in test_seqs:
        key = (test_s['user'], test_s['item'])
        if key not in train_index:
            continue
        train_idx = train_index[key]
        last_psi = model.psi[train_idx][-1]  # last fitted state from training prefix

        psi_t = last_psi
        prev_r = train_seqs[train_idx]['r'][-1]
        for t in range(len(test_s['r'])):
            x_t = prev_r
            mu_t = beta_mean[test_s['item']] + gamma_mean[test_s['user']] * x_t
            psi_t = phi_mean * psi_t + mu_t  # forward-propagate mean (no noise, point forecast)
            p = 1.0 / (1.0 + np.exp(-psi_t))

            preds.append(p)
            truths.append(test_s['r'][t])
            if t >= len(test_s['r']) // 2:  # later half of the held-out suffix
                late_preds.append(p)
                late_truths.append(test_s['r'][t])

            prev_r = test_s['r'][t]

    return np.array(truths), np.array(preds), np.array(late_truths), np.array(late_preds)


def static_model_predict(model, test_seqs):
    """Static model: same fixed probability for every held-out review of a pair."""
    preds, truths = [], []
    late_preds, late_truths = [], []

    for test_s in test_seqs:
        p = model.predict_proba(test_s['user'], test_s['item'])
        for t in range(len(test_s['r'])):
            preds.append(p)
            truths.append(test_s['r'][t])
            if t >= len(test_s['r']) // 2:
                late_preds.append(p)
                late_truths.append(test_s['r'][t])

    return np.array(truths), np.array(preds), np.array(late_truths), np.array(late_preds)


def evaluate_all(dynamic_model, static_model, train_seqs, test_seqs):
    dyn_truth, dyn_pred, dyn_late_truth, dyn_late_pred = dynamic_model_predict(
        dynamic_model, test_seqs, train_seqs)
    stat_truth, stat_pred, stat_late_truth, stat_late_pred = static_model_predict(
        static_model, test_seqs)

    results = {
        'dynamic': {
            'MAE': compute_mae(dyn_truth, dyn_pred),
            'AUC': compute_auc(dyn_truth, dyn_pred),
            'late_AUC': compute_auc(dyn_late_truth, dyn_late_pred),
        },
        'static': {
            'MAE': compute_mae(stat_truth, stat_pred),
            'AUC': compute_auc(stat_truth, stat_pred),
            'late_AUC': compute_auc(stat_late_truth, stat_late_pred),
        }
    }
    return results


if __name__ == "__main__":
    # Quick self-test of compute_auc against a known-answer case
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.35, 0.8]  # classic textbook example, true AUC = 0.75
    auc = compute_auc(y_true, y_score)
    print(f"AUC self-test: {auc:.3f} (expected 0.750)")
    assert abs(auc - 0.75) < 1e-9, "AUC implementation bug!"
    print("AUC implementation verified correct.")
