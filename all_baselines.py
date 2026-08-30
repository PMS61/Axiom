import pickle
import numpy as np
from scipy.optimize import minimize

from data_pipeline import train_test_split_sequences
from dynamic_retention_model import DynamicRetentionModel
from static_retention_model import StaticRetentionModel
from evaluation import evaluate_all, compute_mae, compute_auc


class HalfLifeRegression:
    """
    Duolingo Half-Life Regression (Settles & Meeder, 2016) Baseline.
    Evaluated on the exact same held-out test sequence events.
    """
    def __init__(self, l2_reg: float = 0.1):
        self.l2_reg = l2_reg
        self.weights = None

    def _extract_features(self, sequences):
        X_list, dt_list, y_list = [], [], []

        for seq in sequences:
            if isinstance(seq, dict) and 'r' in seq and 'dt' in seq:
                r_arr = np.atleast_1d(seq['r'])
                dt_arr = np.atleast_1d(seq['dt'])

                right_count, wrong_count = 0, 0

                for i, (dt_val, y_val) in enumerate(zip(dt_arr, r_arr)):
                    x_feat = [1.0, float(i), float(right_count), float(wrong_count)]

                    X_list.append(x_feat)
                    dt_list.append(max(float(dt_val), 1e-4))
                    y_list.append(float(y_val))

                    if y_val == 1:
                        right_count += 1
                    else:
                        wrong_count += 1

        return np.array(X_list), np.array(dt_list), np.array(y_list)

    def fit(self, train_seqs):
        X, dt, y = self._extract_features(train_seqs)
        n_features = X.shape[1]

        def loss_fn(w):
            log2_h = np.clip(X @ w, -10.0, 15.0)
            h = np.power(2.0, log2_h)
            p_pred = np.clip(np.power(2.0, -dt / h), 1e-6, 1.0 - 1e-6)
            return np.mean((p_pred - y) ** 2) + self.l2_reg * np.sum(w ** 2)

        res = minimize(loss_fn, x0=np.zeros(n_features), method='L-BFGS-B')
        self.weights = res.x
        return self

    def evaluate(self, test_seqs):
        """Predicts held-out review probabilities and computes metrics using evaluation.py functions."""
        X, dt, y_true = self._extract_features(test_seqs)
        log2_h = np.clip(X @ self.weights, -10.0, 15.0)
        h = np.power(2.0, log2_h)
        p_pred = np.clip(np.power(2.0, -dt / h), 0.0, 1.0)

        mae = compute_mae(y_true, p_pred)
        auc = compute_auc(y_true, p_pred)

        # Late predictions (second half of each test sequence)
        late_preds, late_truths = [], []
        for test_s in test_seqs:
            X_s, dt_s, y_s = self._extract_features([test_s])
            if len(y_s) == 0:
                continue
            log2_h_s = np.clip(X_s @ self.weights, -10.0, 15.0)
            h_s = np.power(2.0, log2_h_s)
            p_s = np.clip(np.power(2.0, -dt_s / h_s), 0.0, 1.0)

            half = len(y_s) // 2
            late_preds.extend(p_s[half:])
            late_truths.extend(y_s[half:])

        late_auc = compute_auc(late_truths, late_preds)

        return {
            'MAE': mae,
            'AUC': auc,
            'late_AUC': late_auc
        }


if __name__ == '__main__':
    pickle_path = "processed_sequences.pkl"
    print(f"=== Loading preprocessed sequences from '{pickle_path}' ===")

    with open(pickle_path, "rb") as f:
        data = pickle.load(f)

    sequences = data["sequences"]
    print(f"Loaded {len(sequences)} sequence traces.")

    # Train/Test split identical to convergence_check.py
    train_seqs, test_seqs = train_test_split_sequences(
        sequences, test_frac_per_seq=0.25, min_train_len=2, random_state=42
    )
    print(f"Train/Test split: {len(train_seqs)} sequence pairs.\n")

    # 1. Fit Static Model
    print("[1/3] Fitting Static Retention Model...")
    static_model = StaticRetentionModel(train_seqs, rng=np.random.default_rng(42))
    static_model.run(n_iter=1000, burn_in=400, thin=2, verbose=False)

    # 2. Fit HLR Baseline
    print("[2/3] Fitting Half-Life Regression (HLR)...")
    hlr_model = HalfLifeRegression(l2_reg=0.1)
    hlr_model.fit(train_seqs)
    hlr_results = hlr_model.evaluate(test_seqs)

    # 3. Fit Dynamic Model
    print("[3/3] Fitting Dynamic Retention Model...")
    dynamic_model = DynamicRetentionModel(train_seqs, rng=np.random.default_rng(42))
    dynamic_model.run(n_iter=1000, burn_in=400, thin=2, verbose=False)

    # Run full evaluation using evaluation.py
    eval_results = evaluate_all(dynamic_model, static_model, train_seqs, test_seqs)

    static_res = eval_results['static']
    dynamic_res = eval_results['dynamic']

    # Print Comparison Table
    print("\n" + "=" * 52)
    print(f"{'Model':<22} {'MAE':>8} {'AUC':>8} {'LateAUC':>8}")
    print("-" * 52)
    print(f"{'Static Model':<22} {static_res['MAE']:>8.4f} {static_res['AUC']:>8.4f} {static_res['late_AUC']:>8.4f}")
    print(f"{'HLR (Duolingo)':<22} {hlr_results['MAE']:>8.4f} {hlr_results['AUC']:>8.4f} {hlr_results['late_AUC']:>8.4f}")
    print(f"{'Dynamic (Ours)':<22} {dynamic_res['MAE']:>8.4f} {dynamic_res['AUC']:>8.4f} {dynamic_res['late_AUC']:>8.4f}")
    print("=" * 52)