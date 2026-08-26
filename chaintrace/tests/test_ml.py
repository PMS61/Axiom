from app.ml import FEATURES, feature_importances, get_model, score_wallet, wallet_features
from app.fetchers import get_transactions


def test_model_trains_and_is_better_than_a_coin_flip():
    _, accuracy, rows = get_model()
    assert rows >= 200
    assert accuracy > 0.7


def test_features_are_computed_for_a_real_wallet():
    txs = get_transactions("SUSPECT_WALLET_DEMO_1")
    features = wallet_features("SUSPECT_WALLET_DEMO_1", txs)
    assert set(features) == set(FEATURES)
    assert features["num_outputs"] == 2.0        # the wallet split its funds
    assert features["total_amount"] == 46000.0   # 42000 + 4000 sats


def test_scoring_returns_a_probability():
    risk = score_wallet("SUSPECT_WALLET_DEMO_1", allow_network=False)
    assert 0.0 <= risk.likelihood <= 1.0
    assert risk.percent == round(risk.likelihood * 100, 1)


def test_wallet_with_no_activity_scores_zero():
    risk = score_wallet("NOT_A_WALLET_WE_KNOW", allow_network=False)
    assert risk.likelihood == 0.0
    assert "No outgoing activity" in risk.note


def test_feature_importances_cover_every_feature():
    importances = feature_importances()
    assert set(importances) == set(FEATURES)
    assert abs(sum(importances.values()) - 1.0) < 0.05
