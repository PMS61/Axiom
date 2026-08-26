from app.scoring import confidence_band, score_path
from app.trace import trace

OFFLINE = {"allow_network": False}


def test_exchange_trace_scores_higher_than_the_mixer_branch():
    result = trace("SUSPECT_WALLET_DEMO_1", **OFFLINE)
    exchange_path, mixer_path = result.paths[0], result.paths[1]
    assert exchange_path.confidence > mixer_path.confidence
    assert confidence_band(exchange_path.confidence) == "High"


def test_every_factor_is_explained():
    result = trace("SUSPECT_WALLET_DEMO_1", **OFFLINE)
    factors = result.best_path.confidence_factors
    assert factors, "the score must show its working"
    for factor in factors:
        assert set(factor) == {"factor", "delta", "detail"}
    # The factors add up to the score (before clamping).
    assert round(sum(f["delta"] for f in factors), 1) == result.best_path.confidence


def test_more_hops_lowers_confidence():
    short = trace("SUSPECT_WALLET_DEMO_1", **OFFLINE).best_path
    long = trace("SUSPECT_WALLET_DEMO_5", max_depth=8, **OFFLINE).best_path
    assert long.hops > short.hops
    assert long.confidence < short.confidence


def test_bridge_hop_is_penalised_and_flagged():
    result = trace("0xSuspectWalletDemo3333333333333333333333333", **OFFLINE)
    path = result.best_path
    assert any(f["code"] == "bridge_hop" for f in path.flags)
    assert any(f["factor"] == "Crossed a bridge" and f["delta"] < 0
               for f in path.confidence_factors)


def test_mixer_terminal_is_marked_trail_broken():
    path = trace("SUSPECT_WALLET_DEMO_2", **OFFLINE).best_path
    mixer_flag = next(f for f in path.flags if f["code"] == "mixer_hop")
    assert mixer_flag["title"] == "Trail broken here"
    assert mixer_flag["severity"] == "high"


def test_scores_stay_inside_the_band():
    for wallet in ("SUSPECT_WALLET_DEMO_1", "SUSPECT_WALLET_DEMO_2", "SUSPECT_WALLET_DEMO_5"):
        for path in trace(wallet, **OFFLINE).paths:
            score, _ = score_path(path)
            assert 5.0 <= score <= 99.0
