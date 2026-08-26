from app.trace import (
    STATUS_ATTRIBUTED,
    STATUS_DEAD_END,
    STATUS_DEPTH_LIMIT,
    STATUS_NO_ACTIVITY,
    STATUS_TRAIL_BROKEN,
    TERMINAL_EXCHANGE,
    trace,
)

OFFLINE = {"allow_network": False}


def test_clean_trace_reaches_an_exchange():
    result = trace("SUSPECT_WALLET_DEMO_1", **OFFLINE)
    assert result.status == STATUS_ATTRIBUTED
    assert result.attributed_entity == "SampleExchange Alpha"
    assert result.hops == 3
    best = result.best_path
    assert best.terminal_kind == TERMINAL_EXCHANGE
    assert best.nodes[0] == "SUSPECT_WALLET_DEMO_1"
    assert best.edges[0].txid == "tx0001"


def test_branch_into_mixer_is_also_reported():
    result = trace("SUSPECT_WALLET_DEMO_1", **OFFLINE)
    # The wallet split funds: one branch cashes out, the other hits a mixer.
    assert len(result.paths) == 2
    assert result.mixers_seen == ["1MixerPoolXXX9999999999999999999999999"]
    # Best path is ranked first and is the exchange one, not the mixer one.
    assert result.paths[0].is_success
    assert not result.paths[1].is_success


def test_trace_that_dead_ends_at_a_mixer():
    result = trace("SUSPECT_WALLET_DEMO_2", **OFFLINE)
    assert result.status == STATUS_TRAIL_BROKEN
    assert result.attributed_entity is None
    assert result.best_path.entity == "Sample Tumbler Two"


def test_ethereum_trace_through_a_bridge():
    result = trace("0xSuspectWalletDemo3333333333333333333333333", **OFFLINE)
    assert result.status == STATUS_ATTRIBUTED
    assert result.chain == "ethereum"
    assert result.attributed_entity == "SampleExchange Gamma"
    # The bridge is flagged but the trace walked straight through it.
    assert result.bridges_seen == ["0xBridgeContractDDD44444444444444444444444444"]


def test_dead_end_when_funds_stop_moving():
    result = trace("SUSPECT_WALLET_DEMO_4", **OFFLINE)
    assert result.status == STATUS_DEAD_END


def test_depth_limit_stops_the_walk():
    result = trace("SUSPECT_WALLET_DEMO_5", **OFFLINE)
    assert result.status == STATUS_DEPTH_LIMIT
    assert result.best_path.hops == 5
    # Raise the limit and the same wallet reaches its exchange.
    deeper = trace("SUSPECT_WALLET_DEMO_5", max_depth=8, **OFFLINE)
    assert deeper.status == STATUS_ATTRIBUTED
    assert deeper.attributed_entity == "SampleExchange Zeta"
    assert deeper.hops == 7


def test_unknown_wallet_reports_no_activity():
    result = trace("A_WALLET_WE_HAVE_NEVER_SEEN", **OFFLINE)
    assert result.status == STATUS_NO_ACTIVITY
    assert result.paths == []
