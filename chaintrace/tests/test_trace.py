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


def test_wide_graph_is_bounded(monkeypatch):
    """A real busy address fans out into hundreds of branches.

    Streamlit refuses to render more than 500 elements in one container, so the
    engine has to hand back a bounded, ranked set rather than everything it saw.
    """
    from app.config import MAX_NODES, MAX_PATHS
    from app.models import FetchResult, Tx

    def fan_out(address, allow_network=True):
        # Every address pays six new ones - six-way branching at every hop.
        outputs = [f"{address}x{index}" for index in range(6)]
        tx = Tx(
            txid=f"tx{abs(hash(address)) % 99999}",
            timestamp="2026-08-10T14:22:00Z",
            chain="bitcoin",
            inputs=[address],
            outputs=outputs,
            amounts={output: 1000.0 for output in outputs},
        )
        return FetchResult(address=address, chain="bitcoin", transactions=[tx], source="sample")

    monkeypatch.setattr("app.trace.fetch", fan_out)
    result = trace("WIDE_WALLET", max_depth=5, allow_network=False)

    assert len(result.nodes) <= MAX_NODES + 6   # the node cap trips mid-transaction
    assert len(result.paths) <= MAX_PATHS
    assert any("best-ranked" in note for note in result.notes)
    assert any("too wide" in note for note in result.notes)
