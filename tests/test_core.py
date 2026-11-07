"""Small sanity checks of the building blocks (run with: python -m pytest -q)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from flare.energy import EnergyModel  # noqa: E402
from flare.network import Links  # noqa: E402
from flare.predictors import NLMSBank  # noqa: E402
from flare.spatial import SpatialModel  # noqa: E402


def test_dual_prediction_bounds_error_on_lossless_links():
    rng = np.random.default_rng(0)
    x = np.cumsum(rng.normal(0, 0.05, size=(500, 3)), axis=0) + 20
    node, sink = NLMSBank(3), NLMSBank(3)
    node.warm_start(x[:4])
    sink.warm_start(x[:4])
    eps = 0.2
    for t in range(4, 500):
        pn, ps = node.predict(), sink.predict()
        assert np.allclose(pn, ps)                      # copies stay identical
        send = np.abs(x[t] - pn) > eps
        v = np.where(send, x[t], pn)
        assert np.all(np.abs(v - x[t]) <= eps + 1e-12)  # sink error never exceeds eps
        node.advance(v, pn)
        sink.advance(v, ps)


def test_first_order_radio():
    em = EnergyModel()
    assert np.isclose(em.tx(1000, 0.0), 1000 * 50e-9)
    assert em.tx(1000, 20.0) > em.tx(1000, 10.0)
    assert np.isclose(em.rx(1000), 1000 * 50e-9)


def test_etx_routing_reaches_sink():
    prr = np.array([[0, 0, 0], [0.9, 0, 0.9], [0.1, 0.9, 0]])
    L = Links(prr, np.array([[0.0, 0.0], [5.0, 0.0]]))
    nxt, dist = L.shortest_paths(np.array([1, 2]))
    assert nxt[1] == 0 and nxt[2] == 1               # weak direct link avoided
    # penalising relay 1 sends node 2 over its weak direct link instead
    nxt2, _ = L.shortest_paths(np.array([1, 2]), relay_penalty=np.array([100.0, 0.0]))
    assert nxt2[2] == 0


def test_irreplaceability_low_for_redundant_node():
    rng = np.random.default_rng(1)
    base = np.cumsum(rng.normal(0, 0.1, 2000))
    H = np.c_[base, base + rng.normal(0, 0.01, 2000), rng.normal(0, 1.0, 2000) + base]
    pos = np.array([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]])
    sp = SpatialModel(H, pos, k=2, window=2000)
    r = sp.irreplaceability(frozenset([0, 1, 2]), [0, 1, 2])
    assert r[1] < 0.05 < r[2]


def test_bursty_links_keep_the_measured_mean():
    from flare.network import BurstyLinks
    rng = np.random.default_rng(1)
    prr = rng.uniform(0.0, 1.0, size=(6, 6))
    links = Links(prr, rng.uniform(0, 30, size=(5, 2)))
    b = BurstyLinks(links, np.random.default_rng(2), burst_len=5, bad_share=0.2, beta=0.1)
    mean = 0.8 * b.good_f + 0.2 * b.bad_f
    assert np.allclose(mean, links.fwd)                 # long-term PRR unchanged
    assert np.all((b.good_f <= 1) & (b.bad_f >= 0))
    share = np.mean([(b.step(), b.state_f.mean())[1] for _ in range(4000)])
    assert abs(share - 0.2) < 0.02                      # stationary share of bad states


def test_leach_rotation_serves_each_node_once_per_cycle():
    from flare.methods import LEACH, Context
    n, p = 40, 0.1
    pol = LEACH()
    rng = np.random.default_rng(3)
    served = np.zeros(n, int)
    counts = []
    for epoch in range(10):                              # one full cycle of 1/p epochs
        ctx = Context(rng=rng, epoch=epoch, alive=np.ones(n, bool), E=np.ones(n), E0=np.ones(n),
                      links=None, pos=np.zeros((n, 2)), spatial=None, curve=None, grid=None,
                      eps_bar=0.5, p_ch=p, rc=7.0, rate_hist=[], epoch_len=30, em=None)
        ch, _, _ = pol.plan(ctx)
        served += ch
        counts.append(int(ch.sum()))
    assert served.max() == 1                             # nobody serves twice in a cycle
    assert served.sum() == n                             # and, with T(n) = 1 at the end, everybody serves


def test_ideal_links_keep_measured_topology_and_sink():
    rng = np.random.default_rng(4)
    prr = rng.uniform(0.0, 1.0, size=(6, 6))
    prr[prr < 0.3] = 0.0
    pos = rng.uniform(0, 30, size=(5, 2))
    a, b = Links(prr, pos), Links(prr, pos, ideal=True)
    assert np.array_equal(a.usable, b.usable)
    assert np.allclose(a.xy, b.xy)
    assert np.all(b.fwd[b.usable] == 1.0)
