"""Trace-driven round-based simulator.

One round is one 2-minute sampling period of the Intel Lab evaluation trace.
In each round every live, non-silent node samples its true reading, runs the
dual predictor and sends an update when the prediction error exceeds its
threshold: to its cluster head (clustered policies), which aggregates, or
straight along the ETX routing tree (clustering-free policies). Every hop uses
the measured link probabilities with ARQ, and the sink confirms delivery end
to end. The sink rebuilds a value for every original location: from the
synchronised predictor for nodes that report, and by spatial inference from
the reporting neighbours for nodes that are silent or dead.
"""
from dataclasses import dataclass, field

import numpy as np

from .energy import EnergyModel
from .methods import Context, build
from .network import MAX_TRIES, BurstyLinks, Links
from .predictors import NLMSBank, update_rate_curves
from .spatial import SpatialModel


@dataclass
class Config:
    signal: str = "temp"
    scenario: str = "A"            # A = tuning week, B = held-out week
    eps_bar: float = 0.20          # mean threshold budget (RMS over live nodes)
    E0: float = 0.02               # J per node (scaled so deaths fall inside the trace)
    e0_jitter: float = 0.05        # +-5 % manufacturing / battery spread
    epoch_len: int = 30            # rounds per clustering epoch (1 h)
    p_ch: float = 0.10
    rc: float = 7.0                # cluster radius (m)
    window: int = 180              # rounds in the fidelity window (6 h)
    profile: str = "first_order"
    p_floor: float = 0.0
    ideal_links: bool = False
    k_nb: int = 6
    spatial_window: int = 1440     # rounds (48 h) of sink history for the spatial model
    nlms_p: int = 4
    nlms_mu: float = 0.1
    grid: tuple = tuple(np.round(np.r_[np.arange(0.02, 0.20, 0.01), np.arange(0.20, 1.01, 0.02)], 3))
    extra: dict = field(default_factory=lambda: {"delta_d": 0.35})


class Data:
    """Everything that depends neither on the method nor on the seed."""

    def __init__(self, root, cfg: Config):
        P = root / "data" / "processed"
        s, sc = cfg.signal, cfg.scenario
        self.hist = np.load(P / f"{s}_{sc}_hist.npy").astype(float)     # calibration days
        self.test = np.load(P / f"{s}_{sc}_eval.npy").astype(float)
        self.pos = np.load(P / f"pos_{sc}.npy")
        self.prr = np.load(P / f"prr_{sc}.npy")
        self.ids = np.load(P / f"ids_{sc}.npy")
        self.n = self.test.shape[1]
        self.links = Links(self.prr, self.pos, ideal=cfg.ideal_links, dist_scale=cfg.extra.get("dist_scale", 1.0))
        scale = cfg.extra.get("eps_scale", 1.0)          # thresholds are given in temperature units
        self.grid = np.array(cfg.grid) * scale
        self.curve = update_rate_curves(self.hist, self.grid, p=cfg.nlms_p, mu=cfg.nlms_mu)


def run(data: Data, method: str, seed: int, cfg: Config, keep_series: bool = False):
    rng_link = np.random.default_rng([seed, 1])
    rng_pol = np.random.default_rng([seed, 2])
    rng_e = np.random.default_rng([seed, 3])
    rng_ack = np.random.default_rng([seed, 5])       # only used when end-to-end ACKs can be lost
    ack_loss = bool(cfg.extra.get("ack_loss", False))
    listen_mode = cfg.extra.get("listen", "ch")      # "ch" (default), "none" or "all"
    burst_len = cfg.extra.get("burst_len", 0)
    n, L, X = data.n, data.links, data.test
    T = len(X)
    # optional bursty losses; their own random stream leaves the default runs untouched
    bursts = (BurstyLinks(L, np.random.default_rng([seed, 4]), burst_len,
                          cfg.extra.get("bad_share", 0.2), cfg.extra.get("beta", 0.1))
              if burst_len else None)
    em = EnergyModel(profile=cfg.profile, p_floor=cfg.p_floor)
    pk = em.packets
    policy = build(method)
    spatial = SpatialModel(data.hist, data.pos, k=cfg.k_nb, window=cfg.spatial_window)

    E0 = cfg.E0 * (1 + cfg.e0_jitter * (2 * rng_e.random(n) - 1))
    E = E0.copy()
    alive = np.ones(n, bool)
    known = np.ones(n, bool)
    death = np.full(n, -1)
    ch_count = np.zeros(n, int)

    # energy ledger per node: 0 sensing/CPU, 1 control, 2 own updates,
    # 3 CH listening, 4 CH reception of member updates (and aggregation),
    # 5 CH forwarding of member updates, 6 relaying, 7 end-to-end acknowledgements
    ledger = np.zeros((n, 8))
    who = {"tx": 0, "ctx": 2, "crx": 4, "own": 0.0}

    def spend(idx, joules):
        if idx > 0:
            E[idx - 1] -= joules
            if idx == who["tx"]:
                # a CH aggregate may carry the CH's own reading: that share is its own update
                ledger[idx - 1, 2] += who["own"] * joules
                ledger[idx - 1, who["ctx"]] += (1.0 - who["own"]) * joules
            else:
                ledger[idx - 1, who["crx"]] += joules

    def charge(idx, joules, cat):
        E[idx] -= joules
        ledger[idx, cat] += joules

    node = NLMSBank(n, p=cfg.nlms_p, mu=cfg.nlms_mu)
    hist = data.hist[-cfg.nlms_p:]
    node.warm_start(hist)
    sink = NLMSBank(n, p=cfg.nlms_p, mu=cfg.nlms_mu)
    sink.warm_start(hist)
    span = data.hist.max(axis=0) - data.hist.min(axis=0)
    for b in (node, sink):
        b.set_range(data.hist.min(axis=0) - span, data.hist.max(axis=0) + span)
    cyc = node.cycles()

    ch = np.zeros(n, bool)
    head_of = np.full(n, -1)
    route = {}
    suspect = np.zeros(n, bool)
    relay_pen, nxt = None, None
    eps = np.full(n, cfg.eps_bar)
    silent = np.zeros(n, bool)        # live nodes told not to report this epoch
    valid = np.ones(n, bool)          # sink copy of the node predictor is in sync
    unconfirmed = np.zeros(n, bool)   # node missed the end-to-end ACK of its last update
    reporting = np.ones(n, bool)
    rate_hist, epoch_updates = [], np.zeros(n)
    err = np.zeros(T)
    est_hist = np.zeros((T, n)) if keep_series else None
    err_dead = np.zeros(T)            # contribution of inferred locations (dead or silent)
    err_live = np.zeros(T)            # contribution of reporting nodes
    silent_series = np.zeros(T, int)
    alive_series = np.zeros(T, int)
    gen = deliv = 0
    tx_bits = 0.0
    epoch = 0

    for t in range(T):
        if bursts is not None and t > 0:
            bursts.step()
        # ------------------------------------------------------------ epoch
        if t % cfg.epoch_len == 0:
            known = alive.copy()
            if t > 0:
                rate_hist.append(epoch_updates.copy())
            epoch_updates[:] = 0
            # status / join / schedule exchange, identical for every method
            for i in np.flatnonzero(alive):
                charge(i, em.tx(pk.control, 10.0) + em.rx(pk.control), 1)
            if t > 0:
                spatial.refit(reporting & known)
            ctx = Context(rng=rng_pol, epoch=epoch, alive=known.copy(), E=np.maximum(E, 0), E0=E0,
                          links=L, pos=data.pos, spatial=spatial, curve=data.curve,
                          grid=data.grid, eps_bar=cfg.eps_bar, p_ch=cfg.p_ch, rc=cfg.rc,
                          rate_hist=rate_hist, epoch_len=cfg.epoch_len, em=em, cfg=cfg.extra)
            if known.any():
                ch, eps, pen = policy.plan(ctx)
            else:
                ch, pen = np.zeros(n, bool), None
            ch &= known
            silent = known & ~np.isfinite(eps)
            valid &= ~silent                     # silent nodes leave the predictor sync
            eps = np.where(np.isfinite(eps), eps, data.grid[-1])
            ch_count += ch
            heads = np.flatnonzero(ch)
            suspect = np.zeros(n, bool)
            relay_pen = None if pen is None else np.asarray(pen, float)
            nxt, _ = L.shortest_paths(np.flatnonzero(known) + 1, relay_pen)
            head_of[:] = -1
            for i in np.flatnonzero(known & ~ch):
                if len(heads):
                    etx = L.etx[i + 1, heads + 1]
                    j = int(np.argmin(etx))
                    if np.isfinite(etx[j]):
                        head_of[i] = heads[j]
            route = {}
            for h in heads:                                   # CH schedule broadcast + joins
                m = int((head_of == h).sum())
                charge(h, m * em.rx(pk.control) + em.tx(pk.control, 10.0), 1)
            epoch += 1

        # ------------------------------------------------------------ sensing
        x = X[t]
        sensing = alive & ~silent
        charge(sensing, em.cpu(cyc) + em.e_sense, 0)
        charge(alive, em.p_floor, 0)
        pn = node.predict()
        # a node that resumes reporting must first re-synchronise with the sink
        send = sensing & (~valid | unconfirmed | (np.abs(x - pn) > eps))
        gen += int(send.sum())
        epoch_updates += send
        buf = {}
        live_ext = np.r_[True, alive]
        newly_suspect = []
        for i in np.flatnonzero(send):
            if ch[i] or head_of[i] < 0:
                buf.setdefault(i, []).append(i)
                continue
            h = head_of[i]
            who.update(tx=i + 1, ctx=2, crx=4, own=0.0)
            ok, ack = L.send(rng_link, i + 1, h + 1, pk.update, em, spend, live_ext, bursts)
            tx_bits += pk.update
            if ok:
                buf.setdefault(h, []).append(i)
            elif not alive[h]:
                newly_suspect.append(h)          # silent CH detected by its member
        # CH idle listening over the TDMA schedule ("none": switched off as a check)
        if listen_mode != "none":
            for h in np.flatnonzero(ch & alive):
                charge(h, em.listen(int((head_of == h).sum())), 3)
        if listen_mode == "all":
            # symmetric check: every relay also listens to one slot per node that
            # may route through it (CHs and unclustered nodes)
            senders = np.flatnonzero(known & (ch | (head_of < 0)))
            parents = nxt[senders + 1] - 1
            for j, c in zip(*np.unique(parents[parents >= 0], return_counts=True)):
                if alive[j]:
                    charge(int(j), em.listen(int(c)), 6)
        delivered = np.zeros(n, bool)
        confirmed = np.zeros(n, bool)
        for h, items in buf.items():
            if not alive[h]:
                continue
            if h not in route:
                route[h] = L.path(nxt, h + 1)
            hops = route[h]
            if hops is None:
                continue
            if ch[h]:
                charge(h, em.aggregation(len(items)), 4)
            bits = pk.aggregate(len(items))
            ok = True
            for a, b in hops:
                first = a == h + 1
                own = (1.0 / len(items) if h in items else 0.0) if (first and ch[h]) else 0.0
                who.update(tx=a, ctx=(5 if ch[h] else 2) if first else 6, crx=6, own=own)
                got, _ = L.send(rng_link, a, b, bits, em, spend, live_ext, bursts)
                tx_bits += bits
                if not got:
                    if b > 0 and not alive[b - 1]:
                        newly_suspect.append(b - 1)   # dead relay detected
                    ok = False
                    break
            if ok:
                delivered[items] = True
                members = [i for i in items if i != h]
                if not ack_loss:
                    # end-to-end acknowledgement: back along the route, then one
                    # beacon from the CH that its members overhear (assumed lossless)
                    for a, b in hops:
                        if b > 0:
                            charge(b - 1, em.tx(pk.ack, L.dist[a, b]), 7)
                        if a > 0:
                            charge(a - 1, em.rx(pk.ack), 7)
                    if members:
                        charge(h, em.tx(pk.ack, 10.0), 7)
                        charge(np.asarray(members), em.rx(pk.ack), 7)
                    confirmed[items] = True
                else:
                    # lossy check: the ACK travels back hop by hop with link-layer
                    # ARQ on the measured links; the CH beacon is heard by each
                    # member with the measured PRR of the CH-to-member link
                    back = True
                    for a, b in reversed(hops):
                        # b -> a uses the reverse direction of the data link
                        if bursts is None:
                            p_ack, p_ll = L.rev[a, b], L.fwd[a, b]
                        else:
                            p_ack, p_ll = bursts.rev_now[a, b], bursts.fwd_now[a, b]
                        got = False
                        for _ in range(MAX_TRIES):
                            if b > 0:
                                charge(b - 1, em.tx(pk.ack, L.dist[a, b]), 7)
                            if rng_ack.random() < p_ack:
                                if not got and a > 0:
                                    charge(a - 1, em.rx(pk.ack), 7)
                                got = True
                                if a > 0:
                                    charge(a - 1, em.tx(pk.ack, L.dist[a, b]), 7)
                                if rng_ack.random() < p_ll:
                                    if b > 0:
                                        charge(b - 1, em.rx(pk.ack), 7)
                                    break
                        if not got:
                            back = False
                            break
                    if back:
                        if h in items:
                            confirmed[h] = True
                        if members:
                            charge(h, em.tx(pk.ack, L.dist[h + 1, np.asarray(members) + 1].max()), 7)
                            charge(np.asarray(members), em.rx(pk.ack), 7)
                            heard = rng_ack.random(len(members)) < L.rev[np.asarray(members) + 1, h + 1]
                            confirmed[np.asarray(members)[heard]] = True
        # local repair: a detected dead CH or relay is reported with the next
        # packet; orphaned members route straight to the sink and paths are
        # recomputed around the hole
        if newly_suspect:
            suspect[newly_suspect] = True
            known &= ~suspect
            orphaned = np.isin(head_of, np.flatnonzero(suspect))
            head_of[orphaned] = -1
            ch &= ~suspect
            nxt, _ = L.shortest_paths(np.flatnonzero(known) + 1, relay_pen)
            route = {}
        deliv += int(delivered.sum())

        # ------------------------------------------------------- predictors
        # a node commits a reading to its model only when the sink confirmed it.
        # With lossless end-to-end ACKs node and sink copies stay identical; when
        # an ACK can be lost, a node that delivered but heard no ACK diverges from
        # the sink for one round and re-sends, and both copies restart from the
        # re-sent reading once it is confirmed
        in_sync = sensing & valid
        node.advance(np.where(send & confirmed, x, pn), pn, in_sync)
        ps = sink.predict()
        vs = np.where(delivered, x, ps)
        sink.advance(vs, ps, in_sync & known)
        resync = confirmed & (~valid | unconfirmed)
        unconfirmed = (unconfirmed | (delivered & ~confirmed)) & ~resync
        if resync.any():
            node.reset(resync, x)
            sink.reset(resync, x)
            valid |= resync
        reporting = known & valid & ~silent
        est = vs.copy()
        inferred = np.flatnonzero(~reporting)
        if len(inferred):
            key = frozenset(np.flatnonzero(reporting).tolist())
            for i in inferred:
                est[i] = spatial.estimate(i, key, est)
            est[inferred] = np.clip(est[inferred], node.lo[inferred], node.hi[inferred])
        spatial.push(est, reporting)
        if keep_series:
            est_hist[t] = est
        sq = (est - x) ** 2
        err[t] = np.sqrt(np.mean(sq))
        err_dead[t] = np.sum(sq[~reporting]) / n      # inferred locations (dead, silent)
        err_live[t] = np.sum(sq[reporting]) / n

        # ------------------------------------------------------------ deaths
        newly = alive & (E <= 0)
        death[newly] = t
        alive &= ~newly
        alive_series[t] = alive.sum()
        silent_series[t] = (alive & silent).sum()

    def first_cross(series, thr):
        idx = np.flatnonzero(series > thr)
        return int(idx[0]) if len(idx) else T

    roll = np.sqrt(np.convolve(err ** 2, np.ones(cfg.window) / cfg.window, mode="full")[:T])
    deaths = np.sort(np.where(death >= 0, death, T))
    res = {
        "method": method, "seed": seed,
        "FND": int(deaths[0]), "HND": int(deaths[n // 2 - 1]), "LND": int(deaths[-1]),
        "dead_end": int((death >= 0).sum()),
        "mean_rmse": float(np.mean(err)),
        "energy_used": float(np.sum(E0 - np.maximum(E, 0))),
        "updates_generated": gen, "updates_delivered": deliv,
        "pdr": deliv / max(gen, 1),
        "tx_kbits": tx_bits / 1e3,
        "ch_per_epoch": float(ch_count.sum() / max(epoch, 1)),
        "T": T,
        "ledger_share": (ledger.sum(axis=0) / max(ledger.sum(), 1e-12)).round(4).tolist(),
    }
    for name, thr in cfg.extra.get("deltas", {"FL05": 0.5, "FL10": 1.0}).items():
        # FL: rounds until the 6-h field RMSE first exceeds delta;
        # CT: rounds (over the whole trace) in which it stays within delta
        res[name] = first_cross(roll, thr)
        res["CT" + name[2:]] = int(np.sum(roll <= thr))
    if keep_series:
        res["series"] = {"rmse": err, "roll": roll, "mse_dead": err_dead, "mse_live": err_live,
                         "silent": silent_series, "alive": alive_series, "death": death,
                         "ch_count": ch_count, "est": est_hist, "ledger": ledger}
    return res
