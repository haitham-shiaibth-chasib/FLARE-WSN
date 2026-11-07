"""Data-collection policies compared in the paper.

At the start of every epoch the sink builds a :class:`Context` and asks the
policy for three things: the cluster heads (none for clustering-free
collection), the reporting threshold of every node (``np.inf`` = silent) and
an optional relay penalty for routing. Everything else is shared by all
policies: dual prediction, joining the CH with the lowest ETX, ETX shortest-path
routing, local repair, control and acknowledgement accounting, and the sink's
spatial reconstruction. The comparison therefore isolates the policy.
"""
from dataclasses import dataclass, field, replace

import numpy as np

from .spatial import coverage_cost


@dataclass
class Context:
    rng: np.random.Generator
    epoch: int
    alive: np.ndarray            # sink view of the live nodes (bool, n)
    E: np.ndarray                # residual energy reported in status packets
    E0: np.ndarray
    links: object
    pos: np.ndarray
    spatial: object
    curve: np.ndarray            # (G, n) update rate vs threshold on the calibration days
    grid: np.ndarray
    eps_bar: float
    p_ch: float
    rc: float                    # cluster radius / CH exclusion radius (m)
    rate_hist: list              # per-epoch update counts observed by the sink
    epoch_len: int
    em: object
    cfg: dict = field(default_factory=dict)


# ---------------------------------------------------------------- helpers
def k_target(ctx):
    return max(1, int(np.ceil(ctx.p_ch * ctx.alive.sum())))


def greedy_exclusion(score, ctx, k=None):
    """CHs in descending score; nodes within rc of a chosen CH are excluded."""
    k = k_target(ctx) if k is None else k
    alive = np.flatnonzero(ctx.alive)
    order = alive[np.argsort(-score[alive], kind="stable")]
    d = np.linalg.norm(ctx.pos[:, None] - ctx.pos[None, :], axis=2)
    chosen, blocked = [], np.zeros(len(score), bool)
    for i in order:
        if len(chosen) >= k:
            break
        if blocked[i]:
            continue
        chosen.append(i)
        blocked |= d[i] < ctx.rc
    ch = np.zeros(len(score), bool)
    ch[chosen] = True
    return ch


def uniform_eps(ctx):
    return np.full(len(ctx.alive), ctx.eps_bar)


def no_heads(ctx):
    return np.zeros(len(ctx.alive), bool)


class Policy:
    name = "base"

    def plan(self, ctx):
        raise NotImplementedError


# ------------------------------------------------------- clustered baselines
class LEACH(Policy):
    """Heinzelman et al. (2002): stochastic rotation with threshold T(n).

    G holds the nodes that have not yet served as CH in the current cycle of
    1/p epochs; it is refilled at the start of every cycle, so that on average
    a fraction p of the nodes serves in every epoch and each node serves once
    per cycle."""
    name = "LEACH"

    def __init__(self):
        self.G = None

    def plan(self, ctx):
        n = len(ctx.alive)
        p = ctx.p_ch
        period = int(round(1 / p))
        r = ctx.epoch
        if self.G is None or r % period == 0:
            self.G = np.ones(n, bool)
        T = p / (1 - p * (r % period))
        eligible = ctx.alive & self.G
        ch = eligible & (ctx.rng.random(n) < T)
        if not ch.any():
            pool = np.flatnonzero(eligible) if eligible.any() else np.flatnonzero(ctx.alive)
            ch[ctx.rng.choice(pool)] = True
        self.G[ch] = False
        return ch, uniform_eps(ctx), None


class HEED(Policy):
    """Younis and Fahmy (2004): iterative election driven by residual energy.

    Simplified: a node that draws below its election probability becomes a
    final CH at once (no tentative stage and no cost-based choice among
    tentative heads); coverage is judged by the cluster radius rc."""
    name = "HEED"

    def __init__(self, c_prob=0.05, p_min=1e-4):
        self.c_prob, self.p_min = c_prob, p_min

    def elect(self, ctx):
        n = len(ctx.alive)
        alive = ctx.alive
        d = np.linalg.norm(ctx.pos[:, None] - ctx.pos[None, :], axis=2)
        prob = np.where(alive, np.clip(self.c_prob * ctx.E / ctx.E0.max(), self.p_min, 1.0), 0.0)
        ch = np.zeros(n, bool)
        while True:
            covered = (d[:, ch] < ctx.rc).any(axis=1) if ch.any() else np.zeros(n, bool)
            todo = alive & ~covered & ~ch
            if not todo.any():
                break
            ch |= todo & (ctx.rng.random(n) < prob)
            if (prob[todo] >= 1.0).all():
                ch |= todo                     # uncovered nodes finally declare themselves
                break
            prob = np.minimum(prob * 2.0, 1.0)
        return ch

    def plan(self, ctx):
        return self.elect(ctx), uniform_eps(ctx), None


class CoverageAware(Policy):
    """Soro and Heinzelman (2009): spend nodes in densely covered areas first."""
    name = "CA-CH"

    def __init__(self, rs=6.0):
        self.rs = rs

    def plan(self, ctx):
        cost = coverage_cost(ctx.pos, ctx.alive, self.rs)
        score = np.where(np.isfinite(cost), (ctx.E / ctx.E0) / cost, 0.0)
        return greedy_exclusion(score, ctx), uniform_eps(ctx), None


class TrafficStability(Policy):
    """Ibrahim et al. (IJIES 2026): 0.4 traffic load + 0.3 traffic stability +
    0.3 residual energy, top 10 % as CHs. Only past epochs are used."""
    name = "TS-CH"

    def plan(self, ctx):
        n = len(ctx.alive)
        hist = ctx.rate_hist[-6:]
        if hist:
            H = np.array(hist)
            traffic, var = H[-1], (H.var(axis=0) if len(H) > 1 else np.zeros(n))
        else:
            traffic = ctx.curve[np.searchsorted(ctx.grid, ctx.eps_bar)]
            var = np.zeros(n)
        a = ctx.alive

        def mm(v):
            v = v.astype(float)
            lo, hi = v[a].min(), v[a].max()
            return (v - lo) / (hi - lo) if hi > lo else np.zeros_like(v)
        score = 0.4 * mm(traffic) + 0.3 * mm(1.0 / (1.0 + var)) + 0.3 * mm(ctx.E)
        score[~a] = -np.inf
        ch = np.zeros(n, bool)
        ch[np.argsort(-score, kind="stable")[: k_target(ctx)]] = True
        return ch & a, uniform_eps(ctx), None


class PEAGC(Policy):
    """Al-Robyee et al. (IJIES 2026): Leiden communities on an energy-distance
    graph; per community the CH maximises 0.4 energy + 0.2 centrality +
    0.2 sink proximity + 0.2 predicted survivability."""
    name = "PEAGC"

    def plan(self, ctx):
        import igraph as ig
        import leidenalg as la
        n = len(ctx.alive)
        idx = np.flatnonzero(ctx.alive)
        L = ctx.links
        d = L.dist[1:, 1:]
        sub = L.usable[1:, 1:][np.ix_(idx, idx)]
        dbar = d[np.ix_(idx, idx)][sub].mean() if sub.any() else 10.0
        edges, weights = [], []
        for a_, i in enumerate(idx):
            for b_, j in enumerate(idx):
                if j > i and (L.usable[i + 1, j + 1] or L.usable[j + 1, i + 1]):
                    edges.append((a_, b_))
                    weights.append(np.exp(-d[i, j] / dbar) * (ctx.E[i] + ctx.E[j]) / (2 * ctx.E0.mean()))
        g = ig.Graph(n=len(idx), edges=edges)
        part = la.find_partition(g, la.ModularityVertexPartition, weights=weights or None,
                                 seed=int(ctx.rng.integers(1 << 30)))
        wdeg = np.array(g.strength(weights=weights)) if edges else np.zeros(len(idx))
        dsink = L.dist[0, idx + 1]
        em = ctx.em
        u = ctx.curve[np.searchsorted(ctx.grid, ctx.eps_bar)]
        ch = np.zeros(n, bool)
        for comm in part:
            comm = np.array(comm)
            nodes = idx[comm]
            m = len(nodes)
            ru = float(u[nodes].mean())
            e_pred = ctx.epoch_len * (em.listen(m - 1) + (m - 1) * ru * em.rx(em.packets.update)
                                      + em.tx(em.packets.aggregate(max(1, int(m * ru))), 10.0))
            e_future = ctx.E[nodes] - e_pred
            elig = e_future >= 0.1 * ctx.E0[nodes]
            if not elig.any():
                elig[:] = True
            cen = wdeg[comm] / max(wdeg[comm].max(), 1e-12)
            prox = 1.0 - dsink[comm] / max(dsink.max(), 1e-12)
            score = (0.4 * ctx.E[nodes] / ctx.E0[nodes] + 0.2 * cen + 0.2 * prox
                     + 0.2 * np.clip(e_future / ctx.E0[nodes], 0, 1))
            score[~elig] = -np.inf
            ch[nodes[np.argmax(score)]] = True
        return ch, uniform_eps(ctx), None


def correlation_sleep(ctx, tau):
    """Node selection after Singh and Soni (2021): in each correlated
    neighbourhood the node with the least energy sleeps when its location can
    be inferred within tau; its six nearest neighbours stay awake."""
    live = np.flatnonzero(ctx.alive)
    r = ctx.spatial.irreplaceability(frozenset(live.tolist()), list(live))
    d = np.linalg.norm(ctx.pos[:, None] - ctx.pos[None, :], axis=2)
    protected = np.zeros(len(ctx.alive), bool)
    eps = uniform_eps(ctx)
    for i in live[np.argsort(ctx.E[live], kind="stable")]:
        if r[i] <= tau and not protected[i]:
            eps[i] = np.inf
            protected[live[np.argsort(d[i, live], kind="stable")[1:7]]] = True
    return eps


class CorrSleep(Policy):
    """Correlation-based node sleeping on top of HEED (Singh and Soni, 2021).
    Sleeping nodes take no part in the CH election."""
    name = "CORR-SL"

    def __init__(self):
        self.heed = HEED()

    def plan(self, ctx):
        eps = correlation_sleep(ctx, ctx.cfg.get("tau_s", 0.3))
        awake = replace(ctx, alive=ctx.alive & np.isfinite(eps))
        return self.heed.elect(awake), eps, None


# ------------------------------------------------- clustering-free policies
class Flat(Policy):
    """No clustering: every node sends its updates along the ETX tree."""
    name = "FLAT"

    def plan(self, ctx):
        return no_heads(ctx), uniform_eps(ctx), None


class FlatEnergyAware(Policy):
    """Clustering-free collection with residual-energy-aware relaying: a hop
    into relay j costs ETX * (1 + lam * (E0_j / E_j - 1)), the classic way of
    steering traffic away from depleted nodes."""
    name = "FLAT-EA"

    def plan(self, ctx):
        lam = ctx.cfg.get("lam", 1.0)
        pen = lam * np.clip(ctx.E0 / np.maximum(ctx.E, 1e-9) - 1.0, 0.0, 20.0)
        return no_heads(ctx), uniform_eps(ctx), pen


class FlatCorrSleep(Policy):
    """Clustering-free collection with correlation-based sleeping."""
    name = "FLAT-CS"

    def plan(self, ctx):
        return no_heads(ctx), correlation_sleep(ctx, ctx.cfg.get("tau_s", 0.3)), None


# ------------------------------------------------------------------ FLARE
class FLARE(Policy):
    """Proposed scheme: Fidelity-Lifetime-Aware Relaying and rEdundancy silencing.

    The fidelity of the reconstructed field is lost when the network loses a
    location it cannot infer from the others. At every epoch the sink
      1. computes the irreplaceability r_i of every live node, i.e. the error
         of inferring its location from the other live nodes, taken on the
         worst 6-h window of the last 48 h of its own history;
      2. turns it into a duty weight w_i = clip(r_i / median(r), 0.2, 5)^gamma;
      3. routes every packet over the ETX tree in which a hop into relay j
         costs ETX * (1 + lam * w_j), so irreplaceable nodes stop being the
         relays everybody leans on;
      4. silences the least irreplaceable nodes as long as the expected field
         error stays within the design budget delta_d (greedy, re-checked
         against the nodes that keep reporting); silent nodes save their
         sensing and reporting energy and relay for the others.
    Collection is clustering-free (no CH duty, no TDMA listening), which on a
    deployment where most nodes reach the sink in one or two hops is cheaper
    than any of the clustered schemes.
    """
    name = "FLARE"

    def __init__(self, use_sil=True, use_rt=True, weight="info", name=None):
        self.use_sil, self.use_rt, self.weight = use_sil, use_rt, weight
        if name:
            self.name = name

    def weights(self, ctx, r):
        live = ctx.alive
        gamma = ctx.cfg.get("gamma", 1.0)
        base = r
        if self.weight == "geo":
            # geometric stand-in: coverage cost (Soro and Heinzelman, 2009)
            base = coverage_cost(ctx.pos, live, 6.0)
            base = np.where(np.isfinite(base), base, 0.0)
        w = np.ones(len(live))
        w[live] = np.clip(base[live] / max(np.median(base[live]), 1e-12), 0.2, 5.0) ** gamma
        return w

    def silence(self, ctx, r, r_dead):
        """Greedy silencing of the least irreplaceable nodes under the budget
            sum_silent r_i(R)^2 + sum_dead r_j(R)^2 + a * eps_bar^2 * |R| <= N * delta_d^2
        where R is the set of nodes that keep reporting and a = 1/3 (error of a
        reporting node spread over [-eps, eps])."""
        a = 1.0 / 3.0
        n = len(ctx.alive)
        live = np.flatnonzero(ctx.alive)
        kappa = ctx.cfg.get("kappa", 1.0)
        room = n * ctx.cfg["delta_d"] ** 2 - np.sum(np.asarray(r_dead) ** 2) \
            - a * len(live) * ctx.eps_bar ** 2
        d = np.linalg.norm(ctx.pos[:, None] - ctx.pos[None, :], axis=2)
        S, used = [], 0.0
        inS = np.zeros(n, bool)
        for i in live[np.argsort(r[live], kind="stable")]:
            nb = live[np.argsort(d[i, live], kind="stable")[1:7]]
            if inS[nb].sum() >= 2:                # keep enough reporting anchors around i
                continue
            rep = frozenset(np.setdiff1d(live, S + [i]).tolist())
            ri = ctx.spatial.irreplaceability(rep, [i], peak=True)[i]
            if used + (kappa * ri) ** 2 > room:
                break
            S.append(int(i))
            inS[i] = True
            used += (kappa * ri) ** 2
        # re-check with every silent location inferred from the final reporting set
        while S:
            R = frozenset(np.setdiff1d(live, S).tolist())
            rS = ctx.spatial.irreplaceability(R, S, peak=True)
            if float(np.sum((kappa * rS[S]) ** 2)) <= room:
                break
            S.remove(S[int(np.argmax(rS[S]))])
        return S

    def plan(self, ctx):
        live = np.flatnonzero(ctx.alive)
        dead = np.flatnonzero(~ctx.alive)
        r = ctx.spatial.irreplaceability(frozenset(live.tolist()), list(live) + list(dead), peak=True)
        eps = uniform_eps(ctx)
        if self.use_sil:
            S = self.silence(ctx, r, r[dead])
            eps[S] = np.inf
        pen = None
        if self.use_rt:
            pen = ctx.cfg.get("lam", 3.0) * self.weights(ctx, r)
        self.last = {"r": r}
        return no_heads(ctx), eps, pen


POLICIES = {
    "LEACH": LEACH, "HEED": HEED, "CA-CH": CoverageAware, "TS-CH": TrafficStability,
    "PEAGC": PEAGC, "CORR-SL": CorrSleep,
    "FLAT": Flat, "FLAT-EA": FlatEnergyAware, "FLAT-CS": FlatCorrSleep,
    "FLARE": FLARE,
    "FLARE-noSIL": lambda: FLARE(use_sil=False, name="FLARE-noSIL"),
    "FLARE-noRT": lambda: FLARE(use_rt=False, name="FLARE-noRT"),
    "FLARE-geo": lambda: FLARE(weight="geo", name="FLARE-geo"),
}


def build(name):
    return POLICIES[name]()
