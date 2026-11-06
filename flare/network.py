"""Measured links, ARQ transmission and ETX routing.

Index 0 is the base station (sink); nodes are 1..N in this module and 0..N-1
everywhere else (``node = idx - 1``). Link success probabilities come straight
from the Intel Lab connectivity file, which is asymmetric. Acknowledgements use
the reverse link; when the reverse link was never observed we fall back to the
forward value and say so in the paper.
"""
import heapq

import numpy as np

MIN_PRR = 0.10          # links below this are treated as unusable
MAX_TRIES = 3           # link-layer ARQ attempts


class Links:
    def __init__(self, prr: np.ndarray, pos: np.ndarray, ideal: bool = False):
        n1 = prr.shape[0]
        self.n = n1 - 1
        fwd = prr.copy()
        np.fill_diagonal(fwd, 0.0)
        rev = fwd.T.copy()
        rev = np.where(rev > 0, rev, fwd)          # missing reverse -> symmetric
        if ideal:                                   # unit-disk reference used in a sensitivity run
            fwd = np.where(fwd >= MIN_PRR, 1.0, 0.0)
            rev = np.where(fwd > 0, 1.0, 0.0)
        self.fwd, self.rev = fwd, rev
        self.usable = (fwd >= MIN_PRR) & (rev >= MIN_PRR)
        # expected transmissions under ARQ with ACK on the reverse link
        q = np.clip(fwd * rev, 1e-9, 1.0)
        self.etx = np.where(self.usable, 1.0 / q, np.inf)
        # positions: nodes from mote_locs, sink at the PRR-weighted centroid of
        # the nodes that hear it best (its position is not published)
        to_sink = fwd[1:, 0] ** 4
        sink_xy = (pos * to_sink[:, None]).sum(axis=0) / to_sink.sum()
        self.xy = np.vstack([sink_xy, pos])
        self.dist = np.linalg.norm(self.xy[:, None] - self.xy[None, :], axis=2)

    # ------------------------------------------------------------------ routing
    def shortest_paths(self, alive_idx: np.ndarray, relay_penalty: np.ndarray | None = None):
        """Dijkstra from every live node to the sink over usable links.

        The cost of a hop i->j is ETX(i,j) * (1 + relay_penalty[j]) for a relay j
        (the penalty is zero for the sink). Returns parent[] (next hop, -1 if
        unreachable) for all indices 0..N.
        """
        n1 = self.n + 1
        live = np.zeros(n1, bool)
        live[0] = True
        live[alive_idx] = True
        pen = np.zeros(n1) if relay_penalty is None else np.r_[0.0, relay_penalty]
        dist = np.full(n1, np.inf)
        nxt = np.full(n1, -1)
        dist[0] = 0.0
        heap = [(0.0, 0)]
        while heap:
            d, j = heapq.heappop(heap)
            if d > dist[j]:
                continue
            cand = np.flatnonzero(live & self.usable[:, j])
            for i in cand:
                c = d + self.etx[i, j] * (1.0 + pen[j])
                if c < dist[i]:
                    dist[i] = c
                    nxt[i] = j
                    heapq.heappush(heap, (c, i))
        return nxt, dist

    def path(self, nxt: np.ndarray, src: int):
        hops, s, guard = [], src, 0
        while s != 0 and s != -1 and guard <= self.n:
            hops.append((s, nxt[s]))
            s = nxt[s]
            guard += 1
        return hops if s == 0 else None

    # ------------------------------------------------------------- transmission
    def send(self, rng, i: int, j: int, bits: int, em, spend, true_alive, bursts=None):
        """One hop with ARQ. ``spend(idx, joules)`` charges energy. Returns
        (delivered, acked). A dead receiver (not yet known dead) never answers.
        With ``bursts`` the success probabilities of the current round come from
        the two-state link model below instead of the long-term PRR."""
        delivered = acked = False
        rx_ok = j == 0 or true_alive[j]
        if bursts is None:
            pf, pr = self.fwd[i, j], self.rev[i, j]
        else:
            pf, pr = bursts.fwd_now[i, j], bursts.rev_now[i, j]
        for _ in range(MAX_TRIES):
            spend(i, em.tx(bits, self.dist[i, j]))
            if rx_ok and rng.random() < pf:
                if not delivered:
                    spend(j, em.rx(bits))
                delivered = True
                spend(j, em.tx(em.packets.ack, self.dist[i, j]))
                if rng.random() < pr:
                    spend(i, em.rx(em.packets.ack))
                    acked = True
                    break
        return delivered, acked


class BurstyLinks:
    """Gilbert-Elliott losses on top of the measured links (sensitivity run).

    Every directed link is either good or bad, and the state changes only between
    rounds, so the ARQ retries of one round share the state of the link. A bad
    period lasts ``burst_len`` rounds on average and a link spends the fraction
    ``bad_share`` of the time in it. In the bad state the success probability is
    ``beta`` times the measured PRR; the good-state probability is raised so that
    the long-term mean of every link stays equal to its measured PRR (when that
    would need a value above one, the good state is set to one and the bad state
    absorbs the rest). Routing keeps using the long-term ETX, as a real node would.
    """

    def __init__(self, links: Links, rng, burst_len: float, bad_share: float = 0.2, beta: float = 0.1):
        self.rng = rng
        self.q_bg = 1.0 / burst_len                                  # bad -> good
        self.q_gb = bad_share / (1.0 - bad_share) / burst_len       # good -> bad
        self.good_f, self.bad_f = self._split(links.fwd, bad_share, beta)
        self.good_r, self.bad_r = self._split(links.rev, bad_share, beta)
        shape = links.fwd.shape
        self.state_f = rng.random(shape) < bad_share                # start in the stationary mix
        self.state_r = rng.random(shape) < bad_share
        self._refresh()

    @staticmethod
    def _split(p, pi_b, beta):
        bad = beta * p
        good = (p - pi_b * bad) / (1.0 - pi_b)
        over = good > 1.0
        good = np.where(over, 1.0, good)
        bad = np.where(over, (p - (1.0 - pi_b)) / pi_b, bad)
        return good, np.clip(bad, 0.0, 1.0)

    def _refresh(self):
        self.fwd_now = np.where(self.state_f, self.bad_f, self.good_f)
        self.rev_now = np.where(self.state_r, self.bad_r, self.good_r)

    def step(self):
        """Advance every link by one round."""
        for name in ("state_f", "state_r"):
            st = getattr(self, name)
            u = self.rng.random(st.shape)
            setattr(self, name, np.where(st, u >= self.q_bg, u < self.q_gb))
        self._refresh()
