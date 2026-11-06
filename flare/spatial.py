"""Sink-side spatial inference and node irreplaceability.

The sink keeps a sliding window (48 h) of its reconstructed field and a mask
of the locations that were actually *reported* (live, not silent) in each
round; the window starts with the pre-deployment calibration days. When a node
stops reporting (it dies or is told to be silent) the sink freezes a copy of
the window as it was at that moment, so the node keeps a record of how it
co-varied with the others.

A location that is not reported is estimated by ridge regression on its k
nearest reporting neighbours, fitted only on the rounds in which the node and
all those neighbours were reported together (taken from the frozen copy for a
node that is not reporting). The root-mean-square residual of that fit is the
node's irreplaceability r_i(R): the error expected at location i when it is
inferred from the reporting set R. All methods share this component.
"""
import numpy as np


class SpatialModel:
    def __init__(self, history: np.ndarray, pos: np.ndarray, k: int = 6, lam: float = 1e-2,
                 window: int = 1440, min_rows: int = 120, peak_len: int = 180):
        self.pos = pos
        self.peak_len = peak_len
        self.k, self.lam, self.window, self.min_rows = k, lam, window, min_rows
        self.n = history.shape[1]
        self.dist = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
        h = history[-window:]
        self.buf = list(h)
        self.mbuf = [np.ones(self.n, bool) for _ in range(len(h))]
        self.frozen = {}                      # node -> (H, M) when it stopped reporting
        self.reporting = np.ones(self.n, bool)
        self.refit(self.reporting)

    # ----------------------------------------------------------- statistics
    def push(self, row: np.ndarray, reported: np.ndarray):
        self.buf.append(row.copy())
        self.mbuf.append(reported.copy())
        if len(self.buf) > self.window:
            self.buf.pop(0)
            self.mbuf.pop(0)

    def refit(self, reporting: np.ndarray):
        """Called once per epoch with the set of nodes that reported."""
        self.H = np.asarray(self.buf)
        self.M = np.asarray(self.mbuf)
        stopped = self.reporting & ~reporting
        for i in np.flatnonzero(stopped):
            if self.M[:, i].sum() >= self.min_rows:
                self.frozen[i] = (self.H, self.M)
        for i in np.flatnonzero(reporting):
            self.frozen.pop(i, None)
        self.reporting = reporting.copy()
        self._fit_cache = {}
        self._r_cache = {}

    # -------------------------------------------------------------- inference
    def _data(self, i):
        return self.frozen.get(i, (self.H, self.M))

    def _fit(self, i: int, rep_key: frozenset):
        key = (i, rep_key)
        hit = self._fit_cache.get(key)
        if hit is not None:
            return hit
        H, M = self._data(i)
        cand = np.array(sorted(j for j in rep_key if j != i), dtype=int)
        own = M[:, i]
        nb = cand[np.argsort(self.dist[i, cand], kind="stable")[: self.k]] if cand.size else cand
        # drop the neighbours least often reported together with i until
        # enough common rounds remain
        rows = own & (M[:, nb].all(axis=1) if nb.size else True)
        while nb.size and rows.sum() < self.min_rows:
            together = (M[:, nb] & own[:, None]).sum(axis=0)
            nb = np.delete(nb, int(np.argmin(together)))
            rows = own & (M[:, nb].all(axis=1) if nb.size else True)
        Hr = H[rows] if rows.sum() >= 2 else H[own] if own.sum() >= 2 else H
        y = Hr[:, i]
        mu_y = float(y.mean())
        if nb.size == 0:
            res = (nb, np.zeros(0), mu_y, float(y.std()), float(y.std()))
        else:
            X = Hr[:, nb]
            mu_x = X.mean(axis=0)
            Xc, yc = X - mu_x, y - mu_y
            G = Xc.T @ Xc / len(Xc)
            reg = self.lam * np.trace(G) / len(nb) + 1e-9
            b = np.linalg.solve(G + reg * np.eye(len(nb)), Xc.T @ yc / len(Xc))
            resid = yc - Xc @ b
            # peak error: worst block of `peak_len` consecutive usable rounds
            # (the fidelity metric is evaluated on windows of that length)
            sq = resid ** 2
            L = self.peak_len
            if len(sq) >= L:
                c = np.cumsum(np.r_[0.0, sq])
                peak = float(np.sqrt(np.max((c[L:] - c[:-L]) / L)))
            else:
                peak = float(np.sqrt(sq.mean()))
            res = (nb, b, float(mu_y - b @ mu_x), float(np.sqrt(np.mean(sq))), peak)
        self._fit_cache[key] = res
        return res

    def estimate(self, i: int, rep_key: frozenset, values: np.ndarray) -> float:
        nb, b, a = self._fit(i, rep_key)[:3]
        return a if nb.size == 0 else a + float(b @ values[nb])

    def irreplaceability(self, rep_key: frozenset, nodes, peak: bool = False) -> np.ndarray:
        """Residual RMS of inferring each node in ``nodes`` from the reporting
        set; with ``peak`` the RMS of the worst window instead of the average."""
        nodes = tuple(int(i) for i in nodes)
        ck = (rep_key, nodes, peak)
        hit = self._r_cache.get(ck)
        if hit is not None:
            return hit
        r = np.zeros(self.n)
        for i in nodes:
            r[i] = self._fit(i, rep_key)[4 if peak else 3]
        self._r_cache[ck] = r
        return r


def coverage_cost(pos: np.ndarray, alive: np.ndarray, rs: float, step: float = 0.5) -> np.ndarray:
    """Coverage-aware cost of Soro and Heinzelman (2009): for each node, the sum
    over the points it covers of 1/(number of live nodes covering the point).
    A low cost means the node sits in a densely covered area."""
    idx = np.flatnonzero(alive)
    lo = pos.min(axis=0) - rs
    hi = pos.max(axis=0) + rs
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0], step), np.arange(lo[1], hi[1], step))
    pts = np.c_[gx.ravel(), gy.ravel()]
    cover = np.linalg.norm(pts[:, None, :] - pos[None, idx, :], axis=2) <= rs
    cnt = cover.sum(axis=1)
    w = np.where(cnt > 0, 1.0 / np.maximum(cnt, 1), 0.0)
    cost = np.full(len(pos), np.inf)
    cost[idx] = (cover * w[:, None]).sum(axis=0) * step * step
    return cost
