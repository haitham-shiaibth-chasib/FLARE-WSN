"""Dual-prediction models, vectorised over nodes.

Node and sink run identical copies of a normalised LMS predictor (Santini and
Roemer, 2006). A copy is advanced with the real reading when an update is
delivered and with its own prediction otherwise, so the copies stay identical
without extra signalling. Copies that drift apart because of losses are
re-synchronised at every clustering epoch.
"""
import numpy as np


class NLMSBank:
    def __init__(self, n: int, p: int = 4, mu: float = 0.1, eps: float = 1e-2):
        self.n, self.p, self.mu, self.eps = n, p, mu, eps
        self.w = np.zeros((n, p))
        self.w[:, 0] = 1.0                       # start as a persistence predictor
        self.h = np.zeros((n, p))                # most recent value first
        self.base = np.zeros(n)                  # level removed before filtering
        self.lo = np.full(n, -np.inf)            # physical range, set from training data
        self.hi = np.full(n, np.inf)

    def cycles(self) -> int:
        """MCU cycles per sample on an MSP430 with hardware multiplier:
        p MACs to predict, about 3p for the update, plus bookkeeping."""
        return 10 * 4 * self.p + 60

    def warm_start(self, history: np.ndarray):
        """history: (p, n) readings, oldest first."""
        self.base = history[-1].copy()
        self.h = (history[::-1] - self.base).T.copy()

    def set_range(self, lo, hi):
        self.lo, self.hi = np.asarray(lo, float), np.asarray(hi, float)

    def predict(self) -> np.ndarray:
        return np.clip(self.base + np.einsum("ij,ij->i", self.w, self.h), self.lo, self.hi)

    def advance(self, value: np.ndarray, pred: np.ndarray, mask=None):
        """Feed ``value`` (reading or prediction). mask selects rows to advance."""
        rows = slice(None) if mask is None else mask
        v = value[rows] - self.base[rows]
        u = self.h[rows]
        e = v - (pred[rows] - self.base[rows])
        norm = self.eps + np.einsum("ij,ij->i", u, u)
        self.w[rows] = self.w[rows] + (self.mu * e / norm)[:, None] * u
        u = np.roll(u, 1, axis=1)
        u[:, 0] = v
        self.h[rows] = u

    def reset(self, mask, value):
        """Restart the selected predictors as persistence at ``value``."""
        self.w[mask] = 0.0
        self.w[mask, 0] = 1.0
        self.base[mask] = value[mask]
        self.h[mask] = 0.0

    def copy_from(self, other: "NLMSBank", mask):
        self.w[mask] = other.w[mask]
        self.h[mask] = other.h[mask]
        self.base[mask] = other.base[mask]


def update_rate_curves(series: np.ndarray, eps_grid, p: int = 4, mu: float = 0.1) -> np.ndarray:
    """Share of samples that trigger an update under threshold eps, per node,
    on lossless links. series: (T, n). Returns (len(eps_grid), n)."""
    T, n = series.shape
    out = np.zeros((len(eps_grid), n))
    for g, eps in enumerate(eps_grid):
        bank = NLMSBank(n, p=p, mu=mu)
        bank.warm_start(series[:p])
        count = np.zeros(n)
        for t in range(p, T):
            pred = bank.predict()
            x = series[t]
            send = np.abs(x - pred) > eps
            count += send
            bank.advance(np.where(send, x, pred), pred)
        out[g] = count / (T - p)
    # enforce monotonicity (finite-sample noise can make the curve wiggle)
    return np.minimum.accumulate(out, axis=0)
