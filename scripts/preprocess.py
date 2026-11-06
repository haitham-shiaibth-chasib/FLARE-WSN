"""Turn the raw Intel Berkeley Research Lab files into aligned traces.

Steps (all decisions are logged to data/processed/preprocess_report.json):
1. parse data.txt.gz, keep motes 1..54 and physically plausible readings; a
   reading above 40 C marks a sensor fault (it happens when a mote's battery
   runs low; mote 18 reaches 50-120 C on 8-9 March) and invalidates all three
   channels of that record; a mote with more than 5 % faulty records in a
   scenario window is excluded from that scenario;
2. remove isolated spikes with a Hampel filter (window 15 samples, 4 scaled MAD,
   MAD floored at a few quantisation steps so flat segments are not flagged);
3. average each mote onto a regular 2-minute grid (one simulation round);
4. keep the motes whose raw coverage in the study window is >= 60 %;
5. fill gaps by time interpolation (gaps are recorded, not hidden);
6. build two scenarios with their own node sets:
   A (tuning)    : calibration 28 Feb - 3 Mar, evaluation 3 Mar - 10 Mar;
   B (held out)  : calibration 7 Mar - 10 Mar, evaluation 10 Mar - 15 Mar.
   Every parameter of every method is chosen on A; results are reported on B.
The link file is restricted to the kept motes plus the base station (id 0).

Usage: python scripts/preprocess.py
"""
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"

SCENARIOS = {
    "A": (pd.Timestamp("2004-02-28 01:00"), pd.Timestamp("2004-03-03 01:00"), pd.Timestamp("2004-03-10 01:00")),
    "B": (pd.Timestamp("2004-03-07 01:00"), pd.Timestamp("2004-03-10 01:00"), pd.Timestamp("2004-03-15 01:00")),
}
START = min(v[0] for v in SCENARIOS.values())
END = max(v[2] for v in SCENARIOS.values())
BIN = "2min"
# lower bound on the robust scale, about 2-3 sensor quantisation steps
MAD_FLOOR = {"temp": 0.1, "hum": 0.3, "light": 5.0}
MIN_COVERAGE = 0.60
LIMITS = {"temp": (5.0, 40.0), "hum": (0.0, 100.0), "light": (0.0, 2500.0)}
FAULT_TEMP = 40.0
MAX_FAULT_SHARE = 0.05
EXPECTED_SHA256 = {
    "data.txt.gz": "d99288c8f406ca6604d359ceaa0d8adfffa79e7095061a1e27dc4399f48c7225",
    "mote_locs.txt": "3865c0263110c24c40e3377690cecaa552e0575cf56cdb9f5f8bd17130b6bf04",
    "connectivity.txt": "267a2bf3fcfb911df7e6746a9aee38bfbdd124caa566e05d2466d8a11666a6d8",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hampel(x: pd.Series, floor: float, k: int = 15, nsig: float = 4.0) -> pd.Series:
    med = x.rolling(k, center=True, min_periods=3).median()
    mad = (x - med).abs().rolling(k, center=True, min_periods=3).median()
    bad = (x - med).abs() > nsig * 1.4826 * mad.clip(lower=floor)
    return x.mask(bad)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"sha256": {}}
    for name, ref in EXPECTED_SHA256.items():
        got = sha256(RAW / name)
        report["sha256"][name] = got
        if got != ref:
            print(f"WARNING: checksum of {name} differs from the reference copy")

    cols = ["date", "time", "epoch", "mote", "temp", "hum", "light", "volt"]
    with gzip.open(RAW / "data.txt.gz", "rt") as f:
        df = pd.read_csv(f, sep=r"\s+", header=None, names=cols, on_bad_lines="skip")
    report["raw_rows"] = int(len(df))
    df = df.dropna(subset=["mote", "temp"])
    df["mote"] = df["mote"].astype(int)
    df = df[(df.mote >= 1) & (df.mote <= 54)]
    df["ts"] = pd.to_datetime(df["date"] + " " + df["time"], errors="coerce", format="mixed")
    df = df[(df.ts >= START) & (df.ts < END)]
    report["rows_in_window"] = int(len(df))

    df["fault"] = df["temp"] > FAULT_TEMP
    df.loc[df["fault"], ["temp", "hum", "light"]] = np.nan
    grid = pd.date_range(START, END, freq=BIN, inclusive="left")
    traces, coverage, removed = {}, {}, {}
    for sig, (lo, hi) in LIMITS.items():
        mats = {}
        for m, g in df.groupby("mote"):
            s = g.set_index("ts")[sig].sort_index()
            s = s[~s.index.duplicated()]
            s = s.where((s >= lo) & (s <= hi))
            n_before = s.notna().sum()
            s = hampel(s, MAD_FLOOR[sig])
            removed.setdefault(sig, {})[int(m)] = int(n_before - s.notna().sum())
            b = s.resample(BIN).mean().reindex(grid)
            mats[int(m)] = b
        traces[sig] = pd.DataFrame(mats)
        coverage[sig] = traces[sig].notna().mean()

    locs = pd.read_csv(RAW / "mote_locs.txt", sep=r"\s+", header=None, names=["mote", "x", "y"]).set_index("mote")
    links = pd.read_csv(RAW / "connectivity.txt", sep=r"\s+", header=None, names=["src", "dst", "prr"])
    report["hampel_removed"] = {s_: int(sum(v.values())) for s_, v in removed.items()}
    for name, (h0, e0, e1) in SCENARIOS.items():
        win = (grid >= h0) & (grid < e1)
        cov = pd.concat({sig: traces[sig].loc[win].notna().mean() for sig in LIMITS}, axis=1).min(axis=1)
        w = df[(df.ts >= h0) & (df.ts < e1)]
        fault_share = w.groupby("mote")["fault"].mean()
        faulty = {int(m) for m, v in fault_share.items() if v > MAX_FAULT_SHARE}
        kept = sorted(int(m) for m in cov.index
                      if cov[m] >= MIN_COVERAGE and m in locs.index and int(m) not in faulty)
        rep = {"window": [str(h0), str(e0), str(e1)], "kept_motes": kept,
               "dropped_motes": sorted(set(range(1, 55)) - set(kept)),
               "faulty_motes": sorted(faulty),
               "coverage": {int(k): round(float(v), 3) for k, v in cov.items()}}
        for sig in LIMITS:
            m = traces[sig].loc[win, kept]
            rep[f"filled_fraction_{sig}"] = float(m.isna().mean().mean())
            m = m.interpolate(method="time", limit_direction="both")
            hist = m.loc[(m.index >= h0) & (m.index < e0)].to_numpy(np.float32)
            ev = m.loc[(m.index >= e0) & (m.index < e1)].to_numpy(np.float32)
            np.save(OUT / f"{sig}_{name}_hist.npy", hist)
            np.save(OUT / f"{sig}_{name}_eval.npy", ev)
        rep["rounds_hist"], rep["rounds_eval"] = int(len(hist)), int(len(ev))
        ids = [0] + kept
        index = {m_: i for i, m_ in enumerate(ids)}
        prr = np.zeros((len(ids), len(ids)))
        for s_, d_, p_ in links.itertuples(index=False):
            if s_ in index and d_ in index and s_ != d_:
                prr[index[s_], index[d_]] = p_
        rep["links_prr_ge_0.5"] = int((prr >= 0.5).sum())
        np.save(OUT / f"prr_{name}.npy", prr)
        np.save(OUT / f"pos_{name}.npy", locs.loc[kept, ["x", "y"]].to_numpy(float))
        np.save(OUT / f"ids_{name}.npy", np.array(kept))
        report[name] = rep
    (OUT / "preprocess_report.json").write_text(json.dumps(report, indent=1))
    for name in SCENARIOS:
        r = report[name]
        print(name, r["window"], "nodes", len(r["kept_motes"]), "dropped", r["dropped_motes"],
              "faulty", r["faulty_motes"],
              "rounds", r["rounds_hist"], r["rounds_eval"],
              "filled", {s_: round(r[f"filled_fraction_{s_}"], 3) for s_ in LIMITS})


if __name__ == "__main__":
    main()
