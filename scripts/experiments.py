"""All experiments of the paper.

    python scripts/experiments.py tune      # week A: choose every method's parameters
    python scripts/experiments.py main      # week B: main comparison, 30 paired seeds
    python scripts/experiments.py ablation  # week B: FLARE components
    python scripts/experiments.py sens      # week B: sensitivity analysis
    python scripts/experiments.py signals   # week B: humidity and light
    python scripts/experiments.py burst     # week B: bursty-loss rows of the sensitivity analysis
    python scripts/experiments.py all

Raw per-run results go to results/raw (one JSON per run, cached by a hash of
the configuration and of the simulator source). Tuned parameters are written
to results/tuned.json and read back by the later stages.
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flare.runner import run_many  # noqa: E402
from flare.sim import Config  # noqa: E402

RES = ROOT / "results"
CLUSTERED = ["LEACH", "HEED", "CA-CH", "TS-CH", "PEAGC", "CORR-SL"]
FLAT = ["FLAT", "FLAT-EA", "FLAT-CS"]
BASELINES = CLUSTERED + FLAT
EPS_GRID = [0.4, 0.5, 0.6, 0.7, 0.8]
TAU_GRID = [0.2, 0.3, 0.4]
LAM_EA = [0.5, 1.0, 3.0]
DD_GRID = [0.3, 0.35, 0.4]
LAM_GRID = [1.0, 3.0]
TARGETS = {"FL05": 0.5, "FL10": 1.0}
E0 = 0.006
TUNE_SEEDS = range(10)
EVAL_SEEDS = range(30)


def base_cfg(**kw):
    extra = kw.pop("extra", {})
    kw.setdefault("E0", E0)
    cfg = Config(**kw)
    cfg.extra = {"delta_d": 0.35, **extra}
    return cfg


def candidates(method):
    """(label, Config kwargs) pairs explored for a method during tuning."""
    if method.startswith("FLARE"):
        return [(f"eps={e},dd={dd},lam={lam}", {"eps_bar": e, "extra": {"delta_d": dd, "lam": lam}})
                for e in EPS_GRID for dd in DD_GRID for lam in LAM_GRID]
    if method == "FLAT-EA":
        return [(f"eps={e},lam={lam}", {"eps_bar": e, "extra": {"lam": lam}})
                for e in EPS_GRID for lam in LAM_EA]
    if method in ("CORR-SL", "FLAT-CS"):
        return [(f"eps={e},tau={t}", {"eps_bar": e, "extra": {"tau_s": t}})
                for e in EPS_GRID for t in TAU_GRID]
    return [(f"eps={e}", {"eps_bar": e}) for e in EPS_GRID]


def frame(results, labels=None):
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "config"} for r in results])
    if labels is not None:
        df["setting"] = labels
    return df


# ------------------------------------------------------------------ stages
def tune():
    jobs, labels, sets = [], [], {}
    for m in BASELINES + ["FLARE"]:
        for lab, kw in candidates(m):
            cfg = base_cfg(scenario="A", **kw)
            sets[(m, lab)] = kw
            for s in TUNE_SEEDS:
                jobs.append((cfg, m, s))
                labels.append(lab)
    df = frame(run_many(jobs), labels)
    df.to_csv(RES / "tables" / "tuning_weekA_runs.csv", index=False)
    g = df.groupby(["method", "setting"])[list(TARGETS)].mean()
    tuned = {}
    for m in BASELINES + ["FLARE"]:
        tuned[m] = {}
        for tgt in TARGETS:
            best = g.loc[m][tgt].idxmax()
            tuned[m][tgt] = {"setting": best, "kw": sets[(m, best)],
                             "weekA_mean": float(g.loc[(m, best), tgt])}
    (RES / "tuned.json").write_text(json.dumps(tuned, indent=1))
    g.round(1).to_csv(RES / "tables" / "tuning_weekA_summary.csv")
    print(json.dumps({m: {t: v["setting"] for t, v in d.items()} for m, d in tuned.items()}, indent=1))


def tuned_cfg(method, target, scenario="B", **over):
    tuned = json.loads((RES / "tuned.json").read_text())
    key = "FLARE" if method.startswith("FLARE") else method
    kw = json.loads(json.dumps(tuned[key][target]["kw"]))
    extra = {**kw.pop("extra", {}), **over.pop("extra", {})}
    return base_cfg(scenario=scenario, extra=extra, **{**kw, **over})


def main_stage():
    rows = []
    for tgt in TARGETS:
        jobs = [(tuned_cfg(m, tgt), m, s) for m in BASELINES + ["FLARE"] for s in EVAL_SEEDS]
        df = frame(run_many(jobs))
        df["target"] = tgt
        rows.append(df)
    out = pd.concat(rows)
    out.to_csv(RES / "tables" / "main_weekB_runs.csv", index=False)
    print(out.groupby(["target", "method"])[["FL05", "FL10", "CT05", "CT10", "FND", "HND", "LND"]].mean().round(1))


def ablation():
    variants = ["FLARE", "FLARE-noSIL", "FLARE-noRT", "FLARE-geo", "FLAT"]
    rows = []
    for tgt in TARGETS:
        jobs = [(tuned_cfg("FLARE", tgt), v, s) for v in variants for s in EVAL_SEEDS]
        df = frame(run_many(jobs))
        df["target"] = tgt
        rows.append(df)
    out = pd.concat(rows)
    out.to_csv(RES / "tables" / "ablation_weekB_runs.csv", index=False)
    print(out.groupby(["target", "method"])[list(TARGETS) + ["FND", "HND"]].mean().round(1))


SENS_METHODS = ["HEED", "LEACH", "FLAT", "FLAT-EA", "FLARE"]
# bursty losses (Gilbert-Elliott): mean length of a bad period in rounds; 0 = off
BURST = {"burst_len": [0, 1, 15]}


def burst():
    """Only the bursty-loss rows of the sensitivity analysis. They were added after
    the other sweeps; the rows are merged into sensitivity_weekB_runs.csv."""
    tgt, seeds = "FL05", range(10)
    jobs, lab = [], []
    for m in SENS_METHODS:
        for v in BURST["burst_len"]:
            cfg = tuned_cfg(m, tgt, extra={"burst_len": v})
            for s in seeds:
                jobs.append((cfg, m, s))
                lab.append(("burst_len", str(v)))
    df = frame(run_many(jobs))
    df["factor"] = [a for a, _ in lab]
    df["value"] = [b for _, b in lab]
    f = RES / "tables" / "sensitivity_weekB_runs.csv"
    old = pd.read_csv(f)
    old = old[old.factor != "burst_len"]
    pd.concat([old, df]).to_csv(f, index=False)
    print(df.groupby(["value", "method"])[["FL05", "CT05", "FND", "HND"]].mean().round(0))


def sens():
    """One factor at a time around the tuned settings, for the two strongest
    clustered protocols, the clustering-free designs and FLARE."""
    tgt = "FL05"
    seeds = range(10)
    factors = {
        "E0": [0.003, 0.006, 0.012],
        "epoch_len": [15, 30, 60],
        "profile": ["first_order", "cc2420"],
        "ideal_links": [False, True],
        "k_nb": [4, 6, 8],
        "e0_jitter": [0.0, 0.05, 0.2],
        "p_ch": [0.05, 0.1, 0.2],
    }
    factors.update(BURST)
    flare_only = {"lam": [0.5, 1.0, 3.0, 10.0], "gamma": [0.5, 1.0, 2.0],
                  "kappa": [1.0, 1.25, 1.5], "delta_d": [0.25, 0.3, 0.35, 0.4, 0.45]}
    jobs, lab = [], []
    for m in SENS_METHODS:
        for f, vals in factors.items():
            if f == "p_ch" and m not in CLUSTERED:
                continue
            for v in vals:
                cfg = (tuned_cfg(m, tgt, extra={f: v}) if f in BURST
                       else tuned_cfg(m, tgt, **{f: v}))
                for s in seeds:
                    jobs.append((cfg, m, s))
                    lab.append((f, str(v)))
    for f, vals in flare_only.items():
        for v in vals:
            cfg = tuned_cfg("FLARE", tgt, extra={f: v})
            for s in seeds:
                jobs.append((cfg, "FLARE", s))
                lab.append((f, str(v)))
    df = frame(run_many(jobs))
    df["factor"] = [a for a, _ in lab]
    df["value"] = [b for _, b in lab]
    df.to_csv(RES / "tables" / "sensitivity_weekB_runs.csv", index=False)
    print(df.groupby(["factor", "value", "method"])[["FL05", "CT05", "FND", "HND"]].mean().round(0))


def signals():
    """Humidity (thresholds scaled x3, in %RH) and light (x40, in lux)."""
    scale = {"hum": 3.0, "light": 40.0}
    target_thr = {"hum": {"FL05": 1.5, "FL10": 3.0}, "light": {"FL05": 20.0, "FL10": 40.0}}
    rows = []
    for sig, sc in scale.items():
        deltas = {k: v for k, v in target_thr[sig].items()}
        for m in BASELINES + ["FLARE"]:
            cfg = tuned_cfg(m, "FL05")
            cfg = replace(cfg, signal=sig, eps_bar=cfg.eps_bar * sc)
            cfg.extra = {**cfg.extra, "eps_scale": sc, "deltas": deltas,
                         "delta_d": cfg.extra["delta_d"] * sc}
            if "tau_s" in cfg.extra:
                cfg.extra["tau_s"] = cfg.extra["tau_s"] * sc
            df = frame(run_many([(cfg, m, s) for s in range(10)]))
            df["signal"] = sig
            rows.append(df)
    out = pd.concat(rows)
    out.to_csv(RES / "tables" / "signals_weekB_runs.csv", index=False)
    print(out.groupby(["signal", "method"])[["FL05", "FL10", "FND", "HND"]].mean().round(0))


if __name__ == "__main__":
    (RES / "tables").mkdir(parents=True, exist_ok=True)
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    stages = {"tune": tune, "main": main_stage, "ablation": ablation, "sens": sens, "signals": signals, "burst": burst}
    for name, fn in stages.items():
        if stage in (name, "all"):
            print(f"== {name}")
            fn()
