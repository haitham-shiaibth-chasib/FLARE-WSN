"""Parallel execution of (method, seed, config) jobs with on-disk caching."""
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .sim import Config, Data, run

ROOT = Path(__file__).resolve().parents[1]
_DATA = {}


def code_version() -> str:
    """Fingerprint of the simulator source, so cached results never outlive a code change."""
    h = hashlib.sha1()
    for f in sorted((ROOT / "flare").glob("*.py")):
        h.update(f.read_bytes())
    return h.hexdigest()[:10]


def _key(cfg: Config, method: str, seed: int) -> str:
    blob = json.dumps({"cfg": asdict(cfg), "m": method, "s": seed, "code": code_version()},
                      sort_keys=True, default=str)
    return hashlib.sha1(blob.encode()).hexdigest()[:16]


def _data_for(cfg: Config) -> Data:
    sig = (cfg.signal, cfg.scenario, cfg.ideal_links, cfg.nlms_p, cfg.nlms_mu, tuple(cfg.grid),
           cfg.extra.get("eps_scale", 1.0))
    if sig not in _DATA:
        _DATA[sig] = Data(ROOT, cfg)
    return _DATA[sig]


def _job(args):
    cfg, method, seed, out = args
    path = Path(out) / f"{_key(cfg, method, seed)}.json"
    if path.exists():
        return json.loads(path.read_text())
    res = run(_data_for(cfg), method, seed, cfg)
    res["config"] = {k: v for k, v in asdict(cfg).items() if k != "grid"}
    res["code_version"] = code_version()
    path.write_text(json.dumps(res, default=float))
    return res


def run_many(jobs, out=ROOT / "results" / "raw", workers=None):
    """jobs: iterable of (Config, method, seed). Returns list of result dicts."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(c, m, s, str(out)) for c, m, s in jobs]
    workers = workers or max(1, (os.cpu_count() or 2))
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(_job, jobs, chunksize=1))
