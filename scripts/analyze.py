"""Tables, statistics and figures of the paper, generated from results/.

    python scripts/analyze.py

Every number quoted in the manuscript is written to results/numbers.json by
this script, so text, tables and figures come from the same files.
Rounds are converted to hours (one round = 2 min).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from flare.sim import Data, run  # noqa: E402
from flare.spatial import SpatialModel  # noqa: E402

RES = ROOT / "results"
TAB = RES / "tables"
FIG = RES / "figures"
H = 2.0 / 60.0                       # hours per round
CLUSTERED = ["LEACH", "HEED", "CA-CH", "TS-CH", "PEAGC", "CORR-SL"]
FREE = ["FLAT", "FLAT-EA", "FLAT-CS", "FLARE"]
ORDER = CLUSTERED + FREE
# H1 (primary): plain clustering-free collection against every clustered protocol.
# H2 (secondary): the additions made on top of clustering-free collection.
H2 = [("FLAT-EA", "FLAT"), ("FLAT-CS", "FLAT"), ("FLARE", "FLAT"), ("FLARE", "FLAT-EA")]

# figure style: Times-compatible serif, 10 pt, thin marks, recessive grid
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Liberation Serif", "Times New Roman", "DejaVu Serif"],
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10, "xtick.labelsize": 10,
    "ytick.labelsize": 10, "legend.fontsize": 10, "axes.linewidth": 0.6,
    "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e",
    "ytick.color": "#52514e", "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300,
    "savefig.bbox": "tight", "lines.linewidth": 1.6,
})
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#8d8c87"
INK2 = "#52514e"


def fmt(m, s, d=1):
    return f"{m:.{d}f} ± {s:.{d}f}"


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (len(p) - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def paired(df, metric, base, other):
    """Paired comparison other - base over common seeds, in hours."""
    a = df[df.method == other].set_index("seed")[metric]
    b = df[df.method == base].set_index("seed")[metric]
    common = a.index.intersection(b.index)
    d = (a[common] - b[common]).to_numpy(float) * H
    if np.allclose(d, 0):
        return 1.0, 0.0, 0.0, (0.0, 0.0)
    res = stats.wilcoxon(d, zero_method="wilcox", alternative="two-sided")
    pos, neg = (d > 0).sum(), (d < 0).sum()
    rb = (pos - neg) / max(pos + neg, 1)               # matched-pairs rank-biserial (sign form)
    rng = np.random.default_rng(0)
    boot = [np.median(rng.choice(d, len(d))) for _ in range(5000)]
    return float(res.pvalue), float(np.median(d)), float(rb), tuple(np.percentile(boot, [2.5, 97.5]))


# --------------------------------------------------------------------- tables
def main_tables(numbers):
    df = pd.read_csv(TAB / "main_weekB_runs.csv")
    out = {}
    for tgt, ct in [("FL05", "CT05"), ("FL10", "CT10")]:
        sub = df[df.target == tgt]
        rows = []
        for m in ORDER:
            g = sub[sub.method == m]
            rows.append({
                "Method": m,
                "Fidelity lifetime (h)": fmt(g[tgt].mean() * H, g[tgt].std() * H),
                "Compliant time (h)": fmt(g[ct].mean() * H, g[ct].std() * H),
                "FND (h)": fmt(g.FND.mean() * H, g.FND.std() * H),
                "HND (h)": fmt(g.HND.mean() * H, g.HND.std() * H),
                "PDR (%)": fmt(100 * g.pdr.mean(), 100 * g.pdr.std()),
            })
            numbers[f"{tgt}_{m}_FL_h"] = round(g[tgt].mean() * H, 1)
            numbers[f"{tgt}_{m}_FL_sd_h"] = round(g[tgt].std() * H, 1)
            numbers[f"{tgt}_{m}_CT_h"] = round(g[ct].mean() * H, 1)
            numbers[f"{tgt}_{m}_FND_h"] = round(g.FND.mean() * H, 1)
            numbers[f"{tgt}_{m}_HND_h"] = round(g.HND.mean() * H, 1)
            numbers[f"{tgt}_{m}_PDR"] = round(100 * g.pdr.mean(), 1)
        t = pd.DataFrame(rows)
        t.to_csv(TAB / f"table_main_{tgt}.csv", index=False)
        out[tgt] = t
        # statistics: H1 = FLAT vs each clustered protocol (Holm over 6 per metric),
        # H2 = additions on top of clustering-free collection (Holm over 4 per metric)
        srows = []
        for metric in (tgt, ct):
            mk = "FL" if metric == tgt else "CT"
            for fam, pairs in (("H1", [("FLAT", b) for b in CLUSTERED]), ("H2", H2)):
                res = [paired(sub, metric, b, a) for a, b in pairs]
                adj = holm([r[0] for r in res])
                for (a, b), r, pa in zip(pairs, res, adj):
                    srows.append({"Family": fam, "Metric": mk, "Comparison": f"{a} vs {b}",
                                  "Median diff (h)": round(r[1], 1),
                                  "95% CI (h)": f"[{r[3][0]:.1f}, {r[3][1]:.1f}]",
                                  "Rank-biserial": round(r[2], 2), "p (Holm)": pa})
                    numbers[f"{tgt}_{mk}_{a}_vs_{b}_p"] = float(pa)
                    numbers[f"{tgt}_{mk}_{a}_vs_{b}_meddiff_h"] = round(r[1], 1)
                    numbers[f"{tgt}_{mk}_{a}_vs_{b}_rb"] = round(r[2], 2)
                    numbers[f"{tgt}_{mk}_{a}_vs_{b}_ci"] = [round(r[3][0], 1), round(r[3][1], 1)]
        st = pd.DataFrame(srows)
        st.to_csv(TAB / f"table_stats_{tgt}.csv", index=False)
        # relative gains of the clustering-free designs over the best clustered protocol
        g = sub.groupby("method")[[tgt, ct, "FND", "HND"]].mean()
        best_cl = g.loc[CLUSTERED, tgt].idxmax()
        numbers[f"{tgt}_best_clustered"] = best_cl
        for m in FREE:
            for col, key in ((tgt, "FL"), (ct, "CT"), ("HND", "HND"), ("FND", "FND")):
                numbers[f"{tgt}_gain_{key}_{m}_vs_bestcl_pct"] = round(100 * (g.loc[m, col] / g.loc[best_cl, col] - 1), 1)
        numbers[f"{tgt}_best_clustered_HND"] = g.loc[CLUSTERED, "HND"].idxmax()
        bh = numbers[f"{tgt}_best_clustered_HND"]
        for m in FREE:
            numbers[f"{tgt}_gain_HND_{m}_vs_bestHND_pct"] = round(100 * (g.loc[m, "HND"] / g.loc[bh, "HND"] - 1), 1)
    numbers["T_eval_h"] = round(df["T"].iloc[0] * H, 1)
    numbers["n_seeds"] = int(df.seed.nunique())
    # energy ledger shares (FL05 configuration)
    sub = df[df.target == "FL05"]
    led = sub.groupby("method")["ledger_share"].apply(
        lambda s: np.mean([json.loads(x) if isinstance(x, str) else x for x in s], axis=0))
    led = pd.DataFrame(led.tolist(), index=led.index,
                       columns=["Sensing/CPU", "Control", "Own updates", "CH duty", "Relaying", "E2E ACK"])
    led.loc[ORDER].round(3).to_csv(TAB / "energy_ledger.csv")
    for m in ORDER:
        numbers[f"ledger_{m}"] = led.loc[m].round(3).to_dict()
    return out, led


def ablation_table(numbers):
    df = pd.read_csv(TAB / "ablation_weekB_runs.csv")
    rows = []
    for tgt, ct in [("FL05", "CT05"), ("FL10", "CT10")]:
        sub = df[df.target == tgt]
        for v in ["FLARE", "FLARE-noSIL", "FLARE-noRT", "FLARE-geo", "FLAT"]:
            g = sub[sub.method == v]
            rows.append({"Target": tgt, "Variant": v, "Fidelity lifetime (h)": fmt(g[tgt].mean() * H, g[tgt].std() * H),
                         "Compliant time (h)": fmt(g[ct].mean() * H, g[ct].std() * H),
                         "FND (h)": fmt(g.FND.mean() * H, g.FND.std() * H),
                         "HND (h)": fmt(g.HND.mean() * H, g.HND.std() * H)})
            numbers[f"abl_{tgt}_{v}_FL_h"] = round(g[tgt].mean() * H, 1)
            numbers[f"abl_{tgt}_{v}_CT_h"] = round(g[ct].mean() * H, 1)
            numbers[f"abl_{tgt}_{v}_FND_h"] = round(g.FND.mean() * H, 1)
        for v in ["FLARE-noSIL", "FLARE-noRT", "FLARE-geo", "FLAT"]:
            p, md, rb, ci = paired(sub, tgt, v, "FLARE")
            numbers[f"abl_{tgt}_FLARE_vs_{v}_p"] = p
            numbers[f"abl_{tgt}_FLARE_vs_{v}_meddiff_h"] = round(md, 1)
    pd.DataFrame(rows).to_csv(TAB / "table_ablation.csv", index=False)


def sensitivity_table(numbers):
    f = TAB / "sensitivity_weekB_runs.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    grp = df.groupby(["factor", "value", "method"])[["FL05", "CT05", "HND"]]
    g = (grp.mean() * H).round(1)
    sd = (grp.std() * H).round(1).add_suffix("_sd")
    g = g.join(sd).reset_index()
    g.to_csv(TAB / "table_sensitivity.csv", index=False)
    numbers["sensitivity"] = g.to_dict(orient="records")


def signals_table(numbers):
    f = TAB / "signals_weekB_runs.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    rows = []
    for sig in ["hum", "light"]:
        sub = df[df.signal == sig]
        for m in ORDER:
            g = sub[sub.method == m]
            rows.append({"Signal": sig, "Method": m, "FL (h)": fmt(g.FL05.mean() * H, g.FL05.std() * H),
                         "CT (h)": fmt(g.CT05.mean() * H, g.CT05.std() * H),
                         "HND (h)": fmt(g.HND.mean() * H, g.HND.std() * H)})
            numbers[f"sig_{sig}_{m}_FL_h"] = round(g.FL05.mean() * H, 1)
            numbers[f"sig_{sig}_{m}_CT_h"] = round(g.CT05.mean() * H, 1)
            numbers[f"sig_{sig}_{m}_HND_h"] = round(g.HND.mean() * H, 1)
            numbers[f"sig_{sig}_{m}_FND_h"] = round(g.FND.mean() * H, 1)
        means = sub.groupby("method").FL05.mean()
        best_cl = means.loc[CLUSTERED].idxmax()
        numbers[f"sig_{sig}_best_clustered"] = best_cl
        for m in FREE:
            numbers[f"sig_{sig}_gain_{m}_vs_bestcl_pct"] = round(100 * (means[m] / means[best_cl] - 1), 1)
        p, md, rb, ci = paired(sub, "FL05", best_cl, "FLAT")
        numbers[f"sig_{sig}_FLAT_vs_bestcl_p"] = p
        numbers[f"sig_{sig}_FLAT_vs_bestcl_meddiff_h"] = round(md, 1)
    pd.DataFrame(rows).to_csv(TAB / "table_signals.csv", index=False)


# -------------------------------------------------------------------- figures
W1 = 3.2                               # one journal column (inches)
W2 = 6.7                               # full text width
COL = {"FLARE": "#2a78d6", "FLAT": "#eb6834", "FLAT-EA": "#1baf7a"}   # categorical slots 1-3
COL_CL = "#eda100"                     # slot 4: best clustered protocol
DASH = {"FLARE": "-", "FLAT": "--", "FLAT-EA": "-.", "CL": ":"}


def traces(numbers):
    """Seed-0 trajectories on week B for the clustering-free designs and the best clustered protocol."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from experiments import tuned_cfg
    best_cl = numbers["FL05_best_clustered"]
    out, data = {}, None
    for m in [best_cl, "FLAT", "FLAT-EA", "FLARE"]:
        cfg = tuned_cfg(m, "FL05")
        if data is None:
            data = Data(ROOT, cfg)
        out[m] = run(data, m, 0, cfg, keep_series=True)["series"]
    return out, data, best_cl


def validity_check(data, numbers):
    """With practically unlimited energy no design may lose fidelity: the
    fidelity lifetime then measures energy, not the predictability of the trace."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from experiments import tuned_cfg
    for m in [numbers["FL05_best_clustered"], "FLAT", "FLAT-EA"]:
        cfg = tuned_cfg(m, "FL05", E0=1.0)
        r = run(data, m, 0, cfg, keep_series=True)
        numbers[f"inf_energy_{m}_max_roll"] = round(float(r["series"]["roll"].max()), 2)
        numbers[f"inf_energy_{m}_FL05_h"] = round(r["FL05"] * H, 1)


def diagnose(data, numbers):
    """Run 0: which locations carry the error at the moment fidelity is lost?
    A live node whose sink value is off by more than its threshold is not
    reporting (silent), because a reporting node is known within epsilon."""
    sys.path.insert(0, str(ROOT / "scripts"))
    from experiments import tuned_cfg
    out = {}
    for m, e0 in [("FLAT", 0.006), ("FLAT-EA", 0.006), ("FLARE", 0.006),
                  ("FLAT", 0.012), ("FLAT-EA", 0.012), ("FLARE", 0.012)]:
        cfg = tuned_cfg(m, "FL05", E0=e0)
        r = run(data, m, 0, cfg, keep_series=True)
        s, t = r["series"], r["FL05"]
        rec = {"FL_h": round(t * H, 1), "FND_h": round(r["FND"] * H, 1)}
        if t < r["T"]:
            w = slice(max(t - 30, 0), min(t + 30, r["T"]))
            err = np.sqrt(np.mean((s["est"][w] - data.test[w]) ** 2, axis=0))
            top = np.argsort(-err)[:3]
            rec["alive_at_FL"] = int(s["alive"][t])
            rec["worst"] = [{"mote": int(data.ids[i]), "rmse": round(float(err[i]), 2),
                             "state": "dead" if 0 <= s["death"][i] <= t else "live, not reporting"
                             if err[i] > cfg.eps_bar else "reporting"} for i in top]
            rec["worst_share_pct"] = round(100 * float(np.sum(err[top] ** 2) / np.sum(err ** 2)), 1)
        out[f"{m}@{e0}"] = rec
    numbers["diagnosis_run0"] = out


def early_losses(data, numbers):
    """Runs (delta = 0.5) that lose fidelity before any node has died: which
    location carries the error at that moment?"""
    sys.path.insert(0, str(ROOT / "scripts"))
    from experiments import tuned_cfg
    df = pd.read_csv(TAB / "main_weekB_runs.csv")
    sub = df[(df.target == "FL05") & (df.FL05 < df.FND)]
    numbers["early_loss_counts"] = {m: int((sub.method == m).sum()) for m in ORDER}
    numbers["early_loss_median_h"] = {m: round(float(sub[sub.method == m].FL05.median() * H), 1)
                                      for m in ORDER if (sub.method == m).any()}
    worst = {}
    for m in ["LEACH", "CA-CH", "TS-CH", "PEAGC"]:
        cfg = tuned_cfg(m, "FL05")
        for seed in sub[sub.method == m].seed:
            r = run(data, m, int(seed), cfg, keep_series=True)
            t = r["FL05"]
            w = slice(max(t - 30, 0), t + 1)
            err = np.sqrt(np.mean((r["series"]["est"][w] - data.test[w]) ** 2, axis=0))
            k = int(data.ids[int(np.argmax(err))])
            worst[k] = worst.get(k, 0) + 1
    numbers["early_loss_worst_mote"] = {str(k): v for k, v in sorted(worst.items(), key=lambda kv: -kv[1])}
    i21 = int(np.flatnonzero(data.ids == 21)[0])
    x = data.test[:, i21]
    numbers["mote21_jump"] = [round(float(x[179]), 1), round(float(x[185]), 1)]   # 06:58 and 07:10


def _style(m, best_cl):
    return (COL_CL, DASH["CL"]) if m == best_cl else (COL[m], DASH[m])


def fig_layout(data, numbers):
    sp = SpatialModel(data.hist, data.pos)
    n = data.n
    r = sp.irreplaceability(frozenset(range(n)), range(n), peak=True)
    L = data.links
    fig, ax = plt.subplots(figsize=(W1, 2.55))
    ax.grid(False)
    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            if L.fwd[i, j] >= 0.5 and L.fwd[j, i] >= 0.5:
                ax.plot(*zip(L.xy[i], L.xy[j]), color="#d5d4cf", lw=0.5, zorder=1)
    sc = ax.scatter(data.pos[:, 0], data.pos[:, 1], c=r, cmap="Blues", vmin=0, vmax=np.quantile(r, 0.95),
                    s=36, edgecolors="#52514e", linewidths=0.5, zorder=3)
    for i in np.argsort(-r)[:4]:
        right = data.pos[i, 0] > 30
        ax.annotate(f"{data.ids[i]}", data.pos[i], xytext=(0 if right else 4, -15 if right else 4), textcoords="offset points",
                    fontsize=10, color="#0b0b0b", ha="center" if right else "left")
    ax.scatter(*L.xy[0], marker="s", s=60, color="#eb6834", edgecolors="#52514e", linewidths=0.5, zorder=4)
    ax.annotate("Sink", L.xy[0], xytext=(-6, -12), textcoords="offset points", fontsize=10, ha="right")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect("equal")
    cb = fig.colorbar(sc, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("Irreplaceability (°C)")
    cb.outline.set_linewidth(0.5)
    fig.savefig(FIG / "fig2_layout.png")
    plt.close(fig)
    numbers["r_peak_min"], numbers["r_peak_med"], numbers["r_peak_max"] = (
        round(float(r.min()), 2), round(float(np.median(r)), 2), round(float(r.max()), 2))
    numbers["r_peak_q90"] = round(float(np.quantile(r, 0.9)), 2)
    numbers["r_top_motes"] = [int(data.ids[i]) for i in np.argsort(-r)[:6]]
    numbers["links_good"] = int(((L.fwd[1:, 1:] >= 0.5) & (L.fwd[1:, 1:].T >= 0.5)).sum() // 2)
    numbers["n_nodes_B"] = int(n)
    numbers["sink_neighbours_prr05"] = int((L.fwd[1:, 0] >= 0.5).sum())
    raw = data.prr.copy()
    np.fill_diagonal(raw, 0)
    numbers["reverse_missing_links"] = int(((raw >= 0.1) & (raw.T == 0)).sum())
    nxt, _ = L.shortest_paths(np.arange(1, n + 1))
    hops = np.array([len(L.path(nxt, i) or []) for i in range(1, n + 1)])
    numbers["hops_mean"] = round(float(hops.mean()), 2)
    numbers["hops_max"] = int(hops.max())
    numbers["one_hop_share_pct"] = round(100 * float((hops == 1).mean()), 1)
    for e in (0.4, 0.6, 0.8):
        g = int(np.argmin(np.abs(data.grid - e)))
        numbers[f"dps_rate_eps{int(e * 10):02d}_pct"] = round(100 * float(data.curve[g].mean()), 1)


def fig_timeseries(tr, best_cl, numbers):
    order = [best_cl, "FLAT", "FLAT-EA", "FLARE"]
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for m in order:
        c, ls = _style(m, best_cl)
        s = tr[m]["roll"]
        ax.plot(np.arange(len(s)) * H / 24, s, color=c, ls=ls, lw=1.4, label=m)
    ax.axhline(0.5, color=INK2, lw=0.8)
    ax.set_ylim(0, 2.0)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Field RMSE (°C)")
    ax.legend(frameon=False, loc="upper left", handlelength=2.2)
    fig.savefig(FIG / "fig5_rmse.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for m in order:
        c, ls = _style(m, best_cl)
        a = tr[m]["alive"]
        ax.plot(np.arange(len(a)) * H / 24, a, color=c, ls=ls, lw=1.4, label=m)
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Live nodes")
    ax.legend(frameon=False, loc="lower left", handlelength=2.2)
    fig.savefig(FIG / "fig6_alive.png")
    plt.close(fig)
    numbers["silent_mean_first_day"] = round(float(np.mean(tr["FLARE"]["silent"][:720])), 1)
    numbers["silent_max"] = int(np.max(tr["FLARE"]["silent"]))


def fig_distribution(numbers):
    df = pd.read_csv(TAB / "main_weekB_runs.csv")
    sub = df[df.target == "FL05"]
    fig, ax = plt.subplots(figsize=(W1, 2.45))
    for k, m in enumerate(ORDER):
        v = sub[sub.method == m].FL05.to_numpy() * H
        jitter = (np.random.default_rng(k).random(len(v)) - 0.5) * 0.4
        c = "#2a78d6" if m in FREE else GRAY
        ax.scatter(np.full(len(v), k) + jitter, v, s=8, color=c, alpha=0.75, linewidths=0)
        ax.plot([k - 0.3, k + 0.3], [np.median(v)] * 2, color="#0b0b0b", lw=1.4)
    ax.set_xticks(range(len(ORDER)))
    ax.set_xticklabels(ORDER, rotation=60, ha="right")
    ax.set_ylabel("Fidelity lifetime (h)")
    ax.grid(axis="x", visible=False)
    fig.savefig(FIG / "fig3_distribution.png")
    plt.close(fig)


def fig_ledger(led):
    cats = ["CH duty", "Control", "Own updates", "Relaying", "E2E ACK", "Sensing/CPU"]
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    fig, ax = plt.subplots(figsize=(W1, 3.0))
    left = np.zeros(len(ORDER))
    for c, col in zip(cats, colors):
        v = led.loc[ORDER, c].to_numpy() * 100
        ax.barh(range(len(ORDER)), v, left=left, color=col, edgecolor="white", linewidth=0.8,
                height=0.72, label=c)
        left += v
    ax.set_yticks(range(len(ORDER)))
    ax.set_yticklabels(ORDER)
    ax.invert_yaxis()
    ax.set_xlabel("Share of energy spent (%)")
    ax.set_xlim(0, 100)
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.4, -0.2), columnspacing=0.8,
              handlelength=1.2)
    fig.savefig(FIG / "fig4_ledger.png")
    plt.close(fig)


def fig_framework():
    """Block diagram of the evaluation framework (original drawing)."""
    fig, ax = plt.subplots(figsize=(W2, 3.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 42.5)
    ax.axis("off")

    def box(x, y, w, h, text, face="#f4f3f0", edge="#52514e"):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.3,rounding_size=1.0",
                                                       facecolor=face, edgecolor=edge, linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.9, shrinkA=0, shrinkB=0))

    box(0.5, 25, 20, 16, "Intel Lab trace:\nreadings,\npositions, PRR")
    box(24, 25, 19, 16, "Cleaning,\n2-min bins,\nscenarios A, B")
    box(46.5, 25, 28, 16, "Shared simulator core:\ndual prediction, ETX,\nARQ, end-to-end ACK,\nledger, sink inference",
        face="#e6eefb", edge="#2a78d6")
    box(78, 25, 21.5, 16, "Metrics:\nfidelity lifetime,\ncompliant time,\nFND, HND, PDR")
    arrow(20.9, 33, 23.6, 33)
    arrow(43.4, 33, 46.1, 33)
    arrow(74.9, 33, 77.6, 33)
    box(0.5, 1, 23, 16, "Protocol:\ntune on scenario A,\nreport on scenario B,\n30 paired seeds")
    box(27, 1, 23, 16, "Clustered:\nLEACH, HEED,\nCA-CH, TS-CH,\nPEAGC, CORR-SL")
    box(53.5, 1, 23, 16, "Clustering-free:\nFLAT, FLAT-EA,\nFLAT-CS, FLARE")
    box(80, 1, 19.5, 16, "Paired Wilcoxon\ntests, Holm\ncorrection")
    arrow(42, 17.4, 54, 24.6)
    arrow(63, 17.4, 63, 24.6)
    arrow(89, 24.6, 89, 17.4)
    fig.savefig(FIG / "fig1_framework.png")
    plt.close(fig)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    numbers = {}
    _, led = main_tables(numbers)
    ablation_table(numbers)
    sensitivity_table(numbers)
    signals_table(numbers)
    tr, data, best_cl = traces(numbers)
    validity_check(data, numbers)
    diagnose(data, numbers)
    early_losses(data, numbers)
    fig_framework()
    fig_layout(data, numbers)
    fig_timeseries(tr, best_cl, numbers)
    fig_distribution(numbers)
    fig_ledger(led)
    tuned = json.loads((RES / "tuned.json").read_text())
    numbers["tuned"] = {m: {t: v["setting"] for t, v in d.items()} for m, d in tuned.items()}
    numbers["weekA_FL05_h"] = {m: round(d["FL05"]["weekA_mean"] * H, 1) for m, d in tuned.items()}
    numbers["weekA_FL10_h"] = {m: round(d["FL10"]["weekA_mean"] * H, 1) for m, d in tuned.items()}
    (RES / "numbers.json").write_text(json.dumps(numbers, indent=1, default=float))
    print(json.dumps({k: v for k, v in numbers.items() if "gain" in k or "best" in k}, indent=1))


if __name__ == "__main__":
    main()
