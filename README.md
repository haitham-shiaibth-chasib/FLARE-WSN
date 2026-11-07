# Fidelity lifetime of clustered and clustering-free collection in dual-prediction WSNs

Code, data and results for the paper by Haitham Shiaibth Chasib
(General Directorate of Education in Babylon, Ministry of Education, Iraq), *"Fidelity Lifetime of Clustered and Clustering-Free Collection in Dual-Prediction Wireless Sensor Networks"*.

The study asks whether clustering still extends network lifetime when nodes already run a dual
prediction scheme, and it measures lifetime by **fidelity lifetime**: the time until the 6-hour windowed
error of the field rebuilt at the sink exceeds a tolerance δ. Everything runs on the Intel Berkeley Lab
trace, which provides real readings, mote positions and measured, asymmetric link qualities. The links are
not drawn from a distance model.

## Main results (held-out period, 30 paired runs)

| Policy | Fidelity lifetime, δ = 0.5 °C (h) | Fidelity lifetime, δ = 1.0 °C (h) |
|---|---|---|
| Best clustered (HEED at both tolerances) | 57.0 | 69.6 |
| FLAT (clustering-free, ETX tree) | 81.8 | 103.3 |
| FLAT-EA (FLAT plus an energy-aware relay penalty) | 82.0 | 104.4 |
| FLARE (irreplaceability-aware; reported as a negative result) | 81.3 | 82.0 |

FLAT outlived all six clustering protocols in every paired run at both tolerances (Holm-adjusted
p < 0.0001 for each comparison). FLARE did not improve on FLAT. The complete tables are in
`results/tables/`, and every number quoted in the paper is in `results/numbers.json`.

The result belongs to a dense indoor deployment in which 81 % of the motes reach the sink in one hop and
every link is shorter than the crossover distance of the radio model. In that setting a cluster head adds
a hop to every member update and costs about 44 µJ per node and hour more than clustering-free collection
(`results/tables/energy_rate_uJ_per_node_hour.csv`). The ranking does not depend on dual prediction (it
also holds with ε = 0), but it turns when all distances are stretched six-fold (see the sensitivity table).

## Repository layout

```
flare/            simulator package
  energy.py       first-order and CC2420 radio models, packet sizes, CPU cost
  predictors.py   NLMS dual predictor (node and sink copies)
  network.py      measured links, ARQ, ETX shortest-path routing
  spatial.py      sink-side ridge inference and node irreplaceability
  methods.py      the ten policies (LEACH, HEED, CA-CH, TS-CH, PEAGC, CORR-SL, FLAT, FLAT-EA, FLAT-CS, FLARE)
  sim.py          round-based trace-driven simulator and metrics
  runner.py       parallel runs with a per-run cache keyed on config, seed and source hash
scripts/
  download_data.py  fetch the Intel Lab files and check their SHA-256
  preprocess.py     cleaning, 2-min resampling, scenarios A (tuning) and B (held out)
  experiments.py    tune | main | ablation | sens | signals
  analyze.py        tables, statistics, figures, results/numbers.json
data/processed/   processed traces (.npy) and preprocess_report.json (every cleaning decision)
results/raw_runs.zip  one JSON per simulation run (config, seed, code version, metrics), 3,200 runs
results/tables/   per-run CSVs and the paper's tables
results/figures/  the paper's figures (fig1 to fig6)
tests/            unit tests of the building blocks
```

## Reproducing the paper

The code needs Python 3.10 or later.

```bash
pip install -r requirements.txt
./reproduce.sh            # download, preprocess, test, all experiments, analysis
```

The per-run JSON files are shipped as `results/raw_runs.zip`; `reproduce.sh` unpacks them into
`results/raw/`, where each run is cached under a hash of its configuration, its seed and the simulator source.
Re-running the scripts therefore reuses finished runs unless the code changes. Delete `results/raw/` to
recompute everything; this takes about an hour on two cores. Individual stages can be run separately:

```bash
python scripts/experiments.py tune       # scenario A: choose every policy's parameters -> results/tuned.json
python scripts/experiments.py main       # scenario B: 30 paired seeds per policy and tolerance
python scripts/experiments.py ablation
python scripts/experiments.py sens
python scripts/experiments.py signals
python scripts/experiments.py burst      # bursty-loss rows only (also part of 'sens')
python scripts/analyze.py                # tables, figures, numbers.json
```

If the MIT server cannot be reached from your network, download `data.txt.gz`, `mote_locs.txt` and
`connectivity.txt` from <https://db.csail.mit.edu/labdata/labdata.html> by hand and put them in
`data/raw/`. `download_data.py` then checks their checksums.

## Protocol in brief

- **Data.** Scenario A (tuning) has 28 Feb to 3 Mar 2004 as history and 3 to 10 Mar as evaluation. Scenario B
  (held out) has 7 to 10 Mar as history and 10 to 15 Mar as evaluation. One round is 2 minutes.
- **Tuning.** Every policy's threshold ε, and its own parameters (τ, λ, δ_d), are chosen on scenario A
  separately for each δ, using 10 runs per candidate. FLARE gets the largest grid.
- **Evaluation.** Scenario B with 30 paired seeds. Runs are compared with the two-sided Wilcoxon signed-rank
  test and Holm correction. Effect sizes are the median paired difference with a bootstrap 95 % interval and
  matched-pairs rank-biserial correlation (Kerby's simple difference formula, computed on the ranks).
  Holm's correction is applied within each family (6 comparisons in H1, 4 in H2) separately for each
  metric and tolerance. SciPy uses the exact null distribution when no two absolute differences tie and the
  normal approximation otherwise.
- **Shared core.** All policies use the same core: NLMS dual prediction (order 4, μ = 0.1), end-to-end
  acknowledgements, ETX routing over measured PRR with 3 ARQ tries, the same control accounting, and the
  same sink-side reconstruction of every location.
- **Idealisations.** In the main runs the end-to-end acknowledgement, the head's confirmation beacon and
  the per-epoch status/join/schedule packets are charged energy but never lost, and the sink knows which
  nodes are alive and how much energy they have. Hop-level transmissions and their link-layer ACKs are
  lossy. A sensitivity run (`extra={"ack_loss": True}`) sends the end-to-end ACK back hop by hop over the
  measured links with ARQ; a node that misses it re-sends its reading and both predictor copies restart.
- **Pairing.** Run *s* of every policy uses seed *s*, so the energy spread is identical across policies.
  Link losses come from the same seed but are drawn packet by packet, so they diverge once policies send
  different packets; the pairing is on the run, not on individual losses.

## Implementation notes

- Baselines are reimplemented from their published descriptions. The code states every adaptation; for
  example, TS-CH uses traffic from past epochs only.
- When the reverse direction of a link was never measured, the forward PRR is reused for it.
- Gaps in the trace are filled by time interpolation. The interpolated share per scenario is recorded in
  `data/processed/preprocess_report.json`.
- The paper calls the policies by the same names as the code (`FLAT`, `FLAT-EA`, `FLARE`, ...).

## Baseline settings, tuning budget and seeds

Every policy runs on the same core and differs only in the three decisions made at each epoch
(cluster heads, per-node thresholds, relay penalty). The fixed rules of each baseline, as coded in
`flare/methods.py`, are:

| Policy | Fixed rules (not tuned) | Tuned on scenario A |
|---|---|---|
| LEACH | threshold T(n) of [Heinzelman et al.] applied to the set G of nodes that have not served in the current cycle of 1/p epochs (G refilled every cycle), CH share p = 0.10 | ε ∈ {0.4, …, 0.8} |
| HEED | initial probability 0.05 scaled by residual energy, p_min = 1e-4, cluster radius 7 m; simplified: a node that wins the draw is a final head at once (no tentative stage) | ε |
| CA-CH | top 10 % by residual energy / coverage cost (sensing radius 6 m), 7 m exclusion radius | ε |
| TS-CH | score 0.4 traffic + 0.3 traffic stability + 0.3 residual energy (min-max scaled); traffic from past epochs only | ε |
| PEAGC | Leiden partition (modularity) of an energy-distance graph; per community the head maximises 0.4 energy + 0.2 centrality + 0.2 sink proximity + 0.2 predicted survivability, and must keep ≥ 10 % of E0 after the predicted epoch cost | ε |
| CORR-SL | HEED + correlation sleeping; the six nearest neighbours of a sleeper stay awake; sleeping nodes are not eligible as heads | ε, τ ∈ {0.2, 0.3, 0.4} |
| FLAT | ETX shortest-path tree, no heads | ε |
| FLAT-EA | relay cost ETX·(1 + λ·min(E0/E − 1, 20)) | ε, λ ∈ {0.5, 1, 3} |
| FLAT-CS | FLAT + correlation sleeping | ε, τ |
| FLARE | Eqs. (11)–(12) of the paper, γ = 1, κ = 1 | ε, δ_d ∈ {0.3, 0.35, 0.4}, λ ∈ {1, 3} |

- **Tuning budget.** 10 runs per candidate on scenario A: 5 candidates for LEACH, HEED, CA-CH, TS-CH,
  PEAGC and FLAT; 15 for CORR-SL, FLAT-CS and FLAT-EA; 30 for FLARE. The candidate with the highest mean
  fidelity lifetime is frozen separately for each δ; the chosen values are in `results/tuned.json`.
- **Seeds.** Run *s* uses random seed *s*: seeds 0–9 on scenario A (tuning and the 10-run sweeps of the
  sensitivity and other-signal tables) and seeds 0–29 on scenario B (main comparison and ablation).
  Run *s* of one policy is paired with run *s* of every other policy.
- **Reporting.** All tables give mean ± standard deviation; percentage gains are computed from
  unrounded means and values are rounded half up to one decimal. Confidence intervals and effect sizes
  for every paired comparison are in `results/tables/table_stats_FL05.csv` and `table_stats_FL10.csv`.

## Bursty links

`flare/network.py` has an optional Gilbert-Elliott model (`BurstyLinks`). Every directed link is good or
bad, the state changes only between rounds (so the ARQ retries of one round share it), a link is bad 20 %
of the time with one tenth of its measured PRR, and the good-state probability is raised so that the
long-term mean equals the measured PRR. It is switched on with `extra={"burst_len": L}`, where L is the
mean length of a bad period in rounds; the paper uses L = 1 and L = 15 (2 and 30 minutes). With the option
off the simulator draws exactly the same random numbers as without the model.

## Checks of the modelling assumptions

The sensitivity stage (`python scripts/experiments.py sens`) also runs, for HEED, LEACH, FLAT, FLAT-EA
and FLARE with 10 paired seeds each:

- `listen`: `"none"` removes the CH listening cost; `"all"` also charges every relay one listening slot per
  node that may route through it, so that listening is charged symmetrically;
- `ack_loss`: lossy end-to-end acknowledgements (see above);
- `dist_scale`: all distances multiplied by 3 or 6, so that the amplifier term of the radio model matters;
- `eps_bar = 0`: no dual prediction, every node reports every reading;
- `ideal_links`: the measured usable links become loss-free; topology and sink position are unchanged;
- `rc`: HEED cluster radius 5, 10 and 14 m.

`results/tables/table_sensitivity_tests.csv` gives, for every row, how many paired runs FLAT and FLAT-EA
were ahead of HEED and LEACH, the median difference and the Wilcoxon p value. The energy ledger has eight
headings; the three CH headings (listening, reception of member updates with aggregation, forwarding of
member updates) are reported separately, and a head's own reading in its aggregate is booked as an own
update.

All runs in `results/raw_runs.zip` were produced by the same version of the simulator source
(`code_version` in every JSON file).

## Data licence and citation

The Intel Lab data were collected by P. Bodik, W. Hong, C. Guestrin, S. Madden, M. Paskin and R. Thibaux
(Intel Berkeley Research Lab, 2004). Please acknowledge them if you use the data. The code in this repository
is released under the MIT licence.
