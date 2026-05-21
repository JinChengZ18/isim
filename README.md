# ISim: sMTJ-based Ising Solver Simulation Framework

A reference implementation of Ising / QUBO benchmarking for stochastic magnetic tunnel junction (sMTJ) p-bit networks. Supports Max-Cut on G-set, TSP on TSPLIB, integer factoring, and comparison against the classical Metropolis-Hastings baseline. Pure Python with a small dependency list; runs on Linux, macOS, and Windows. All commands in this README are single lines so they work identically in bash, zsh, Windows CMD, and PowerShell.

## Layout

```
isim.py                   core framework: SpinBackend, Problem,
                          IsingSolver, multistart, metrics, numba
                          JIT kernels, JSON I/O
problems.py               loaders and random instance generators
                          (G-set, TSPLIB with GEO/EXPLICIT support,
                          factoring, Erdos-Renyi, Sherrington-Kirkpatrick)
plot_style.py             matplotlib style (Arial, Tsinghua-purple
                          palette, thesis-scale font sizes)
fetch_data.py             automatic downloader with multi-mirror
                          fallback
bench_maxcut.py           Max-Cut benchmark driver
bench_maxcut_ablation.py  Max-Cut ablation driver: sweep-budget and
                          annealing-endpoint scans on a fixed instance
bench_tsp.py              TSP benchmark driver (guards n > 20 by default)
bench_tsp_ablation.py     TSP ablation driver: penalty coefficient A scan
bench_tsp_compare.py      TSP comparison driver: QUBO single-spin Ising
                          vs permutation-space cluster-flip SA
perm_tsp.py               Permutation-space TSP solver (2-opt / swap /
                          insert cluster updates under Metropolis
                          acceptance)
bench_factor.py           Integer factoring driver
bench_factor_ablation.py  Bit-budget ablation driver for a fixed M
compare_baselines.py      sMTJ-Gibbs vs Classical SA comparison driver
device_model.py           Behavioral sMTJ p-bit model with five
                          orthogonal non-ideality knobs
                          (drive gain, offset, C2C jitter, plateau,
                          D2D dispersion); calibrated against the
                          Section 2.3 measurements
hardware_metrics.py       Hardware-platform performance models for
                          sMTJ-array, CMOS p-bit ASIC, FPGA SBM,
                          and CPU+Numba; converts solver sweep
                          counts into wall-time TTS and energy
                          per solution
bench_device_ablation.py  Single-knob ablation on a fixed problem;
                          maps non-ideality magnitude onto TTS_99
                          ratio for each of the five device knobs
bench_hardware_compare.py Cross-architecture TTS / energy
                          projection from any summary.csv
demo/                     self-contained pedagogical demo (NumPy +
                          Matplotlib only); see "Toy demonstration"
                          below
```

## Installation

```
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `numba` (JIT acceleration), `tqdm` (progress bars). Numba gives roughly a 100x speedup over pure Python on the inner update loop and is strongly recommended for any production run. If numba is unavailable, pass `--update-mode block` to any driver to use the pure-NumPy block-parallel path, which gives roughly 30x the pure-Python throughput.

## Execution modes

Three update modes are exposed via `--update-mode`:

- `async_numba` (default): JIT-compiled strict-sequential Gibbs or Metropolis updates. Typical throughput 50-100 M spin-updates per second on a single core. Required for large instances (n >= 2000) to finish in reasonable time.
- `block`: greedy-colored block-parallel Gibbs/Metropolis in pure NumPy. Correct by construction: spins in the same color class are non-adjacent so simultaneous sampling agrees with sequential sampling in distribution. Typical throughput 5-20 M updates/s. Preferred when numba is not available.
- `async_python`: reference implementation, roughly 0.2 M updates/s. Intended for validation and debugging; not suitable for production.

Two dynamics via `--dynamics`:

- `gibbs` (default): Glauber-dynamics conditional sampling, the dynamical model of an ideal-sigmoid sMTJ p-bit.
- `metropolis`: classical Metropolis-Hastings acceptance using delta-energy. Used by `compare_baselines.py` as the SA reference.

## Benchmark data

The bench drivers download any missing G-set or TSPLIB instance automatically on first use, trying each configured mirror in sequence until one responds.

### Mirror strategy

Each dataset has a prioritized list of mirrors. Defaults put GitHub-hosted raw files first (HTTPS, CDN-backed, reliable) and keep the Stanford and Heidelberg URLs as secondary fallbacks. G-set instances come from the `0816keisuke/Gset-Max-cut` GitHub mirror as primary source; TSPLIB instances from `mastqe/tsplib` and `pdrozdowski/TSPLib.Net`. TSPLIB parsing supports both coordinate types (`EUC_2D`, `CEIL_2D`, `ATT`, `MAX_2D`, `MAN_2D`, `GEO`) and explicit distance matrices (all row/column and diag/no-diag variants of `EDGE_WEIGHT_FORMAT`).

To override mirrors in restricted networks, set an environment variable: on Windows `set GSET_MIRROR=https://your.mirror/Gset/{name}`, on Unix `export GSET_MIRROR=https://your.mirror/Gset/{name}` (preserve the `{name}` placeholder). The override replaces the default mirror list. To skip the network entirely, place files at `<gset-dir>/G1`, `<tsplib-dir>/berlin52.tsp`, etc., and pass `--no-auto-fetch`.

### Manual pre-fetch via CLI

```
python fetch_data.py gset --instances G1 G14 G22 --output ./gset
python fetch_data.py tsplib --instances berlin52 eil51 --output ./tsplib
python fetch_data.py all --output-root .
```

### Factoring

No external data needed; the driver builds the Ising formulation on the fly for each target integer M.

## Toy demonstration

`demo/toy_demo_maxcut5.py` walks the full annealing flow on a 5-spin Max-Cut graph. The state space is small enough ($2^5 = 32$ states) that the energy landscape can be enumerated exactly, making the demo useful as a teaching aid and as a sanity check on the broader framework. The script depends only on NumPy and Matplotlib and does not import `isim`, `problems`, or any of the bench drivers; the intent is that a reader can run and read the demo as a complete, minimal-viable Ising-annealer reference.

```
cd demo
python toy_demo_maxcut5.py
python schematic_flow.py
```

Outputs land in `demo/out/`:

* `toy_energy_landscape.png` — full enumeration of all 32 spin configurations sorted by energy.
* `toy_annealing_traces.png` — eight independent annealing runs with the geometric `beta(t)` schedule overlaid.
* `toy_optimal_partition.png` — the optimal cut visualised on the graph.
* `ising_flow_schematic.png` — five-stage flow diagram (problem, Ising encoding, p-bit network, annealing, solution).
* `toy_run.npz` — raw arrays for downstream re-plotting.

Both scripts use the same Tsinghua-purple palette as the rest of the production figures, so demo and benchmark plots remain visually consistent when shown side by side.

## Running the benchmarks

### Max-Cut on G-set

```
python bench_maxcut.py --instances G1 G14 G22 --gset-dir ./gset --trials 200 --sweeps 10000 --schedule geometric --beta0 0.1 --betaf 10.0 --update-mode async_numba --jobs 8 --output ./results_maxcut
```

Increase `--sweeps` (e.g., 20000 or 50000) for harder instances such as G14. The `--jobs 8` argument spreads the 200 trials across 8 worker processes; set to the number of available CPU cores.

### Max-Cut ablation (sweep budget and annealing endpoints)

Sweep-budget scan on a single instance (answers whether low success probability is a budget issue or a landscape issue):

```
python bench_maxcut_ablation.py sweeps --instance G14 --gset-dir ./gset --Ts 1000 3000 10000 30000 100000 --beta0 0.1 --betaf 10.0 --trials 100 --update-mode async_numba --jobs 4 --output ./results_maxcut_sweeps
```

Annealing-endpoint scan (checks robustness of main-table results to schedule choice):

```
python bench_maxcut_ablation.py beta --instance G1 --gset-dir ./gset --betaf-values 2 5 10 20 50 --beta0 0.1 --sweeps 10000 --trials 100 --update-mode async_numba --jobs 4 --output ./results_maxcut_beta
```

Each mode writes `sweeps_summary.csv` / `beta_summary.csv`, a dual-axis scan plot (`sweeps_scan.png` / `beta_scan.png`) with success probability on the left axis and TTS_99 on the right log-scale axis, plus per-point JSON results and log files.

### TSP on TSPLIB (small instances only by default)

```
python bench_tsp.py --instances burma14 ulysses16 gr17 fri26 --tsplib-dir ./tsplib --trials 100 --sweeps 50000 --beta0 0.05 --betaf 20.0 --A-margin 2.0 --B 1.0 --update-mode async_numba --jobs 4 --output ./results_tsp
```

The n^2-spin QUBO encoding of TSP is known to converge extremely slowly under SA for n > 20; instances larger than that are refused by default. Pass `--allow-large` to override. For benchmarking, keep n <= 20 (`burma14`, `ulysses16`, `gr17`, `fri26` are all appropriate).

### TSP ablation (penalty coefficient A)

Scan the penalty coefficient A = A_margin * B * max(d) on a single TSP instance to characterize the feasibility-vs-resolution trade-off:

```
python bench_tsp_ablation.py --instance ulysses16 --tsplib-dir ./tsplib --A-margins 1.2 1.5 2.0 3.0 5.0 10.0 --trials 100 --sweeps 50000 --beta0 0.05 --betaf 20.0 --update-mode async_numba --jobs 4 --output ./results_tsp_ablation
```

Output includes `tsp_A_summary.csv`, a dual-axis scan plot showing feasibility rate, best-tour gap, and TTS_99 over the A_margin range, plus per-point JSON results and log files.

### TSP comparison: QUBO vs permutation-space SA

Side-by-side evaluation of the n^2-spin QUBO baseline and a permutation-space cluster-flip alternative (2-opt, swap, insert moves under Metropolis acceptance). The permutation solver stays within the feasibility manifold by construction and serves as the upper bound for what an enhanced sMTJ architecture supporting cluster updates could achieve:

```
python bench_tsp_compare.py --instances burma14 ulysses16 gr17 --tsplib-dir ./tsplib --trials 100 --qubo-sweeps 50000 --perm-sweeps 5000 --update-mode async_numba --jobs 4 --output ./results_tsp_compare
```

Output includes `compare_summary.csv` with per-instance gap, p_success, and TTS_99 for both methods; a dual-axis bar comparison plot (`tsp_compare_bars.png`); and per-instance convergence overlays (`trace_compare_*.png`) showing QUBO energy vs permutation tour length trajectory side-by-side.

### Integer factoring

```
python bench_factor.py --targets 15 21 33 35 51 65 77 91 143 --trials 200 --sweeps 20000 --beta0 0.05 --betaf 30.0 --update-mode async_numba --jobs 4 --output ./results_factor
```

`build_factoring_problem` auto-sizes the bit budget `(bp, bq)` via trial division so that the minimal encoding covering the true factorization is used. A trial is successful only when (a) all auxiliary constraints `z_{i,j} = p_i q_j` hold and (b) the decoded `p * q` equals M with non-trivial factors.

### Factoring bit-budget ablation

```
python bench_factor_ablation.py --M 51 --bq-range 4 7 --trials 200 --sweeps 20000 --beta0 0.05 --betaf 30.0 --update-mode async_numba --jobs 4 --output ./results_factor_ablation
```

Fixes a target M and scans `bq` while holding all other hyperparameters fixed, isolating the effect of encoding size from solver dynamics. Outputs include the scan plot (dual-axis: success probability and TTS_99), the `(p_hat, q_hat)` density plot for the underfit case, per-run JSON, and a summary CSV.

### Comparison vs classical SA

`compare_baselines.py` runs the Gibbs (sMTJ) and Metropolis (classical SA) dynamics on the same problem instance with identical annealing schedule, sweep count, and master seed. Differences in `p_s` and `TTS_99` therefore come entirely from the single-step update rule.

The driver supports three modes, mirroring the three benchmarks in Section 3.3. There is no `random` mode and no synthetic ER/SK instances — the comparison uses exactly the benchmark instance sets.

```
python compare_baselines.py --mode maxcut --instances G1 G14 G22 --gset-dir ./gset --trials 200 --sweeps 10000 --update-mode async_numba --jobs 8 --output ./results_compare_maxcut
```

```
python compare_baselines.py --mode factor --instances 15 21 33 35 51 65 77 91 143 --trials 200 --sweeps 20000 --beta0 0.05 --betaf 30.0 --update-mode async_numba --jobs 4 --output ./results_compare_factor
```

```
python compare_baselines.py --mode tsp --instances burma14 ulysses16 --tsplib-dir ./tsplib --trials 100 --sweeps 50000 --beta0 0.05 --betaf 20.0 --update-mode async_numba --jobs 4 --output ./results_compare_tsp
```

Success criterion is mode-dependent:

* `maxcut`: cut value matches the published BKS (`GSET_BKS` table).
* `factor`: decoded `(p_hat, q_hat)` satisfies `p_hat * q_hat == M` with all auxiliary `z_{ij} = p_i q_j` constraints satisfied. This matches the criterion used in `bench_factor.py`, so the resulting `p_s` values are directly comparable across the two drivers.
* `tsp`: best observed energy across both backends is taken as the target (a placeholder that produces `p_s >= 1/trials` whenever a feasible tour is found by either side; a stricter mode that uses the TSPLIB optimum is left as future work).

Each output directory contains:

* `summary.csv` with columns `instance, n, target, ps_smtj, ps_sa, tts99_smtj, tts99_sa, speedup_sa_over_smtj, energy_min_*, energy_median_*, time_median_*`. A `target` column is empty for `factor` mode because success is judged by decoding rather than by energy threshold.
* `tts_compare.png`, the side-by-side `TTS_99` bar chart. By default the chart skips instances where both dynamics produced `p_s = 0` (controlled by `--filter-no-solve`, on by default; pass `--no-filter-no-solve` to keep them as `n/a` placeholders). The underlying CSV always contains every instance.
* `<instance>/energy_box.png`, per-instance final-energy box plot comparing the two dynamics.
* `<instance>/results.json`, full per-trial energy and timing data for downstream analysis.

## Integrating a non-ideal sMTJ device model

The `SMTJSpin` class in `isim.py` encodes the ideal-device correspondence (`I_bias = 2 * beta * I_0 * h_eff`, `P(+1) = sigma(I_bias / I_0)`). To plug in a device-level behavioral model (switching-time drift, bias offset, thermal jitter, etc.), subclass `SMTJSpin`:

```python
from isim import SMTJSpin

class DeviceSMTJSpin(SMTJSpin):
    def __init__(self, I_0, device_params):
        super().__init__(I_0=I_0)
        from my_device_model import SMTJDevice
        self.device = SMTJDevice(**device_params)

    def sample(self, h_eff, s_cur, beta, rng):
        I_bias = 2.0 * beta * self.I_0 * h_eff
        return self.device.sample_output(I_bias, rng)

    def sample_batch(self, h_eff_arr, s_cur_arr, beta, rng):
        import numpy as np
        I_bias = 2.0 * beta * self.I_0 * h_eff_arr
        return np.array(
            [self.device.sample_output(I, rng) for I in I_bias],
            dtype=np.int8)
```

The `sample()` method takes the current spin value `s_cur`; ideal Gibbs ignores it while Metropolis-style updates use it. When routing through `async_numba` mode, the custom backend is bypassed in favor of the JIT-compiled Gibbs/Metropolis kernels. Use `--update-mode block` or `--update-mode async_python` to actually exercise a custom device model.

## Output artifacts

All outputs are human-readable and git-diffable; no binary archives.

- `summary.csv` — one row per instance or scan point with aggregate metrics (p_success, TTS_99, constraint rates, problem parameters).
- `run_<instance>.json` — per-trial results (energy trajectory, final energy, wall time, seed, full config). Readable in any text editor; loadable via `json.load` or `isim.load_results_json`.
- `states_<instance>.txt` — final spin configurations as a compact 0/1 text matrix, one trial per row.
- `log_<instance>.txt` — copy of the console log for that instance, including per-progress updates and aggregate metrics.
- PNG figures: `trace_*.png` (energy convergence), `hist_*.png` (cut-value distribution), `bar_success.png` (per-target success with TTS_99 overlay on the right log-scale axis), `ablation_scan.png` (bit-budget scan with TTS_99 overlay), `ablation_density_*.png` (pseudo-optimum visualization), `energy_compare_*.png` (box plots of final energies), `tts_compare.png` (TTS bars across instances). All use Arial, Tsinghua-purple palette, and thesis-scale fonts; no filename or section tags appear in titles or legends.

Re-plotting from a JSON output:

```python
import json, matplotlib.pyplot as plt
d = json.load(open("results_maxcut/run_G1.json"))
print(d["meta"]["summary"])     # aggregate metrics
traj0 = d["runs"][0]["trajectory"]
plt.plot(traj0["sweeps"], traj0["energies"])
plt.show()
```

## Hardware-aware evaluation

Two drivers translate the algorithm-level metrics into device- and platform-level figures, matching the chapter section on hardware-aware evaluation.

### Device non-ideality ablation

`bench_device_ablation.py` scans each of the five non-ideality knobs of the `BehavioralSMTJSpin` model on a fixed problem instance:

```
python bench_device_ablation.py --problem-kind er --er-n 14 --er-p 0.30 --trials 100 --sweeps 2000 --update-mode block --output ./results_dev_ablation
```

The driver supports two problem modes. The `er` mode generates an Erdős–Rényi Max-Cut instance and finds its ground state by brute-force enumeration (n ≤ 22), so it requires no external data. The `gset` mode reuses any G-set instance with a known BKS:

```
python bench_device_ablation.py --problem-kind gset --instance G1 --gset-dir ./gset --trials 100 --sweeps 10000 --update-mode block --output ./results_dev_ablation_G1
```

Output: `device_ablation_summary.csv` with one row per (knob, value) pair plus a five-panel figure showing the TTS_99 ratio relative to the ideal-device baseline. The driver runs in `--update-mode block` because the JIT path bypasses `sample_batch`; this is the documented trade-off for exercising any custom backend.

### Cross-architecture comparison

`bench_hardware_compare.py` projects existing `summary.csv` files through four platform models (sMTJ-array, CMOS p-bit ASIC, FPGA SBM, CPU+Numba). The driver accepts one or more summary CSVs and concatenates their instance sets, allowing a single comparison chart to span multiple benchmark families:

```
python bench_hardware_compare.py --summary results_compare_maxcut/summary.csv --output ./results_hw_compare
```

```
python bench_hardware_compare.py --summary results_compare_maxcut/summary.csv results_compare_factor/summary.csv results_compare_tsp/summary.csv --output ./results_hw_compare_full
```

Instances appear in the figure in the order they are listed across the input files; instance-name collisions across files keep the first occurrence. Instances with `p_success = 0` (e.g., G14 on the Section 3.3 calibration) carry no platform-comparison information and are dropped from the figure by default; pass `--no-skip-unsolved` to keep them as empty slots.

The CPU platform is calibrated at runtime from the CSV's `time_median` column; the other three platforms use literature parameters from `hardware_metrics.py`. Output: a CSV of (instance, platform, TTS_99, energy_per_solution) plus a paired bar chart on log axes. No new solver runs are needed; the script operates only on existing summary data.

> **Prerequisites**: the input CSVs must already exist. If they do not, generate them first by running the corresponding benchmark driver. For the comparison shown in the chapter, the three input CSVs come from `compare_baselines.py --mode maxcut|factor|tsp`.

### Plugging in a custom device model

The `BehavioralSMTJSpin` class in `device_model.py` is a reference implementation of a non-ideal sMTJ p-bit calibrated against the Section 2.3 measurements. Custom device models can be registered through the same mechanism without editing `isim.py`:

```python
from isim import SpinBackend, register_spin_backend

class MyDeviceSMTJ(SpinBackend):
    kind = "my_device"
    def sample_batch(self, h_eff_arr, s_cur_arr, beta, rng, idx=None):
        # Custom sampling rule using your device model
        ...

register_spin_backend("my_device", lambda **kw: MyDeviceSMTJ(**kw))
```

After registration, `spin_spec=("my_device", {...})` is resolvable in `multistart()` and in the parallel worker processes.

## Reproducibility

All multi-start runs use `numpy.random.SeedSequence` to derive per-trial seeds from a single `master_seed`. Given the same master seed, problem, and config, trajectories are bit-reproducible regardless of the number of worker processes. The `states_*.txt` and `run_*.json` outputs are sufficient to recompute any figure or summary statistic without re-running the simulation.
