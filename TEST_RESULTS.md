# DeepLens Optimization Benchmarks

This document records the official benchmark evaluation results comparing the **1st-order Adam optimizer** and the **2nd-order Levenberg-Marquardt (LM) optimizer**, with and without **Differentiable Paraxial ABCD Solves**.

---

## 1. Master Benchmark Results

| Benchmark Design | Initial RMS Spot | Standard Adam (No Paraxial Solve) | Adam + Paraxial ABCD Solve | LM + Paraxial ABCD Solve | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Toy 2-Element Doublet**<br>($\text{EFL}=10\text{ mm}, \text{F}/2.0, 30^\circ\text{ FoV}$)<br>[`toy_starting_point.json`](datasets/lenses/toy/toy_starting_point.json) | `598.43 µm` | `540.12 µm`<br>(Slow without paraxial focus) | `112.45 µm`<br>(-81.2% in 200 iters) | **`87.23 µm`**<br>(**On-axis spot: `0.42 µm`**, 30 iters) | **Matches *Generalized Aberrations* baseline** ($0.42\ \mu\text{m}$ on-axis). $c_4 = -0.1935$ solved at Step 0. |
| **2. Cooke Triplet**<br>($\text{EFL}=50\text{ mm}, \text{F}/4.5, 3\text{ elements}$)<br>[`cooke40_inferior.json`](datasets/lenses/cooke40_inferior.json) | `5,476.83 µm` | `4,030.31 µm`<br>(-26.4% in 95.1s, 600 iters) | **`132.68 µm`**<br>(**-97.6% in 20.1s, 200 iters**) | **`271.38 µm`**<br>(**-95.0% in 75.7s, 35 iters**) | **ABCD solve eliminates defocus drift**, delivering ~95–97% aberration reduction in ~20–75s. |
| **3. Cellphone 80° Lens**<br>(13 surfaces, 6 aspheric elements)<br>[`cellphone80deg.json`](datasets/lenses/cellphone/cellphone80deg.json) | `233.49 µm`<br>(Edge: `14.55 µm` RMS / `44.11 µm` Geo) | Requires 5,000-step curriculum | `62.08 µm`<br>(Edge: `8.43 µm` RMS / `40.14 µm` Geo) | **`51.20 µm`**<br>(Edge: **`7.63 µm` RMS / `25.04 µm` Geo**) | **LM's 2nd-order curvature matrix resolves 40+ aspheric parameters**, cutting edge blur by **38%** vs. Adam. |

---

## 2. CLI Reproduction Commands

```bash
# Benchmark 1: Toy 2-Element Doublet (from Generalized Aberrations)
python -m pytest test/test_toy_lens_lm.py -v

# Benchmark 2: Cooke Triplet (Adam vs LM comparison)
python benchmark_cooke_lm_adam.py

# Benchmark 3: Cellphone 80° Lens (Full aspheric design)
python 1_design_geolens_lm.py

# Fast CI Unit Tests (<5 seconds)
pytest test/test_geolens_optim.py -k TestLMOptimizer -v
```

---

## 3. Quick Usage in Python

```python
from deeplens import GeoLens

lens = GeoLens(filename="datasets/lenses/cooke40_inferior.json")

# Option A: Fast 1st-order Adam with Paraxial ABCD Solve
lens.optimize(iterations=200, target_efl=50.0)

# Option B: High-precision 2nd-order Levenberg-Marquardt with Paraxial ABCD Solve
lens.optimize_lm(iterations=35, target_efl=50.0)
```
