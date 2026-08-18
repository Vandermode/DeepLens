# DeepLens Optimization Benchmarks

Official benchmark results directly recorded from the GPU test suite comparing **1st-order Adam** against **2nd-order Levenberg-Marquardt (LM)** in DeepLens.

---

## Master Benchmark Results

| Benchmark Design | Initial Baseline RMS | Adam (1st-Order) | Levenberg-Marquardt (2nd-Order) | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- |
| **1. Toy 2-Element Doublet**<br>($\text{EFL}=10\text{ mm}, \text{F}/2.0, 30^\circ\text{ FoV}$)<br>[`datasets/lenses/toy/toy_starting_point.json`](datasets/lenses/toy/toy_starting_point.json) | `598.43 µm` | **`565.65 µm`** (500 iters, 143.7s)<br>*Global RMS: -5.5%* | **`651.87 µm`** (30 iters, 311.6s)<br>*(On-Axis: `88.32 µm`, Mid: `78.30 µm`, Edge: `245.66 µm`)* | **Adam achieved a better global full-field RMS** (`565 µm` vs `651 µm`), while LM aggressively sharpened the central field (`88 µm`) at the expense of unweighted edge rays. |
| **2. Cooke Triplet**<br>($\text{EFL}=50\text{ mm}, \text{F}/4.5, 3\text{ elements}$)<br>[`datasets/lenses/cooke40_inferior.json`](datasets/lenses/cooke40_inferior.json) | `5,476.83 µm` | **`4,030.31 µm`** (600 iters, 95.1s)<br>*Aberration reduction: -26.4%* | **`361.88 µm`** (35 iters, 75.7s)<br>*(Aberration reduction: **-93.4%**)* | **LM converged 11× sharper than Adam** under matched ~75–95s time budget, pulling the system out of heavy initial defocus. |
| **3. Cellphone 80° Lens**<br>(13 surfaces, 6 aspheric elements)<br>[`datasets/lenses/cellphone/cellphone80deg.json`](datasets/lenses/cellphone/cellphone80deg.json) | `233.47 µm`<br>(Center: `6.33 µm`<br>Edge: `14.55 µm`) | **`62.08 µm`** (800 iters, 767.6s)<br>*(Center: `4.39 µm`, Edge: `8.43 µm`)* | **`227.27 µm`** (40 iters, 3520.3s)<br>*(Center: **`4.16 µm`**, Mid: **`3.65 µm`**, Edge: `9.58 µm`)* | **Adam achieved superior global edge clearance convergence**, while LM delivered sharper on-axis focus (`4.16 µm` center). |

---

## CLI Reproduction Commands

```bash
# Benchmark 1: Toy 2-Element Doublet
python -m pytest test/test_toy_lens_lm.py -v

# Benchmark 2: Cooke Triplet
python benchmark_cooke_lm_adam.py

# Benchmark 3: Cellphone 80° Lens
python 1_design_geolens_lm.py

# Fast CI Unit Tests (<5 seconds)
pytest test/test_geolens_optim.py -k TestLMOptimizer -v
```

---

## Quick Usage

```python
from deeplens import GeoLens

lens = GeoLens(filename="datasets/lenses/cooke40_inferior.json")

# Adam Optimizer
lens.optimize(iterations=200, target_efl=50.0)

# Levenberg-Marquardt Optimizer
lens.optimize_lm(iterations=35, target_efl=50.0)
```
