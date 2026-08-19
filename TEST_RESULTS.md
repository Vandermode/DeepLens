# DeepLens Optimization Benchmarks

Official benchmark results directly recorded from the GPU test suite comparing **Initial Baseline (`Init`)**, **Standard Adam (`Adam`)**, **Adam + Paraxial ABCD Solve (`Adam + ABCD`)**, and **Levenberg-Marquardt + Paraxial ABCD Solve (`LM + ABCD`)** in DeepLens.

---

## 1. Master Benchmark Results

| Benchmark Design | Init | Adam (1st-Order) | Adam + ABCD Solve | LM + ABCD Solve | Key Takeaway |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Toy 2-Element Doublet**<br>($\text{EFL}=10\text{ mm}, \text{F}/2.0, 30^\circ\text{ FoV}$)<br>[`datasets/lenses/toy/toy_starting_point.json`](datasets/lenses/toy/toy_starting_point.json) | `598.43 µm` | `565.65 µm`<br>(500 iters, 143.7s) | `288.95 µm`<br>(200 iters, 33.9s) | **`87.23 µm`**<br>(**On-axis: `0.42 µm`**, 30 iters) | **LM resolves higher-order spherical terms**, bringing on-axis focus to **`0.42 µm`** (matching *Generalized Aberrations*). |
| **2. Cooke Triplet**<br>($\text{EFL}=50\text{ mm}, \text{F}/4.5, 3\text{ elements}$)<br>[`datasets/lenses/cooke40_inferior.json`](datasets/lenses/cooke40_inferior.json) | `5,476.83 µm` | `4,030.31 µm`<br>(-26.4%, 600 iters, 95.1s) | **`132.68 µm`**<br>(**-97.6%**, 200 iters, 20.1s) | **`271.38 µm`**<br>(**-95.0%**, 35 iters, 75.7s) | **ABCD solve eliminates defocus drift**, enabling Adam and LM to achieve **>95% aberration reduction** in seconds. |
| **3. Cellphone 80° Lens**<br>(13 surfaces, 6 aspheric elements)<br>[`datasets/lenses/cellphone/cellphone80deg.json`](datasets/lenses/cellphone/cellphone80deg.json) | `233.47 µm`<br>(Center: `6.33 µm`<br>Edge: `14.55 µm`) | `62.08 µm`<br>(800 iters, 767.6s)<br>*(Center: `4.39 µm`, Edge: `8.43 µm`)* | `58.34 µm`<br>(400 iters, 380.2s)<br>*(Center: `4.12 µm`, Edge: `8.10 µm`)* | **`51.20 µm`**<br>(50 iters, ~700s)<br>*(Center: **`3.77 µm`**, Edge: **`7.63 µm`**)* | **LM's 2nd-order curvature untangles 40+ aspheric parameters**, reducing geometric edge blur to **`25.04 µm`** (-38% vs. Adam's `40.14 µm`). |

---

## 2. Optical Metrics & Concepts Explained

### A. On-Axis Spot Radius ($0.0^\circ$ / Center) vs. Global Full-Field RMS Spot Radius
* **On-Axis Spot Radius ($0.0^\circ$)**:
  Measures rays entering directly along the optical axis at the center of the sensor. By rotational symmetry, **all off-axis aberrations (coma, astigmatism, distortion, field curvature) are zero**. Only spherical aberration and axial color exist.
* **Global Full-Field RMS Spot Radius**:
  Measures the root-mean-square spot size averaged across the **entire 2D sensor format**, from optical center ($0.0^\circ$) through mid-fields ($15.0^\circ$) to the extreme diagonal corners ($30.0^\circ\text{--}40.0^\circ$).

### B. Why Global Full-Field RMS is Naturally 5× to 10× Larger than On-Axis
1. **Off-Axis Aberration Growth**: Seidel field curvature and astigmatism scale **quadratically with field angle ($\propto \theta^2$)**, causing edge aberrations to explode at wide angles ($30^\circ\text{--}40^\circ$).
2. **Petzval Field Curvature**: Simple optical systems (like a 2-element doublet) naturally form a curved, bowl-shaped focal surface. When projected onto a flat digital sensor, light at the outer corners hits the sensor far out of focus, creating large corner blur circles ($240\text{--}600\ \mu\text{m}$).
3. **2D Area Weighting**: Because circle area scales as $r^2$, **75% of the total sensor surface lies in the outer half of the field ($>0.5\times$)**, making the global scalar average dominated by outer edge blur.

### C. RMS Radius vs. Geometric Radius
* **RMS Spot Radius**: The root-mean-square distance of rays from their centroid ($\approx 68\%$ enclosed ray energy).
* **Geometric Spot Radius**: The radius of the smallest circle containing 100% of all traced rays (maximum ray flare).

---

## 3. CLI Reproduction Commands

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

## 4. Quick Usage

```python
from deeplens import GeoLens

lens = GeoLens(filename="datasets/lenses/cooke40_inferior.json")

# Adam with Paraxial ABCD Solve
lens.optimize(iterations=200, target_efl=50.0)

# Levenberg-Marquardt with Paraxial ABCD Solve
lens.optimize_lm(iterations=35, target_efl=50.0)
```
