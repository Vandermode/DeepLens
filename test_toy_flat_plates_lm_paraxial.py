"""Test optimizing the toy 2-element doublet from completely FLAT glass plates using LM + Paraxial ABCD Solve."""

import os
import time
import torch
import numpy as np

import deeplens
from deeplens import GeoLens
from deeplens.utils import set_seed, set_logger


def main():
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result_dir = "./results/toy_flat_plates_lm_paraxial"
    os.makedirs(result_dir, exist_ok=True)
    set_logger(result_dir)

    print(f"\n{'='*75}")
    print(f"EXPERIMENT: From-Scratch Design from Flat Parallel Plates (c=0, a=0)")
    print(f"ALGORITHM:  Levenberg-Marquardt + Differentiable Paraxial ABCD Solve")
    print(f"DEVICE:     {device}")
    print(f"{'='*75}")

    # 1. Load completely flat parallel glass plates
    lens = GeoLens(filename="./datasets/lenses/toy/toy_flat_plates.json")
    print("\nInitial Geometry (Flat Parallel Glass Plates):")
    for i, s in enumerate(lens.surfaces):
        c_val = float(s.c) if hasattr(s, "c") else 0.0
        d_val = float(s.d) if hasattr(s, "d") else 0.0
        print(f"  Surface {i} ({s.__class__.__name__}): curvature c = {c_val:.4f}, z-pos d = {d_val:.4f} mm")

    # 2. Apply Analytic Paraxial Solve to enforce target EFL = 10.0 mm
    print("\nApplying Step 0 Paraxial ABCD Solve (target EFL = 10.0 mm, solve on surface 4)...")
    lens.solve_paraxial(target_efl=10.0, solve_surf_idx=4)
    print(f"  -> Solved surface 4 curvature: c4 = {lens.surfaces[4].c.item():.6f} (1/mm)")
    print(f"  -> Solved sensor distance:    d_sensor = {lens.d_sensor.item():.4f} mm")
    print(f"  -> System EFL:                {lens.efl:.4f} mm (F-number: F/{lens.fnum:.2f})")

    lens.calc_pupil()
    init_rms = lens.loss_rms(num_grid=16, num_rays=256).item()
    print(f"  -> Initial RMS Spot Radius:   {init_rms * 1000:.2f} um\n")

    # 3. Optimize directly with Levenberg-Marquardt with Paraxial Solve (NO curriculum learning)
    print("--- Running Levenberg-Marquardt Optimizer (35 steps with active ABCD solve) ---")
    t0 = time.time()
    lens.optimize_lm(
        iterations=35,
        test_per_iter=5,
        target_efl=10.0,
        solve_surf_idx=4,
        shape_control=False,
        optim_mat=False,
        lm_lambda=0.1,
        result_dir=result_dir,
    )
    lm_time = time.time() - t0

    # 4. Final Evaluation
    final_rms = lens.loss_rms(num_grid=16, num_rays=256).item()
    print(f"\n{'='*75}")
    print("OPTIMIZATION COMPLETED!")
    print(f"  Wall-Clock Time:        {lm_time:.1f} seconds")
    print(f"  Initial RMS Spot Error: {init_rms * 1000:.2f} um")
    print(f"  Final RMS Spot Error:   {final_rms * 1000:.2f} um")
    print(f"  Aberration Reduction:   {((init_rms - final_rms) / init_rms) * 100:.2f}%")
    print(f"  Final System EFL:       {lens.efl:.4f} mm (Target: 10.0 mm)")
    print(f"{'='*75}\n")


if __name__ == "__main__":
    main()
