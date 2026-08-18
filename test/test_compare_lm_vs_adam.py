"""Comprehensive test suite comparing Levenberg-Marquardt (LM) vs Adam in DeepLens."""

import os
import time
import pytest
import torch
import numpy as np

import deeplens
from deeplens import GeoLens
from deeplens.utils import set_seed


def run_single_comparison(
    lens_path: str,
    benchmark_name: str,
    lm_iters: int = 30,
    adam_iters: int = 500,
    shape_control: bool = False,
    result_base_dir: str = "./results/comparison",
):
    """Run head-to-head comparison between LM and Adam on a given lens prescription."""
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*70}")
    print(f"BENCHMARK: {benchmark_name} ({lens_path}) on {device}")
    print(f"{'='*70}")

    # 1. Baseline Initial Evaluation
    lens_init = GeoLens(filename=lens_path)
    lens_init.calc_pupil()
    init_rms = lens_init.loss_rms(num_grid=16, num_rays=256).item()
    print(f"Initial Baseline RMS Spot Error: {init_rms * 1000:.2f} um")

    # 2. Optimize with Adam
    set_seed(0)
    lens_adam = GeoLens(filename=lens_path)
    adam_dir = f"{result_base_dir}/{benchmark_name}_adam"
    print(f"\n--- Running Adam Optimizer ({adam_iters} iterations) ---")
    t0 = time.time()
    lens_adam.optimize(
        lrs=[1e-3, 1e-4, 1e-1, 1e-4],
        iterations=adam_iters,
        test_per_iter=max(1, adam_iters // 5),
        shape_control=shape_control,
        optim_mat=False,
        result_dir=adam_dir,
    )
    adam_time = time.time() - t0
    adam_rms = lens_adam.loss_rms(num_grid=16, num_rays=256).item()
    print(f"--> Adam Finished in {adam_time:.1f}s | Final RMS: {adam_rms * 1000:.2f} um")

    # 3. Optimize with Levenberg-Marquardt (LM) with Paraxial ABCD solve
    set_seed(0)
    lens_lm = GeoLens(filename=lens_path)
    target_efl = float(lens_init.foclen) if hasattr(lens_init, "foclen") else None
    lm_dir = f"{result_base_dir}/{benchmark_name}_lm"
    print(f"\n--- Running Levenberg-Marquardt Optimizer ({lm_iters} iterations, target_efl={target_efl}) ---")
    t0 = time.time()
    lens_lm.optimize_lm(
        iterations=lm_iters,
        test_per_iter=max(1, lm_iters // 5),
        target_efl=target_efl,
        solve_surf_idx=-1,
        shape_control=shape_control,
        optim_mat=False,
        lm_lambda=0.1,
        result_dir=lm_dir,
    )
    lm_time = time.time() - t0
    lm_rms = lens_lm.loss_rms(num_grid=16, num_rays=256).item()
    print(f"--> LM Finished in {lm_time:.1f}s | Final RMS: {lm_rms * 1000:.2f} um")

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY FOR {benchmark_name}:")
    print(f"  Starting RMS: {init_rms * 1000:.2f} um")
    print(f"  Adam ({adam_iters} iters, {adam_time:.1f}s): {adam_rms * 1000:.2f} um ({((init_rms-adam_rms)/init_rms)*100:+.1f}%)")
    print(f"  LM   ({lm_iters} iters, {lm_time:.1f}s):   {lm_rms * 1000:.2f} um ({((init_rms-lm_rms)/init_rms)*100:+.1f}%)")
    print(f"{'='*70}\n")

    return {
        "init_rms": init_rms,
        "adam_rms": adam_rms,
        "adam_time": adam_time,
        "lm_rms": lm_rms,
        "lm_time": lm_time,
    }


@pytest.mark.slow
def test_toy_doublet_comparison():
    """Test LM vs Adam on the 2-element toy doublet from Generalized Aberrations."""
    res = run_single_comparison(
        lens_path="./datasets/lenses/toy/toy_starting_point.json",
        benchmark_name="ToyDoublet",
        lm_iters=25,
        adam_iters=400,
        shape_control=False,
    )
    assert res["lm_rms"] < res["init_rms"], "LM should reduce RMS error on toy doublet."


@pytest.mark.slow
def test_cooke_triplet_comparison():
    """Test LM vs Adam on the classical Cooke Triplet lens."""
    res = run_single_comparison(
        lens_path="./datasets/lenses/cooke40_inferior.json",
        benchmark_name="CookeTriplet",
        lm_iters=30,
        adam_iters=500,
        shape_control=False,
    )
    assert res["lm_rms"] < res["init_rms"], "LM should reduce RMS error on Cooke triplet."


if __name__ == "__main__":
    test_toy_doublet_comparison()
    test_cooke_triplet_comparison()
