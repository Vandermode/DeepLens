"""Test reproducing the 2-element toy lens design from Generalized Aberrations using DeepLens LM Optimizer."""

import os
import pytest
import torch
import numpy as np

import deeplens
from deeplens import GeoLens
from deeplens.utils import set_seed


@pytest.mark.slow
def test_toy_lens_lm_reproduction():
    """Reproduce the toy doublet lens design using DeepLens with native Levenberg-Marquardt."""
    set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Test] Running Toy Lens LM Reproduction on {device}...")

    json_path = "./datasets/lenses/toy/toy_starting_point.json"
    assert os.path.exists(json_path), f"Toy lens starting config {json_path} not found."

    lens = GeoLens(filename=json_path)
    lens.calc_pupil()

    init_rms = lens.loss_rms(num_grid=16, num_rays=256).item()
    print(f"Initial RMS Spot Radius: {init_rms * 1000:.2f} um")
    assert init_rms > 0.1, "Initial lens should have significant initial aberration."

    # Optimize with native Levenberg-Marquardt optimizer
    result_dir = "./results/test_toy_lens_lm"
    os.makedirs(result_dir, exist_ok=True)

    lens.optimize_lm(
        iterations=30,
        test_per_iter=10,
        shape_control=False,
        optim_mat=False,
        lm_lambda=0.1,
        result_dir=result_dir,
    )

    # Post-optimization verification
    final_rms = lens.loss_rms(num_grid=16, num_rays=256).item()
    print(f"Final RMS Spot Radius: {final_rms * 1000:.2f} um (Initial: {init_rms * 1000:.2f} um)")

    reduction = (init_rms - final_rms) / init_rms
    print(f"Aberration Reduction: {reduction * 100:.2f}%")

    # Assertions
    assert final_rms < init_rms, "Final RMS must be lower than initial RMS."
    assert reduction > 0.70, f"Expected >70% aberration reduction with LM, got {reduction * 100:.2f}%."
    print("✓ Toy lens LM reproduction test passed successfully!")


if __name__ == "__main__":
    test_toy_lens_lm_reproduction()
