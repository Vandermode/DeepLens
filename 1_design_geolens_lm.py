"""GeoLens design example: optimize a cellphone camera lens using Levenberg-Marquardt (LM).

This script demonstrates second-order damped Gauss-Newton (LM) optimization
in DeepLens. It replaces first-order Adam with native Levenberg-Marquardt,
computing the forward-mode Jacobian of transverse ray aberrations to achieve
rapid convergence in fewer iterations.
"""

import os
import random
import string
import logging
from datetime import datetime
import torch

from deeplens import GeoLens
from deeplens.utils import set_logger, set_seed


def main() -> None:
    set_seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tag = "".join(random.choice(string.ascii_letters + string.digits) for _ in range(4))
    result_dir = f"./results/{datetime.now().strftime('%m%d-%H%M%S')}-lens-optim-lm-{tag}"
    os.makedirs(result_dir, exist_ok=True)
    set_logger(result_dir)
    logging.info(f"Device: {device}")

    # Load lens
    lens = GeoLens(filename="./datasets/lenses/cellphone/cellphone80deg.json")
    lens.analysis(save_name=f"{result_dir}/initial")
    logging.info(f"Loaded lens: FoV={lens.rfov:.4f} rad, F/{lens.fnum:.2f}")

    init_rms = lens.loss_rms(num_grid=16, num_rays=512).item()
    logging.info(f"Initial RMS Spot Error: {init_rms * 1000:.2f} um")

    # Run Levenberg-Marquardt Optimization with Paraxial ABCD Solve
    logging.info(f"Starting Levenberg-Marquardt optimization (target EFL = {lens.foclen:.2f} mm)...")
    lens.optimize_lm(
        iterations=50,
        test_per_iter=10,
        target_efl=lens.foclen,
        solve_surf_idx=-1,
        shape_control=True,
        optim_mat=False,
        lm_lambda=0.1,
        result_dir=result_dir,
    )

    # Final result
    lens.prune_surf()
    lens.post_computation()
    lens.write_lens_json(f"{result_dir}/final_lens.json")
    lens.analysis(save_name=f"{result_dir}/final_lens")

    final_rms = lens.loss_rms(num_grid=16, num_rays=512).item()
    logging.info(f"Optimization finished! Final RMS: {final_rms * 1000:.2f} um (Initial: {init_rms * 1000:.2f} um)")
    logging.info(f"Results saved in {result_dir}")


if __name__ == "__main__":
    main()
