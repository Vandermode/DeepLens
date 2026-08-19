import os
import sys

# Ensure local test folder can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "test"))
from test_compare_lm_vs_adam import run_single_comparison

if __name__ == "__main__":
    gpu_id = 3
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Running Cellphone 80deg Benchmark on GPU {gpu_id}...")
    run_single_comparison(
        lens_path="./datasets/lenses/cellphone/cellphone80deg.json",
        benchmark_name="Cellphone80deg_GPU3",
        lm_iters=40,
        adam_iters=800,
        shape_control=True,
        result_base_dir="./results/benchmark_gpu3",
    )
