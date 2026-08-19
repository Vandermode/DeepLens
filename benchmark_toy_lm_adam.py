import os
import sys

# Ensure local test folder can be imported
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "test"))
from test_compare_lm_vs_adam import run_single_comparison

if __name__ == "__main__":
    gpu_id = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"Running Toy Doublet Benchmark on GPU {gpu_id}...")
    run_single_comparison(
        lens_path="./datasets/lenses/toy/toy_starting_point.json",
        benchmark_name="ToyDoublet_GPU1",
        lm_iters=30,
        adam_iters=500,
        shape_control=False,
        result_base_dir="./results/benchmark_gpu1",
    )
