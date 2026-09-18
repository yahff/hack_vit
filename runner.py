#!/usr/bin/env python3
import torch

from VisionTransformerBenchmark import VisionTransformerBenchmark
from vit_hack import hack_vit

def main():
    benchmark = VisionTransformerBenchmark()
    input_tensor = torch.randn(1, 3, 224, 224, device=benchmark.device)

    benchmark.set_implementation('builtin')
    builtin_model = hack_vit(benchmark.load_model())
    builtin_results = benchmark.benchmark_model(builtin_model, input_tensor, warmup_runs=1, benchmark_runs=3)

    benchmark.set_implementation('naive')
    naive_model = hack_vit(benchmark.load_model())
    naive_results = benchmark.benchmark_model(naive_model, input_tensor, warmup_runs=1, benchmark_runs=3)

    benchmark.set_implementation('tiled')
    tiled_model = hack_vit(benchmark.load_model())
    tiled_results = benchmark.benchmark_model(tiled_model, input_tensor, warmup_runs=1, benchmark_runs=3)

    benchmark.set_implementation('custom')
    custom_model = hack_vit(benchmark.load_model())
    custom_results = benchmark.benchmark_model(custom_model, input_tensor, warmup_runs=1, benchmark_runs=3)

    output_diff = torch.norm(builtin_results['output'] - custom_results['output']).item()
    
    print(f"Built-in implementation: {builtin_results['avg_time']:.6f}s")
    print(f"Naive implementation: {naive_results['avg_time']:.6f}s") 
    print(f"Tiled implementation: {tiled_results['avg_time']:.6f}s")
    print(f"custom implementation: {custom_results['avg_time']:.6f}s") 
    print(f"Output difference (builtin vs custom): {output_diff:.8f}")

if __name__ == "__main__":
    main()
