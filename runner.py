#!/usr/bin/env python3
"""
Example usage of the VisionTransformerBenchmark class
"""
import torch
from VisionTransformerBenchmark import VisionTransformerBenchmark
from vit_hack import hack_vit

def main():
    
    
    # Load model and prepare input
    model = benchmark.load_model(model_path='vit_cifar100_finetuned2.pth') #choose model.  default is inbuilt weights
    model = model.to(benchmark.device)
    input_tensor = benchmark.load_image("dogimage.jpeg") if False else \
                   torch.randn(1, 3, 224, 224).to(benchmark.device)
    
    # Test built-in implementation
    benchmark.set_implementation('builtin')
    builtin_model = hack_vit(benchmark.load_model())
    builtin_model = builtin_model.to(benchmark.device)  # Ensure model is on correct device
    builtin_results = benchmark.benchmark_model(builtin_model, input_tensor, warmup_runs=1, benchmark_runs=3)

    # Test naive implementation  
    benchmark.set_implementation('naive')
    naive_model = hack_vit(benchmark.load_model())
    naive_model = naive_model.to(benchmark.device)  # Ensure model is on correct device
    naive_results = benchmark.benchmark_model(naive_model, input_tensor, warmup_runs=1, benchmark_runs=3)
    
    #tiled
    benchmark.set_implementation('tiled')
    tiled_model = hack_vit(benchmark.load_model())
    tiled_model = tiled_model.to(benchmark.device)  # Ensure model is on correct device
    tiled_results = benchmark.benchmark_model(tiled_model, input_tensor, warmup_runs=1, benchmark_runs=3)
    
    #custom
    benchmark.set_implementation('custom')
    custom_model = hack_vit(benchmark.load_model())
    custom_model = custom_model.to(benchmark.device)  # Ensure model is on correct device
    custom_results = benchmark.benchmark_model(custom_model, input_tensor, warmup_runs=1, benchmark_runs=3)
    
    # Compare outputs
    output_diff = torch.norm(builtin_results['output'] - custom_results['output']).item()
    
    print(f"Built-in implementation: {builtin_results['avg_time']:.6f}s")
    print(f"Naive implementation: {naive_results['avg_time']:.6f}s") 
    print(f"tiled implementation: {tiled_results['avg_time']:.6f}s") 
    print(f"custom implementation: {custom_results['avg_time']:.6f}s") 
    print(f"Output difference (builtin vs custom): {output_diff:.8f}")

if __name__ == "__main__":
    import torch
    main()
