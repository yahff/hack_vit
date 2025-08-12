#!/usr/bin/env python3
import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models.vision_transformer import vit_b_16, ViT_B_16_Weights
from pathlib import Path
from typing import Optional, Union, Tuple, Dict, Any
import json

from init import jl
from vit_hack import hack_vit

class VisionTransformerBenchmark:
    
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the benchmark class."""
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        print(f"Using device: {self.device}")
        
        # Image preprocessing pipeline for ViT
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load Julia implementations (all defined in julia_matmul.jl)
        print("Loading Julia implementations...")
        jl.seval('include("julia_matmul.jl")')
        
        # Available implementations (must match functions in julia_matmul.jl)
        self.available_implementations = ['builtin', 'naive', 'tiled', 'custom']
        self.current_implementation = None
        
        # Check what implementations are actually available in Julia
        self._verify_implementations()
        
    def _verify_implementations(self):
        available = []  # test available implementations

        for impl in ['builtin_matmul', 'naive_matmul', 'tiled_matmul', 'custom_matmul']:
            try:
                # Check if function exists in Julia
                exists = jl.seval(f"isdefined(Main, :{impl})")
                if exists:
                    available.append(impl.replace('_matmul', ''))
                else:
                    print(f"Warning: {impl} not found in Julia")
            except Exception as e:
                print(f"Warning: Could not check for {impl}: {e}")

        self.available_implementations = available
        print(f"Available implementations: {self.available_implementations}")

    def set_implementation(self, impl_name: str):
        
        if impl_name not in self.available_implementations:
            raise ValueError(f"Unknown implementation: {impl_name}. "
                           f"Available: {self.available_implementations}")
        
        print(f"Setting implementation to: {impl_name}")
        
        # Tell Julia to use the specified implementation
        # This creates an alias: optimized_matmul = specific_implementation_matmul
        jl.seval(f"optimized_matmul = {impl_name}_matmul")
        
        self.current_implementation = impl_name
        
    def get_implementation_info(self) -> Dict[str, Any]:
        """Get information about available implementations."""
        info = {
            'current': self.current_implementation,
            'available': self.available_implementations,
            'descriptions': {
                'builtin': 'Julia built-in BLAS operations (A * B)',
                'naive': 'Triple-loop implementation for reference',
                'block': 'Block-based multiplication for cache efficiency',
            '   custom': 'User-defined custom implementation (edit julia_matmul.jl)'
            }
        }
        return info
    
    def load_model(self, model_path: Optional[str] = None) -> torch.nn.Module:
        """Load a ViT model, either from a saved .pth file or pretrained weights."""
        print(model_path)

        if model_path and Path(model_path).exists():
            print(f"Loading model from: {model_path}")
            # Create model with correct number of classes for CIFAR-100
            model = vit_b_16(weights=None, num_classes=100)
            
            checkpoint = torch.load(model_path, map_location=self.device)
            try:
                model.load_state_dict(checkpoint['model_state_dict'])
            except RuntimeError as e:
                print(f"Strict load failed due to: {e}")
                print("Loading with strict=False and keeping classifier head from model definition.")
                state_dict = checkpoint['model_state_dict']
                # Remove head weights if they don't match
                state_dict.pop('heads.head.weight', None)
                state_dict.pop('heads.head.bias', None)
                model.load_state_dict(state_dict, strict=False)
        else:
            print("Loading pretrained ViT-B/16 model")
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            model = vit_b_16(weights=weights)

        model = model.to(self.device).eval()
        return model

    def load_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """Load and preprocess an image for ViT input."""
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        print(f"Loading image: {image_path}")
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor.to(self.device)
    
    def benchmark_model(self, 
                       model: torch.nn.Module, 
                       input_tensor: torch.Tensor, 
                       warmup_runs: int = 3, 
                       benchmark_runs: int = 10) -> Dict[str, Any]:
        """Benchmark a model's performance."""
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = model(input_tensor)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        with torch.no_grad():
            for _ in range(benchmark_runs):
                output = model(input_tensor)
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / benchmark_runs
        fps = benchmark_runs / total_time
        
        return {
            'total_time': total_time,
            'avg_time': avg_time,
            'fps': fps,
            'output': output,
            'implementation': self.current_implementation
        }
    
    def run_comparison(self, 
                    model_path: Optional[str] = None,
                    image_path: Optional[str] = None,
                    implementations: Optional[list] = None,
                    warmup_runs: int = 3,
                    benchmark_runs: int = 10,
                    save_results: bool = True) -> Dict[str, Any]:
        """Run a comprehensive comparison of different implementations."""

        if implementations is None:
            implementations = self.available_implementations

        # Validate requested implementations
        invalid_impls = [impl for impl in implementations if impl not in self.available_implementations]
        if invalid_impls:
            raise ValueError(f"Invalid implementations: {invalid_impls}. "
                            f"Available: {self.available_implementations}")

        print("="*60)
        print("VISION TRANSFORMER JULIA MATMUL BENCHMARK")
        print("="*60)

        # Load base model once
        base_model = self.load_model(model_path)

        # Prepare input
        if image_path:
            input_tensor = self.load_image(image_path)
            input_info = f"Image: {image_path}"
        else:
            input_tensor = torch.randn(1, 3, 224, 224).to(self.device)
            input_info = "Random tensor (1, 3, 224, 224)"

        print(f"Input: {input_info}")
        print(f"Model: {'Custom' if model_path else 'Pretrained ViT-B/16'}")
        print(f"Device: {self.device}")
        print(f"Implementations to test: {implementations}")
        print()

        results = {
            'config': {
                'model_path': model_path,
                'image_path': image_path,
                'input_info': input_info,
                'device': str(self.device),
                'implementations': implementations,
                'warmup_runs': warmup_runs,
                'benchmark_runs': benchmark_runs
            },
            'results': {},
            'comparisons': {}
        }

        baseline_output = None
        baseline_time = None

        # Test each implementation
        for impl_name in implementations:
            print(f"Testing {impl_name.upper()} implementation...")

            # Set Julia implementation
            self.set_implementation(impl_name)

            # Create fresh hacked model for this implementation
            test_model = hack_vit(self.load_model(model_path))
            test_model = test_model.to(self.device)  # Ensure model is on correct device

            # Benchmark
            bench_results = self.benchmark_model(
                test_model, input_tensor, warmup_runs, benchmark_runs
            )

            # Calculate comparisons
            if baseline_output is None:
                baseline_output = bench_results['output']
                baseline_time = bench_results['avg_time']
                output_diff = 0.0
                speedup = 1.0
            else:
                output_diff = torch.norm(bench_results['output'] - baseline_output).item()
                speedup = baseline_time / bench_results['avg_time']

            # Store results
            results['results'][impl_name] = {
                'total_time': bench_results['total_time'],
                'avg_time': bench_results['avg_time'],
                'fps': bench_results['fps'],
                'output_difference': output_diff,
                'speedup_vs_baseline': speedup,
                'implementation': impl_name
            }

            # Print results
            print(f"  Time: {bench_results['avg_time']:.6f}s per inference")
            print(f"  FPS: {bench_results['fps']:.2f}")
            print(f"  Output difference from baseline: {output_diff:.8f}")
            print(f"  Speedup vs baseline: {speedup:.3f}x")
            print()

        # Print summary
        self._print_summary(results)

        # Save results
        if save_results:
            self._save_results(results)

        return results
        
    def _print_summary(self, results: Dict[str, Any]):
        """Print a summary of benchmark results."""
        print("="*60)
        print("BENCHMARK SUMMARY")
        print("="*60)
        
        impl_results = results['results']
        sorted_impls = sorted(impl_results.keys(), key=lambda x: impl_results[x]['avg_time'])
        
        print("Performance Ranking (fastest to slowest):")
        for i, impl_name in enumerate(sorted_impls):
            result = impl_results[impl_name]
            print(f"{i+1}. {impl_name.upper()}: {result['avg_time']:.6f}s "
                  f"({result['fps']:.2f} FPS, {result['speedup_vs_baseline']:.3f}x speedup)")
        
        print()
        print("Output Accuracy:")
        for impl_name in sorted_impls:
            result = impl_results[impl_name]
            print(f"  {impl_name.upper()}: difference = {result['output_difference']:.8f}")
        print()
        
    def _save_results(self, results: Dict[str, Any], filename: Optional[str] = None):
        """Save benchmark results to JSON file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"vit_benchmark_results_{timestamp}.json"
        
        serializable_results = self._make_serializable(results)
        with open(filename, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results saved to: {filename}")
    
    def _make_serializable(self, obj):
        """Convert objects to JSON-serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

# Example usage
def example_usage():
    """Example of how to use the improved benchmark class."""
    
    benchmark = VisionTransformerBenchmark()
    
    # Show available implementations
    info = benchmark.get_implementation_info()
    print("Implementation Info:", info)
    
    # Quick comparison
    results = benchmark.run_comparison(
        implementations=['builtin', 'naive'] if 'naive' in info['available'] else ['builtin'],
        benchmark_runs=3
    )

if __name__ == "__main__":
    example_usage()