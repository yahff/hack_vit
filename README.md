# Julia Matrix Multiplication in a Vision Transformer

This project benchmarks several matrix multiplication implementations inside the
self-attention layers of a PyTorch Vision Transformer (ViT-B/16). Attention
projections are routed through Julia so their performance and numerical output
can be compared from Python, or against each other.

Note that the overhead of calling Julia methods often results in worse performance compared to built-in Python implementations. The interface is best used to compare various Julia implementations against each other, rather than against Python implementations. 

## Implementations

The Julia code in `julia_matmul.jl` provides:

- `builtin`: Julia's optimized `A * B` implementation
- `naive`: a straightforward triple-loop reference implementation
- `tiled`: block-based multiplication intended to improve cache locality
- `custom`: an editable implementation for experiments

The Python bridge handles the conversion between PyTorch tensors and Julia
arrays. If a Julia multiplication fails, it falls back to `torch.matmul`.

## Requirements

- Python 3.9 or newer
- Julia 1.9 or newer
- PyTorch and torchvision
- `juliacall`, `numpy`, and `Pillow`
- A working Julia `PythonCall` environment

Install the Python dependencies with:

```bash
pip install torch torchvision juliacall numpy Pillow
```

The first JuliaCall run may install and configure Julia packages. `setup.py`
can also be run to install the Julia `CUDA` package when that is needed:

```bash
python setup.py
```

## Running the benchmark

From this directory, run the example comparison:

```bash
python VisionTransformerBenchmark.py
```

This uses a pretrained ViT-B/16 model, a random `1 x 3 x 224 x 224` input,
three warmup runs, and ten measured runs per available implementation. Results
are written to `vit_benchmark_results_<timestamp>.json`.

For a smaller direct example, run:

```bash
python runner.py
```

To benchmark an image, pass it through `VisionTransformerBenchmark.load_image`
or set the `image_path` argument in `run_comparison`. A saved model checkpoint
can be supplied with `model_path`; checkpoints are expected to contain a
`model_state_dict` entry.

## Project layout

- `VisionTransformerBenchmark.py`: model loading, benchmarking, comparison,
	and result serialization
- `vit_hack.py`: replaces each ViT encoder attention module with the Julia-backed
	implementation
- `julia_bridge.py`: PyTorch autograd bridge and Julia dispatch
- `julia_matmul.jl`: matrix multiplication implementations and batched attention
- `runner.py`: small manual comparison example
- `init.py`: JuliaCall initialization
