#  Quantum Reservoir Computing with JAX (LR-QRC)

A JAX-based Reservoir Computing implementation designed for high-performance numerical computation and flexibility. This project supports various reservoir architectures, including classical Reservoirs and Quantum Reservoirs (Gate-based), and is capable of handling both Regression and Classification tasks.

## Features

- **JAX-based**: Optimized for high-speed numerical computation and GPU acceleration.
- **Unified Pipeline**: Consistent interface for running experiments across different models and datasets.
- **Multiple Architectures**:
    - Classical Reservoir(LI-ESN)
    - Feedforward Neural Network (FNN)
    - Distillation Models (FNN from Reservoir)
    - PassThrough Models (no train. just read out)
    - Quantum Reservoirs (Gate-based)
- **Flexible Data Handling**: Built-in support for synthetic datasets (Lorenz, Mackey-Glass) and MNIST.
- **GPU Support**: Built-in GPU detection and optimization.

## Installation

This project uses `uv` for dependency management.

### Technical Notes: Precision & Determinism

This library enforces **JAX 64-bit precision** (`jax_enable_x64=True`) globally.
- **Why**: `RidgeRegression` (Readout) requires 64-bit precision. If not enforced globally, upstream components (like `RandomProjection`) may run in 32-bit initially, then switch to 64-bit in subsequent runs (after Ridge enables it), causing non-deterministic behavior.
- **Benefit**: Ensures that **all pipeline stages** (Projection, Reservoir, Readout) consistently use 64-bit precision from the start.
- **Entry Points**: x64 is automatically enabled in:
    - `python -m reservoir.cli.main` (CLI)
    - `reservoir.pipelines.run_pipeline` (Python API)
    - `optuna` benchmarks

```bash
# Clone the repository and navigate to the directory
cd /path/to/reservoir

# Install dependencies
uv sync --upgrade

# Activate virtual environment
source .venv/bin/activate
```

### Basic Command

```bash
uv run python -m reservoir.cli.main --model <MODEL> --dataset <DATASET> [OPTIONS]
```

### Arguments

- `--model`: **(Required)** The model architecture to use.
    - `classical_reservoir`
    - `fnn`
    - `fnn_distillation_quantum`
    - `fnn_distillation_classical`
    - `quantum_reservoir`
    - `passthrough`
- `--dataset`: **(Required)** The dataset to use.
    - `lorenz` (Regression)
    - `mackey_glass` (Regression)
    - `mnist` (Classification)
- `--force-cpu`: Force execution on CPU (not tried)

### Examples

**1. Classical Reservoir on Sine Wave**
```bash
uv run python -m reservoir.cli.main --model classical_reservoir --dataset mnist
```

**2. Quantum Gate-based Reservoir on Lorenz Attractor**
```bash
uv run python -m reservoir.cli.main --model quantum_reservoir --dataset lorenz
```

**3. MLP Training (FNN) on MNIST**
```bash
uv run python -m reservoir.cli.main --model fnn --dataset mnist
```

## File Structure

```
.
├── src/reservoir/
│   ├── cli/                # CLI entry point (main.py)
│   ├── core/               # Core definitions (identifiers, config)
│   ├── models/             # Model architectures and presets
│   ├── pipelines/          # Experiment pipelines
│   ├── layers/             # Neural network layers (Reservoir, Attention, etc.)
│   ├── data/               # Data generation and loading
│   └── utils/              # Utilities (GPU, metrics, logging)
├── tests/                  # Unit and integration tests
├── pyproject.toml          # Project configuration and dependencies
└── README.md
```

## References

- Jaeger, H. (2001). The "echo state" approach to analysing and training recurrent neural networks.
- [JAX Documentation](https://jax.readthedocs.io/)
- [TensorCircuit](https://github.com/tensorcircuit/tensorcircuit-ng/) (for Quantum Reservoir Computing)
