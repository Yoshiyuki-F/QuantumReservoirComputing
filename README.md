# Reservoir

JAX-based research codebase for reservoir computing experiments, with a focus on
classical reservoirs, LR-QRC/gate-based quantum reservoirs, FNN baselines,
passthrough baselines, and reservoir-to-FNN distillation.

The project is organized as a config-driven experiment pipeline. `run_pipeline()`
is the single entry point for the experiment flow, while factories build concrete
models and readouts from typed config objects.

## Architecture

This repository is primarily a **pipeline architecture**:

```text
Dataset
  -> PipelineDataManager
  -> PipelineModelBuilder
  -> PipelineExecutor
  -> ResultReporter
```

The main entry point is:

```text
src/reservoir/pipelines/run.py
```

`run_pipeline()` orchestrates:

1. data loading, preprocessing, and projection
2. model/readout construction from config
3. training, feature extraction, strategy selection, and readout fitting
4. metric collection, report formatting, and output writing

Inside the pipeline, the project also uses:

- **Factory pattern** for model/readout construction:
  - `src/reservoir/models/factory.py`
  - `src/reservoir/readout/factory.py`
- **Strategy pattern** for task-specific execution:
  - `src/reservoir/pipelines/strategies.py`
- **Typed config surface** for experiment definitions:
  - `src/reservoir/models/config.py`
  - `src/reservoir/models/presets.py`

In short:

```text
Pipeline = experiment flow
Factory  = concrete model/readout construction
Strategy = classification/regression/closed-loop execution behavior
Config   = experiment surface
```

## Supported experiment families

- Classical reservoir computing, used as LI-ESN style baseline
- LR-QRC / gate-based quantum reservoir computing
- FNN baseline
- Passthrough baseline
- FNN distillation from classical or quantum reservoir teachers
- Ridge, polynomial ridge, and FNN readouts

Active dataset presets:

- `mnist` for classification
- `lorenz` for chaotic time-series regression
- `mackey_glass` for chaotic time-series regression

## Numerical contract

This project treats float64 as part of the numerical contract. Silent float32
fallback is not allowed because it can change reservoir dynamics, chaos metrics,
and reproducibility.

JAX x64 is enforced through the gatekeeper in:

```text
src/reservoir/utils/gpu_utils.py
```

Do not remove the `jax.config.update("jax_enable_x64", True)` call there. It is
the effective enforcement point for the currently locked JAX version.

Array-domain boundaries are explicit:

- `NpF64` is the host/NumPy domain.
- `JaxF64` is the JAX/device domain.
- `src/reservoir/utils/batched_compute.py` owns batched NumPy/JAX conversion for
  memory-safe execution.

## Installation

This project uses `uv`.

## Environment

Current target environment:

- Python: `>=3.14.4`
- JAX backend: `jax[cuda13]`
- CUDA: CUDA 13 compatible NVIDIA environment

The default dependency set installs GPU-enabled JAX through `jax[cuda13]`.
CPU-only environments may require changing the JAX dependency in
`pyproject.toml`.

Runtime install:

```bash
uv sync
```

Full development/experiment install:

```bash
uv sync --all-extras --group dev
```

Runtime dependencies are intentionally kept narrow. Optional dependencies are
split into extras:

- `mnist`: `torch`, `torchvision`
- `viz`: `matplotlib`
- `hpo`: `optuna`
- `quantum-extra`: `qiskit`, `scipy`
- `analysis`: `pandas`
- `notebook`: `ipykernel`

## CLI usage

Use the module entry point:

```bash
uv run python -m reservoir.cli.main --model <MODEL> --dataset <DATASET>
```

Examples:

```bash
uv run python -m reservoir.cli.main --model classical_reservoir --dataset mnist
uv run python -m reservoir.cli.main --model quantum_reservoir --dataset lorenz
uv run python -m reservoir.cli.main --model quantum_reservoir --dataset mackey_glass
uv run python -m reservoir.cli.main --model fnn --dataset mnist
uv run python -m reservoir.cli.main --model passthrough --dataset mnist
uv run python -m reservoir.cli.main --model fnn_distillation_classical --dataset mnist
uv run python -m reservoir.cli.main --model fnn_distillation_quantum --dataset mnist
```

Common model values:

- `classical_reservoir`
- `quantum_reservoir`
- `fnn`
- `passthrough`
- `fnn_distillation_classical`
- `fnn_distillation_quantum`

Common dataset values:

- `mnist`
- `lorenz`
- `mackey_glass`

## Key files

```text
src/reservoir/
  cli/main.py                         CLI entry point
  pipelines/run.py                    unified pipeline entry point
  pipelines/components/               data/model/execution/reporting components
  pipelines/strategies.py             classification/regression strategies
  models/config.py                    typed experiment config objects
  models/presets.py                   task-aware preset registry
  models/factory.py                   model factory router
  readout/factory.py                  readout factory
  models/reservoir/classical/         classical reservoir implementation
  models/reservoir/quantum/model.py   QuantumReservoir model interface
  models/reservoir/quantum/functional.py
                                      JIT quantum circuit execution logic
  core/types.py                       NpF64/JaxF64 and shared typed payloads
  utils/batched_compute.py            batched JAX execution boundary
  utils/gpu_utils.py                  JAX x64/GPU gatekeeper
```

## Development checks

Run these before committing code changes:

```bash
uv run ruff check
uv run python scripts/lint_imports.py
uv run pyrefly check
```

Warning checks used during cleanup:

```bash
uv run --group dev python scripts/check_warnings.py
uv run python -W default -m reservoir.cli.main --model fnn --dataset mnist
```

`suppression_registry.md` tracks intentional `# noqa`, `# type: ignore[...]`,
global ignores, inspection suppressions, and allowed casts. Bare
`# type: ignore` is not allowed.

## References

- Jaeger, H. (2001). The echo state approach to analysing and training recurrent neural networks.
- JAX documentation: https://jax.readthedocs.io/
- TensorCircuit: https://github.com/tensorcircuit/tensorcircuit-ng/
