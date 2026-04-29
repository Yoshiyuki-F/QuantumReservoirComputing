# Reservoir Package Architecture

This package implements a config-driven experiment pipeline for reservoir
computing research.

## High-level flow

```text
run_pipeline(config, dataset, training_config)
  -> PipelineDataManager.prepare()
  -> PipelineModelBuilder.build()
  -> PipelineExecutor.run()
  -> ResultReporter.compile_and_save()
```

The pipeline stages correspond to the experiment flow:

1. Input data
2. Preprocessing
3. Projection
4. Model construction
5. Model execution and feature extraction
6. Aggregation/readout fitting
7. Evaluation and reporting

## Main entry point

```text
pipelines/run.py
```

`run_pipeline()` is intentionally orchestration-only. It should not contain
low-level math, training loops, plotting logic, or reporting details. Those are
delegated to pipeline components.

## Components

```text
pipelines/components/data_manager.py
```

Owns data loading, splitting, preprocessing, projection setup, metadata, and
frontend context preparation.

```text
pipelines/components/model_builder.py
```

Builds the model/readout stack from config by calling the model and readout
factories.

```text
pipelines/components/executor.py
```

Runs training, feature extraction, strategy selection, and readout fitting.

```text
pipelines/components/reporter.py
```

Converts execution results into final metrics, logs, outputs, and report files.

## Factories

```text
models/factory.py
readout/factory.py
```

Factories are construction boundaries. They translate typed config objects into
concrete model and readout instances. They should not own the full experiment
flow.

## Config surface

```text
models/config.py
models/presets.py
```

`PipelineConfig` connects preprocessing, projection, model, and readout config.
`presets.py` maps dataset/task type to concrete experiment presets such as
classical reservoir, quantum reservoir, FNN, passthrough, and distillation.

## Strategy layer

```text
pipelines/strategies.py
```

Strategies own task-specific execution behavior, including classification,
open-loop regression, closed-loop generation, ridge search, and chaos metrics.

## Numerical and array boundaries

```text
core/types.py
utils/batched_compute.py
utils/gpu_utils.py
```

- `NpF64` is the host/NumPy domain.
- `JaxF64` is the JAX/device domain.
- `batched_compute()` owns batched host/device conversion for large feature
  computations.
- `gpu_utils.py` is the JAX x64 gatekeeper.

Internal code should keep array domains explicit and avoid ad-hoc NumPy/JAX
conversion.

## Quantum reservoir boundary

```text
models/reservoir/quantum/model.py
models/reservoir/quantum/functional.py
```

`model.py` exposes `QuantumReservoir` as a normal pipeline model. It adapts
pipeline inputs, feedback settings, measurement settings, and state handling into
the lower-level quantum execution functions.

`functional.py` contains the JIT quantum circuit execution logic, including
Z/ZZ measurements, feedback, reuploading, and noise handling.
