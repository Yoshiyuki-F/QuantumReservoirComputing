# Suppression Registry

This file tracks intentional linter and type-checker suppressions.

## Common Policy

- Every suppression must be justified with a stable scope, risk, owner, and removal condition.
- Broad suppressions are forbidden when a narrower code exists.
- Prefer structural fixes over suppressions. Registry entries are not approval to add similar suppressions elsewhere.
- Suppressions should be removed in favor of structural fixes whenever practical.
- CI should reject unregistered suppressions and bare `# type: ignore`.
- `Line(s)` are informational; `Path` and `Symbol / Scope` are the stable identifiers. CI should not require exact line-number matches.
- Any `Medium` or `High` risk entry must have a concrete removal condition.
- Runtime warning suppressions must be registered here when they are implemented via `warnings.filterwarnings`, pytest `filterwarnings`, environment variables, or equivalent mechanisms.
- Warnings related to dtype truncation, JAX x64 initialization, device/backend fallback, NaN/Inf, overflow, invalid arithmetic, or ignored/deprecated configuration must be fixed rather than suppressed.
- `typing.cast(...)` is forbidden in internal logic. Allowed casts must be isolated at unsafe boundaries, documented with a runtime invariant, and registered here.

## CI Enforcement

- Reject bare `# type: ignore` using a grep/script check.
- Reject `# noqa` and `# type: ignore[...]` entries that are not registered here.
- Reject unregistered `typing.cast(...)` uses outside explicitly allowed boundary scopes.
- Registry matching should use `Path` + suppression code + nearby `Symbol / Scope`; exact line numbers are informational only.
- Reject registry entries whose `Path` no longer contains the registered suppression code.

## Python Runtime Warning Policy

Python warnings are not required to be globally zero in this ML/quantum repository. They must still be classified before being ignored.

Warnings that must be fixed, not suppressed:
- JAX float64 requests being truncated to float32.
- dtype, device, backend fallback, or x64 initialization warnings.
- `RuntimeWarning` for overflow, invalid values, divide-by-zero, NaN, or Inf.
- Project-owned `DeprecationWarning`.
- Warnings that indicate ignored, unknown, or deprecated tool configuration.

Allowed temporary warning suppressions are limited to known third-party noise that does not affect correctness, precision, dtype, device selection, or reproducibility. Every implemented warning suppression must have a registry entry with a stable scope, reason, risk, owner, and removal condition.

Runtime warning visibility commands:

```bash
uv run --group dev python scripts/check_warnings.py
uv run python -W default -m reservoir.cli.main --model fnn --dataset mnist
```

Use these commands to surface import-time warnings across `src/reservoir` modules and runtime warnings during the representative MNIST/FNN path. Python warnings are runtime-only, so no command can fully cover unexecuted project paths. These commands are warning visibility checks, not replacements for `uv run ruff check`, `uv run python scripts/lint_imports.py`, or `uv run pyrefly check`.

`scripts/check_warnings.py` pins `JAX_PLATFORMS=cpu` before importing project modules so the import-time scan does not initialize the CUDA backend. GPU / CUDA / JAX x64 runtime behavior is verified separately through the CLI command above.

## Python Warning Suppression Registry

| Path | Symbol / Scope | Line(s) | Warning | Reason | Risk | Owner | Removal condition |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| _None currently registered_ | - | - | - | No Python runtime warning suppressions are currently allowed. | - | - | - |

## IDE Inspection Suppression Registry

| Path | Symbol / Scope | Line(s) | Inspection | Reason | Risk | Owner | Removal condition |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `src/reservoir/models/reservoir/quantum/backend.py` | TensorCircuit import boundary | 8-9 | `PyPackageRequirements` | Runtime module is provided by the `tensorcircuit-ng[jax]` dependency, whose import package is `tensorcircuit`; adding a separate `tensorcircuit` dependency would risk package collision. | Low | quantum-reservoir | Remove if IDE/package metadata correctly maps `tensorcircuit-ng` to the `tensorcircuit` import or if the dependency is renamed upstream. |
| `src/reservoir/models/reservoir/quantum/functional.py` | TensorCircuit import boundary | 14-15 | `PyPackageRequirements` | Runtime module is provided by the `tensorcircuit-ng[jax]` dependency, whose import package is `tensorcircuit`; adding a separate `tensorcircuit` dependency would risk package collision. | Low | quantum-reservoir | Remove if IDE/package metadata correctly maps `tensorcircuit-ng` to the `tensorcircuit` import or if the dependency is renamed upstream. |

## Typing Cast Registry

Policy additions:
- Prefer precise return types, typed locals, TypedDict/Required fields, Protocols, parser functions, and runtime guards over `typing.cast`.
- Casts are allowed only for third-party typing bugs, dynamic framework APIs, or validated deserialization/config boundaries.
- Repeated third-party casts must be moved behind an adapter module.

| Path | Symbol / Scope | Line(s) | Cast | Reason | Risk | Owner | Removal condition |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `src/reservoir/data/loaders.py` | loader registry decorator | 54 | `cast("LoaderFunc", fn)` | Registration decorator narrows a runtime callable into the loader registry contract after the decorator receives the function object. | Medium | data | Replace with a generic typed registration helper that preserves the callable signature without cast. |
| `src/reservoir/data/loaders.py` | chaos loader config branch | 278 | `cast("ChaosDatasetConfig", config)` | Config kind check selects the chaos config branch, but the checker cannot carry the discriminant through this legacy config object. | Medium | data | Replace with a typed config parser/discriminated config model. |
| `src/reservoir/models/generative.py` | `generate_closed_loop` JAX scan output | 171, 180 | `cast("tuple[JaxF64, JaxF64]", scan_out)`, `cast("JaxF64", scan_out)` | `jax.lax.scan` output type depends on `return_history`; runtime branch is clear but current annotation cannot express branch-specific output. | Medium | models | Split history and non-history generation helpers or add overloads that remove output casts. |
| `src/reservoir/models/nn/base.py` | Flax `TrainState` / parameter tree boundary | 246, 293-295 | `cast(...)` | Flax parameter tree and `apply` stubs are less precise than the runtime `FlaxParamTree` contract used by this model wrapper. | Medium | nn | Introduce a narrow Flax adapter/helper around TrainState/apply that returns project domain types. |
| `src/reservoir/models/reservoir/quantum/functional.py` | TensorCircuit/JAX state boundary | 162, 231, 233 | `cast("jax.Array", current_key)`, `cast("JaxF64", final_state)` | TensorCircuit/JAX quantum state APIs expose dynamic state representations that the checker cannot narrow to the active backend shape. | High | quantum-reservoir | Move TensorCircuit state/key narrowing into a TensorCircuit adapter/wrapper layer. |
| `src/reservoir/pipelines/components/executor.py` | fit result and closed-loop state boundary | 105, 165 | `cast("JaxF64", ...)`, `cast("ClosedLoopGenerativeModel[ModelState]", model)` | Pipeline result dictionaries and closed-loop model state variants are validated by upstream pipeline construction but not expressed precisely enough for the checker. | Medium | pipeline | Replace with typed result accessors and a closed-loop model protocol that exposes stateful generation. |
| `src/reservoir/pipelines/components/reporter.py` | result array / metrics extraction boundary | 112, 113, 140 | `cast("JaxF64", val)`, `cast("NpF64", val)`, `cast("TestMetrics", test_metrics_raw)` | Reporting consumes mixed result-domain containers after upstream validation; casts bridge current loose result dictionary typing. | Medium | reporting | Replace with typed result parser/accessor functions returning concrete reporting payload types. |
| `src/reservoir/pipelines/strategies.py` | FNN and closed-loop generation branch | 574, 877 | `cast("JaxF64", ...)`, `cast("ClosedLoopGenerativeModel[ModelState]", model)` | `generate_closed_loop` currently has branch-dependent return/state typing that is wider than each strategy branch. | Medium | pipeline | Split branch-specific generation helpers or add overloads/protocols that make casts unnecessary. |
| `tests/unit/test_poly_ridge.py` | invalid literal negative test | 192 | `cast("Literal['full', 'square_only', 'interaction_only']", "invalid")` | Test intentionally constructs a statically-invalid literal to verify runtime validation rejects it. | Low | tests | Replace with a public boundary parser test that accepts plain `str` input. |

## Ruff `# noqa` Registry

| Path | Symbol / Scope | Line(s) | Code | Reason | Risk | Owner | Removal condition |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `tests/unit/test_quantum_functional.py` | module import bootstrap | 6-8 | E402 | JAX x64 must be configured before importing JAX/TensorCircuit/project modules that may initialize JAX state; this file intentionally keeps setup before regular imports. | Low | quantum-reservoir | Remove if tests are split into a bootstrap entrypoint that initializes x64 before importing test implementation. |
| `benchmarks/optimize_qrc.py` | module import bootstrap | 24, 25, 27-30, 34, 39 | E402 | JAX x64 must be configured before importing JAX/TensorCircuit/project modules that may initialize JAX state; this file intentionally keeps setup before regular imports. | Medium | benchmarks | Remove after splitting benchmark entrypoints into x64 bootstrap modules and import-only implementation modules. |
| `benchmarks/optimize_qrc_mnist.py` | module import bootstrap | 32, 33, 35-38, 42, 48 | E402 | JAX x64 must be configured before importing JAX/TensorCircuit/project modules that may initialize JAX state; this file intentionally keeps setup before regular imports. | Medium | benchmarks | Remove after splitting benchmark entrypoints into x64 bootstrap modules and import-only implementation modules. |
| `benchmarks/optimize_qrc_multi_seed.py` | module import bootstrap | 23, 24, 26-29, 33, 38 | E402 | JAX x64 must be configured before importing JAX/TensorCircuit/project modules that may initialize JAX state; this file intentionally keeps setup before regular imports. | Medium | benchmarks | Remove after splitting benchmark entrypoints into x64 bootstrap modules and import-only implementation modules. |
| `benchmarks/optimize_rc.py` | module import bootstrap | 35-38, 42, 49 | E402 | JAX x64 must be configured before importing JAX/TensorCircuit/project modules that may initialize JAX state; this file intentionally keeps setup before regular imports. | Medium | benchmarks | Remove after splitting benchmark entrypoints into x64 bootstrap modules and import-only implementation modules. |
| `benchmarks/optimize_rc_mnist.py` | module import bootstrap | 33-35, 39, 46 | E402 | JAX x64 must be configured before importing JAX/TensorCircuit/project modules that may initialize JAX state; this file intentionally keeps setup before regular imports. | Medium | benchmarks | Remove after splitting benchmark entrypoints into x64 bootstrap modules and import-only implementation modules. |
| `benchmarks/optimize_rc_multi_seed.py` | module import bootstrap | 38-41, 45, 52 | E402 | JAX x64 must be configured before importing JAX/TensorCircuit/project modules that may initialize JAX state; this file intentionally keeps setup before regular imports. | Medium | benchmarks | Remove after splitting benchmark entrypoints into x64 bootstrap modules and import-only implementation modules. |
| `src/reservoir/data/__init__.py` | package public API / loader registration | 16, 17 | F401 | Imports intentionally expose package API and trigger loader registration side effects. | Low | data | Add explicit `__all__` and remove local `noqa` if Ruff no longer reports F401. |
| `src/reservoir/data/loaders.py` | loader registry callable aliases | 6 | TC003 | `Callable` is required at runtime because loader type aliases and the registration decorator are evaluated at module import time. | Low | data | Remove only after verifying these aliases can move behind `TYPE_CHECKING` without breaking loader registration. |
| `src/reservoir/layers/__init__.py` | package public API | 3-6 | F401 | Imports intentionally expose layer constructors from the package root. | Low | layers | Add explicit `__all__` and remove local `noqa` if Ruff no longer reports F401. |
| `src/reservoir/utils/__init__.py` | package public API | 12, 13, 18 | F401 | Imports intentionally expose utility APIs from the package root. | Low | utils | Add explicit `__all__` and remove local `noqa` if Ruff no longer reports F401. |
| `src/reservoir/layers/preprocessing.py` | `Preprocessor` and concrete `@beartype` classes | 13 | TC001 | `NpF64` must be importable at runtime because `@beartype` resolves this annotation during runtime validation; moving it under `TYPE_CHECKING` may break validation. | Low | layers | Remove only after verifying `@beartype` resolves postponed annotations safely when this import is moved under `TYPE_CHECKING`. |
| `src/reservoir/layers/projection.py` | `Projection` and concrete `@beartype` classes | 13 | TC001 | `JaxF64` must be importable at runtime because `@beartype` resolves this annotation during runtime validation; moving it under `TYPE_CHECKING` may break validation. | Low | layers | Remove only after verifying `@beartype` resolves postponed annotations safely when this import is moved under `TYPE_CHECKING`. |
| `src/reservoir/models/generative.py` | `ClosedLoopGenerativeModel.generate_closed_loop` callable contract | 6 | TC003 | `Callable` must be importable at runtime because `@beartype` resolves callable annotations during runtime validation. | Low | models | Remove only after verifying `@beartype` resolves postponed callable annotations safely when this import is moved under `TYPE_CHECKING`. |
| `src/reservoir/models/generative.py` | `ClosedLoopGenerativeModel` training contract | 13 | TC001 | `Projection` must be importable at runtime because `@beartype` resolves this annotation during runtime validation; moving it under `TYPE_CHECKING` may break validation. | Low | models | Remove only after verifying `@beartype` resolves postponed annotations safely when this import is moved under `TYPE_CHECKING`. |
| `src/reservoir/models/distillation/model.py` | `DistillationModel.__init__` / training methods | 8, 13, 14, 16 | TC001 | These symbols must be importable at runtime because `@beartype` resolves these annotations during runtime validation; moving them under `TYPE_CHECKING` may break validation. | Low | distillation | Remove only after verifying `@beartype` resolves postponed annotations safely when these imports are moved under `TYPE_CHECKING`. |
| `src/reservoir/models/nn/base.py` | `BaseFlaxModel` Flax module contract | 11 | TC002 | `flax.linen` must be importable at runtime because `@beartype` resolves `nn.Module` annotations and subclasses construct Flax modules from this symbol. | Low | nn | Remove only after verifying `@beartype` resolves postponed `nn.Module` annotations safely when this import is moved under `TYPE_CHECKING`. |
| `src/reservoir/models/nn/base.py` | `BaseModel` / `BaseFlaxModel` public contracts | 15-17 | TC001 | These symbols must be importable at runtime because `@beartype` resolves these annotations during runtime validation; moving them under `TYPE_CHECKING` may break validation. | Low | nn | Remove only after verifying `@beartype` resolves postponed annotations safely when these imports are moved under `TYPE_CHECKING`. |
| `src/reservoir/models/nn/fnn.py` | `FNNModel` sequence/callable contracts | 8 | TC003 | `Sequence` and `Callable` must be importable at runtime because `@beartype` resolves these annotations during runtime validation. | Low | nn | Remove only after verifying `@beartype` resolves postponed collection annotations safely when these imports are moved under `TYPE_CHECKING`. |
| `src/reservoir/models/nn/fnn.py` | `FNNModel` training and prediction contracts | 9-11, 19 | TC001 | These symbols must be importable at runtime because `@beartype` resolves these annotations during runtime validation; moving them under `TYPE_CHECKING` may break validation. | Low | nn | Remove only after verifying `@beartype` resolves postponed annotations safely when these imports are moved under `TYPE_CHECKING`. |
| `src/reservoir/models/passthrough/passthrough.py` | `PassthroughModel` public methods | 17, 18 | TC001 | These symbols must be importable at runtime because `@beartype` resolves these annotations during runtime validation; moving them under `TYPE_CHECKING` may break validation. | Low | models | Remove only after verifying `@beartype` resolves postponed annotations safely when these imports are moved under `TYPE_CHECKING`. |
| `src/reservoir/models/reservoir/classical/classical.py` | `ClassicalReservoir` public methods | 15, 16 | TC001 | These symbols must be importable at runtime because `@beartype` resolves these annotations during runtime validation; moving them under `TYPE_CHECKING` may break validation. | Low | reservoir | Remove only after verifying `@beartype` resolves postponed annotations safely when these imports are moved under `TYPE_CHECKING`. |
| `src/reservoir/pipelines/components/data_coordinator.py` | `DataLoader` and `DataCoordinator` public contracts | 15, 16 | TC001 | These symbols must be importable at runtime because `@beartype` resolves constructor and method annotations during runtime validation; moving them under `TYPE_CHECKING` may break validation. | Low | pipeline | Remove only after verifying `@beartype` resolves postponed annotations safely when these imports are moved under `TYPE_CHECKING`. |
| `src/reservoir/readout/fnn_readout.py` | `FNNReadout` public methods | 10, 11 | TC001 | These symbols must be importable at runtime because `@beartype` resolves these annotations during runtime validation; moving them under `TYPE_CHECKING` may break validation. | Low | readout | Remove only after verifying `@beartype` resolves postponed annotations safely when these imports are moved under `TYPE_CHECKING`. |

## Ruff Global Ignore Registry

| Path | Symbol / Scope | Line(s) | Setting / Code | Reason | Risk | Owner | Removal condition |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `pyproject.toml` | `[tool.ruff.lint]` global exception list | 120 | `extend-ignore` | Central registry for intentionally global Ruff exceptions. | Medium | tooling | Revisit when interface-heavy modules are narrowed enough to use local `noqa` instead. |
| `pyproject.toml` | exception style | 121 | `TRY003` | Allows detailed messages on standard exceptions; avoids unnecessary custom exception classes in research pipeline code. | Medium | tooling | Remove if the project introduces a formal exception hierarchy. |
| `pyproject.toml` | function interface compatibility | 122 | `ARG001` | Loader/factory/strategy functions intentionally keep compatible signatures across implementations. New ignores should still be avoided in application logic. | Medium | tooling | Prefer local suppressions once public interfaces stabilize. |
| `pyproject.toml` | method interface compatibility | 123 | `ARG002` | Subclasses keep parent/library-compatible method signatures even when a parameter is unused. New ignores should still be avoided in application logic. | Medium | tooling | Prefer local suppressions once model/readout interfaces stabilize. |
| `pyproject.toml` | staticmethod interface compatibility | 124 | `ARG004` | Static methods keep shared training/evaluation signatures. New ignores should still be avoided in application logic. | Medium | tooling | Prefer local suppressions once shared interfaces stabilize. |
| `pyproject.toml` | pytest assertions | 129 | `S101` for `tests/**/*.py` | Tests intentionally use `assert`, matching pytest idioms. | Low | tooling | Keep while pytest is the test runner. |

## Pyrefly `# type: ignore` Registry

Policy additions:
- Third-party typing mismatches must be isolated in adapter modules when repeated more than once.
- Internal dynamic-config suppressions are temporary and should be replaced by typed parser functions.
- Runtime invariants must be enforced by guards before using `type: ignore`.
- Ignore codes must match the current Pyrefly diagnostic code; run `uv run pyrefly check` after changing suppressions.

| Path | Symbol / Scope | Line(s) | Ignore | Reason | Risk | Owner | Removal condition |
| --- | --- | ---: | --- | --- | --- | --- | --- |
| `src/reservoir/models/reservoir/quantum/functional.py` | `_make_circuit_logic`: initial fused unitary application | 133 | `type: ignore[bad-argument-type]` | TensorCircuit stubs do not express JAX array unitaries with two-qubit overloads. | High | quantum-reservoir | Replace with a TensorCircuit adapter/wrapper layer that contains this unsafe boundary. |
| `src/reservoir/models/reservoir/quantum/functional.py` | `_make_circuit_logic.apply_noise`: MC Pauli unitary | 188 | `type: ignore[bad-argument-type]` | TensorCircuit stubs do not express JAX-selected matrix unitaries. | High | quantum-reservoir | Replace with a TensorCircuit adapter/wrapper layer that contains this unsafe boundary. |
| `src/reservoir/models/reservoir/quantum/functional.py` | `_make_circuit_logic.apply_noise`: depolarizing channel | 191 | `type: ignore[bad-argument-type]` | TensorCircuit noisy circuit stubs do not match installed backend call signature. | High | quantum-reservoir | Replace with a TensorCircuit adapter/wrapper layer or upstream-compatible stubs. |
| `src/reservoir/models/reservoir/quantum/functional.py` | `_make_circuit_logic.apply_noise`: amplitude damping channel | 195 | `type: ignore[bad-argument-type]` | TensorCircuit noisy circuit stubs do not match installed backend call signature. | High | quantum-reservoir | Replace with a TensorCircuit adapter/wrapper layer or upstream-compatible stubs. |
| `src/reservoir/models/reservoir/quantum/functional.py` | `_make_circuit_logic`: reupload unitary | 199 | `type: ignore[bad-argument-type]` | TensorCircuit stubs do not express JAX array unitaries with two-qubit overloads. | High | quantum-reservoir | Replace with a TensorCircuit adapter/wrapper layer that contains this unsafe boundary. |
| `src/reservoir/models/reservoir/quantum/functional.py` | `_make_circuit_logic`: brickwork CNOT even pass | 205 | `type: ignore[bad-argument-type]` | TensorCircuit stubs do not express runtime backend CNOT overloads. | High | quantum-reservoir | Replace with a TensorCircuit adapter/wrapper layer or upstream-compatible stubs. |
| `src/reservoir/models/reservoir/quantum/functional.py` | `_make_circuit_logic`: brickwork CNOT odd pass | 208 | `type: ignore[bad-argument-type]` | TensorCircuit stubs do not express runtime backend CNOT overloads. | High | quantum-reservoir | Replace with a TensorCircuit adapter/wrapper layer or upstream-compatible stubs. |
| `src/reservoir/models/reservoir/quantum/functional.py` | `_make_circuit_logic`: layer rotation unitary | 211 | `type: ignore[bad-argument-type]` | TensorCircuit stubs do not express JAX array unitary overloads. | High | quantum-reservoir | Replace with a TensorCircuit adapter/wrapper layer that contains this unsafe boundary. |
| `src/reservoir/models/reservoir/quantum/functional.py` | `_make_circuit_logic`: reverse brickwork CNOT odd pass | 214 | `type: ignore[bad-argument-type]` | TensorCircuit stubs do not express runtime backend CNOT overloads. | High | quantum-reservoir | Replace with a TensorCircuit adapter/wrapper layer or upstream-compatible stubs. |
| `src/reservoir/models/reservoir/quantum/functional.py` | `_make_circuit_logic`: reverse brickwork CNOT even pass | 217 | `type: ignore[bad-argument-type]` | TensorCircuit stubs do not express runtime backend CNOT overloads. | High | quantum-reservoir | Replace with a TensorCircuit adapter/wrapper layer or upstream-compatible stubs. |
