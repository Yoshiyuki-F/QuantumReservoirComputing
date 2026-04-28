#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for Quantum Reservoir Computing on MNIST.

Optimizes:
- Projection: BoundedAffinePCA (scale, relative_shift)
- Quantum Reservoir (n_layers, feedback_scale, leak_rate)
- Readout (Ridge/Poly)

Target Preset: QUANTUM_RESERVOIR_PRESET (Classification)

Usage:
uv run python benchmarks/optimize_qrc_mnist.py
uv run python benchmarks/optimize_qrc_mnist.py --trials 100
Visualization:
uv run optuna-dashboard sqlite:////home/yoshi/PycharmProjects/Reservoir/benchmarks/optimize_qrc_mnist.db
"""

import os
# Force 64-bit precision before ANY other imports
os.environ["JAX_ENABLE_X64"] = "True"

import argparse
import dataclasses
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import optuna  # noqa: E402

from reservoir.utils import check_gpu_available  # noqa: E402
from reservoir.pipelines import run_pipeline  # noqa: E402
from reservoir.pipelines.strategies import DivergenceError  # noqa: E402
from reservoir.models.presets import (  # noqa: E402
    QUANTUM_RESERVOIR_PRESET,
    DEFAULT_RIDGE_READOUT,
)
from reservoir.models.config import (  # noqa: E402
    BoundedAffinePCAConfig,
    PolyRidgeReadoutConfig,
)
from reservoir.data.identifiers import Dataset  # noqa: E402

# ---------------------------------------------------------------------------
# Readout Map
# ---------------------------------------------------------------------------
_POLY_LAMBDAS = tuple(np.logspace(-12, 3, 30).tolist())

READOUT_MAP = {
    "ridge":       DEFAULT_RIDGE_READOUT,
    "poly_full":   PolyRidgeReadoutConfig(
        use_intercept=True, lambda_candidates=_POLY_LAMBDAS, degree=2, mode="full",
    ),
    "poly_square": PolyRidgeReadoutConfig(
        use_intercept=True, lambda_candidates=_POLY_LAMBDAS, degree=2, mode="square_only",
    ),
}

VALID_BASES = ("Z", "ZZ", "Z+ZZ")


def build_config(
        scale: float,
        relative_shift: float,
        n_layers: int,
        feedback_scale: float,
        leak_rate: float,
        use_reuploading: bool,
        measurement_basis: str,
        readout_config,
):
    """
    Build a PipelineConfig with dynamically updated parameters.
    Preprocessing from preset (StandardScaler).
    Projection: BoundedAffinePCA (scale, relative_shift).
    """
    base = QUANTUM_RESERVOIR_PRESET

    # Update Projection (BoundedAffinePCA with tuned scale/relative_shift locked to bound=np.pi)
    base_proj = base.projection
    n_units = int(getattr(base_proj, 'n_units', 6))
    new_proj = BoundedAffinePCAConfig(
        n_units=n_units,
        scale=scale,
        relative_shift=relative_shift,
        bound=np.pi,
    )

    # Update Reservoir (n_layers, feedback_scale, leak_rate, measurement_basis)
    new_model = dataclasses.replace(
        base.model,
        n_layers=n_layers,
        feedback_scale=feedback_scale,
        leak_rate=leak_rate,
        measurement_basis=measurement_basis,
        use_reuploading=use_reuploading,
    )

    # Construct final config (preprocess from preset)
    return dataclasses.replace(
        base,
        projection=new_proj,
        model=new_model,
        readout=readout_config,
    )


def make_objective(measurement_basis: str, readout_config, use_reuploading: bool, dataset_enum: Dataset):
    """Factory that returns an Optuna objective."""

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        Searches for optimal QRC parameters to maximize Accuracy.
        """

        # === 1. Suggest Parameters ===

        # Projection (BoundedAffinePCA — controls QC input range)
        scale = trial.suggest_float("scale", 0.00000001, 1.0)

        # relative_shift = trial.suggest_float("relative_shift", -1.0, 1.0)
        relative_shift = trial.suggest_float("relative_shift", 0.0, 0.0)

        # Reservoir
        n_layers = trial.suggest_int("n_layers", 1, 1)
        feedback_scale = trial.suggest_float("feedback_scale", np.pi, np.pi)

        leak_rate = trial.suggest_float("leak_rate", 0.0, 1.0)

        # === 2. Build Config ===
        config = build_config(
            scale=scale,
            relative_shift=relative_shift,
            n_layers=n_layers,
            feedback_scale=feedback_scale,
            leak_rate=leak_rate,
            use_reuploading=use_reuploading,
            measurement_basis=measurement_basis,
            readout_config=readout_config,
        )

        # === 3. Run Pipeline ===
        try:
            results = run_pipeline(config, dataset_enum)

            # === 4. Extract Metrics ===
            test_results = results.get("test", {})
            train_results = results.get("train", {})

            accuracy = test_results.get("accuracy", 0.0)
            best_lambda = train_results.get("best_lambda", None)

            # Store best_lambda
            if best_lambda is not None:
                trial.set_user_attr("best_lambda", float(best_lambda))

            # Store extra metrics
            for key in ["accuracy", "precision", "recall", "f1"]:
                val = test_results.get(key, None)
                if val is not None:
                    trial.set_user_attr(key, float(val))

            if accuracy <= 0.11:  # Random guess threshold roughly
                trial.set_user_attr("status", "low_acc")

            print(f"Trial {trial.number}: Acc={accuracy:.4f}, λ={best_lambda} "
                  f"(scale={scale:.4f}, shift={relative_shift:.4f}, "
                  f"L={n_layers}, fb={feedback_scale:.4f}, lr={leak_rate:.4f})")

            trial.set_user_attr("status", "success")

        except DivergenceError as e:
            # Retrieve stats attached to DivergenceError (from strategies.py)
            if hasattr(e, "stats") and isinstance(e.stats, dict):
                print(f"    [Divergence Stats] {e.stats}")
                for key, val in e.stats.items():
                    trial.set_user_attr(key, float(val))
            
            print(f"Trial {trial.number}: FAILED (Diverged) - {e}")
            trial.set_user_attr("status", "diverged")
            trial.set_user_attr("error", str(e))
            return 0.0

        except ValueError as e:
            if "nan detected" in str(e).lower() or "diverged" in str(e).lower():
                print(f"Trial {trial.number}: FAILED (Diverged) - {e}")
                trial.set_user_attr("status", "diverged")
                trial.set_user_attr("error", str(e))
                return 0.0
            else:
                print(f"Trial {trial.number}: EXCEPTION (ValueError) - {e}")
                trial.set_user_attr("status", "exception")
                trial.set_user_attr("error", str(e))
                return 0.0
        except RuntimeError as e:
            import traceback
            traceback.print_exc()
            print(f"Trial {trial.number}: EXCEPTION - {e}")
            trial.set_user_attr("status", "exception")
            trial.set_user_attr("error", str(e))
            return 0.0
        else:
            return accuracy

    return objective


def derive_names(dataset_name: str, measurement_basis: str, readout_key: str, n_qubits: int, use_reuploading: bool):
    """Derive DB filename and study name from the variant combination."""
    reupload_str = "reupTrue" if use_reuploading else "reupFalse"
    
    base = QUANTUM_RESERVOIR_PRESET
    
    # Projection type (do not use exact label here as we are tuning it!)
    n_units = int(getattr(base.projection, 'n_units', 0))
    proj_tag = f"BAPCA{n_units}"
    
    scaler_tag = getattr(base.preprocess, "label", "raw").lower()

    study_name = f"qrc_{dataset_name}_{scaler_tag}_{proj_tag}_q{n_qubits}_{measurement_basis}_{readout_key}_{reupload_str}_kai3"
    db_name = "optimize_qrc_mnist.db"
    return study_name, db_name


def main():
    parser = argparse.ArgumentParser(description="Optuna QRC Hyperparameter Search (MNIST)")
    parser.add_argument("--trials", type=int, default=500,
                        help="Number of optimization trials (default: 500)")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist"],
                        help="Dataset to optimize on (default: mnist)")
    parser.add_argument("--measurement-basis", type=str, default=None,
                        choices=list(VALID_BASES),
                        help="Measurement basis (default: from preset)")
    parser.add_argument("--readout", type=str, default=None,
                        choices=list(READOUT_MAP.keys()),
                        help="Readout type (default: from preset)")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Override Optuna study name")
    parser.add_argument("--storage", type=str, default=None,
                        help="Override Optuna storage URL")
    parser.add_argument("--force-cpu", action="store_true", help="Force CPU usage")
    args = parser.parse_args()

    if not args.force_cpu:
        try:
            check_gpu_available()
        except RuntimeError as exc:
            print(f"Warning: GPU check failed ({exc}). Continuing...")

    # --- Dataset ---
    dataset_name = args.dataset
    if dataset_name == "mnist":
        dataset_enum = Dataset.MNIST
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # --- Resolve variant from args or preset defaults ---
    base = QUANTUM_RESERVOIR_PRESET

    # Measurement basis
    if args.measurement_basis is not None:
        measurement_basis = args.measurement_basis
    else:
        measurement_basis = base.model.measurement_basis
    
    # Qubits
    n_qubits = base.model.n_qubits

    # Re-uploading
    use_reuploading = base.model.use_reuploading

    # Readout
    if args.readout is not None:
        readout_key = args.readout
        readout_config = READOUT_MAP[readout_key]
    else:
        readout_config = base.readout
        # Reverse-lookup key from preset
        if isinstance(readout_config, PolyRidgeReadoutConfig):
            readout_key = "poly_full" if readout_config.mode == "full" else "poly_square"
        else:
            readout_key = "ridge"

    # --- Derive study / DB names ---
    study_name, db_name = derive_names(dataset_name, measurement_basis, readout_key, n_qubits, use_reuploading)

    if args.study_name is not None:
        study_name = args.study_name
    
    if args.storage is None:
        db_path = Path(__file__).parent / db_name
        storage = f"sqlite:///{db_path}"
    else:
        storage = args.storage

    # --- Create or load study ---
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize",
        load_if_exists=True
    )

    # # # old 0.72
    # scale, relative_shift, f, lr = 0.968869170885191, -0.8417302370197737, 3.196929844938574, 0.15402317414946048 #argmax
    #
    # historical_params = [
    #     # (scale, relative_shift, f, lr)
    #     (0.967577640940977, -0.6192870670956716, 2.875819387521165, 0.1570634471663206),
    #     (0.995636665702267, -0.5318107307637854, 3.1688977417811044, 0.15173143338527922),
    #     (0.9953090740632762, -0.35196129251452035, 2.8319293782780495, 0.156041081549988),  # 69
    #     (0.9987693745576063, -0.28464305548153845, 2.8270143649559527, 0.15513019484918852),
    #     (0.967644464086574, -0.4966138854302412, 2.946499351837381, 0.15772595736379288),
    #     (0.9708741572252473, -0.14283801381427502, 3.016445025661375, 0.1347945608473465),
    #     (0.9951850782391719, -0.4130287394341291, 3.0010278759205815, 0.13875726573433023),
    #     (0.8756732591397667, -0.4259870023669491, 3.0606019943722935, 0.12767110329901754)
    # ]
    #
    # for old_scale, old_rel_shift, f, lr in historical_params:
    #     # --- 新しい [-pi, pi] 空間用にパラメータを変換 ---
    #     b_actual = old_rel_shift * (1.0 - old_scale)
    #     a_new = old_scale / np.pi
    #     r_new = b_actual / ((1.0 - a_new) * np.pi)
    #
    #     # あなたが見つけたベストな値を登録
    #     study.enqueue_trial({
    #         "scale": a_new,
    #         "relative_shift": r_new,
    #         "feedback_scale": f,  # など他のパラメータも
    #         "leak_rate": lr,
    #         "n_layers": 1
    #     })

    print("=" * 60)
    print("Optuna QRC Hyperparameter Search (MNIST)")
    print("=" * 60)
    print(f"  Study            : {study_name}")
    print(f"  Storage          : {storage}")
    print(f"  Dataset          : {dataset_name} ({dataset_enum})")
    print(f"  Trials           : {args.trials}")
    print(f"  Measurement Basis: {measurement_basis}")
    print(f"  Readout          : {readout_key}")
    print(f"  Re-uploading     : {use_reuploading}")
    print("=" * 60)

    # --- Run ---
    objective_fn = make_objective(measurement_basis, readout_config, use_reuploading, dataset_enum)
    study.optimize(objective_fn, n_trials=args.trials)
    
    # --- Report ---
    print("" + "=" * 60)
    print("=== BEST PARAMETERS ===")
    print("=" * 60)
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v:.6f}")
    if study.best_value is not None:
        print(f"  {'Best Accuracy':20s}: {study.best_value:.4f}")
    print("-" * 60)
    print("  [Stored Metrics]")
    for k, v in study.best_trial.user_attrs.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.6f}")
        else:
            print(f"  {k:20s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
