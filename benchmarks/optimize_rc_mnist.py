#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for Classical Reservoir Computing on MNIST.

Optimizes:
- Projection input_scale
- Reservoir spectral_radius
- Reservoir leak_rate
- Reservoir rc_connectivity
- Readout Ridge Alpha (if Ridge)

Target Preset: CLASSICAL_RESERVOIR_PRESET (Classification)

Usage:
uv run python benchmarks/optimize_rc_mnist.py
uv run python benchmarks/optimize_rc_mnist.py --n-trials 100
Visualization:
uv run optuna-dashboard  sqlite:////home/yoshi/PycharmProjects/Reservoir/benchmarks/optimize_rc.db
"""

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import optuna
import jax

jax.config.update("jax_enable_x64", True)

from reservoir.pipelines import run_pipeline  # noqa: E402
from reservoir.utils import check_gpu_available  # noqa: E402
from reservoir.models.presets import (  # noqa: E402
    CLASSICAL_RESERVOIR_PRESET,
    DEFAULT_RIDGE_READOUT,
)
from reservoir.models.config import (  # noqa: E402
    PolyRidgeReadoutConfig,
    RandomProjectionConfig, BoundedAffineScalerConfig,
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


def build_config(
        scale: float,
        relative_shift: float,
        input_scale: float,
        input_connectivity: float,
        bias_scale: float,
        spectral_radius: float,
        leak_rate: float,
        rc_connectivity: float,
        readout_config,
):
    """
    Build a PipelineConfig with dynamically updated parameters.
    """
    base = CLASSICAL_RESERVOIR_PRESET


    if isinstance(base.preprocess, BoundedAffineScalerConfig):
        new_prep = dataclasses.replace(
            base.preprocess,
            scale=scale,
            relative_shift=relative_shift,
        )

    else:
        # Fallback to original if not MinMaxScalerConfig
        new_prep = base.preprocess


    # Ensure base projection is RandomProjectionConfig
    if isinstance(base.projection, RandomProjectionConfig):
        new_proj = dataclasses.replace(
            base.projection, 
            input_scale=input_scale,
            input_connectivity=input_connectivity,
            bias_scale=bias_scale
        )
    else:
        # Fallback
        new_proj = base.projection

    # Update Reservoir (spectral_radius, leak_rate, connectivity)
    new_model = dataclasses.replace(
        base.model,
        spectral_radius=spectral_radius,
        leak_rate=leak_rate,
        rc_connectivity=rc_connectivity,
    )

    # Construct final config
    return dataclasses.replace(
        base,
        preprocess=new_prep,
        projection=new_proj,
        model=new_model,
        readout=readout_config,
    )



def make_objective(readout_config, dataset_enum: Dataset):
    """Factory that returns an Optuna objective."""

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        """

        
        # === 1. Suggest Parameters ===


        #=======Projection======================================================================

        # feature_min = trial.suggest_float("feature_min", -1.0, 0.0)
        # feature_max = trial.suggest_float("feature_max", 0.0, 1.0)



        # (BoundedAffine)

        scale = trial.suggest_float("scale", 0.0, 1.0)
        relative_shift = trial.suggest_float("relative_shift", -1.0, 1.0)


        #=======Projection======================================================================

        input_scale = trial.suggest_float("input_scale", 0.0, 1.0)

        input_connectivity = trial.suggest_float("input_connectivity", 0.0, 1.0)

        bias_scale = trial.suggest_float("bias_scale", 0.0, 1.0)


        # Reservoir
        spectral_radius = trial.suggest_float("spectral_radius", 0.5, 2.5)

        leak_rate = trial.suggest_float("leak_rate", 0.0, 1.0)

        rc_connectivity = trial.suggest_float("rc_connectivity", 0.0, 1.0)


        # === 2. Build Config ===
        config = build_config(
            scale=scale,
            relative_shift=relative_shift,
            input_scale=input_scale,
            input_connectivity=input_connectivity,
            bias_scale=bias_scale,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            rc_connectivity=rc_connectivity,
            readout_config=readout_config,
        )

        # === 3. Run Pipeline ===
        try:
            from typing import Any
            results: dict[str, Any] = run_pipeline(config, dataset_enum)

            # === 4. Extract & Store ALL Metrics ===
            test_results = results.get("test", {})
            train_results = results.get("train", {})
            
            # For classification, look for 'accuracy'
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

            if accuracy <= 0.11: # Random guess threshold roughly
                 trial.set_user_attr("status", "low_acc")
                 # We still return valid accuracy, but mark it.

            print(f"Trial {trial.number}: Acc={accuracy:.4f}, λ={best_lambda} "
                  f"(scale={scale:.2f}, shift={relative_shift:.2f}, in={input_scale:.2f}, ic={input_connectivity:.2f}, bs={bias_scale:.2f}, sr={spectral_radius:.2f}, lr={leak_rate:.2f}, rc={rc_connectivity:.2f})")

            trial.set_user_attr("status", "success")

        except ValueError as e:
            if "diverged" in str(e).lower():
                print(f"Trial {trial.number}: FAILED (Diverged) - {e}")
                trial.set_user_attr("status", "diverged")
                return 0.0
            else:
                print(f"Trial {trial.number}: EXCEPTION (ValueError) - {e}")
                trial.set_user_attr("status", "exception")
                return 0.0
        except RuntimeError as e:
            import traceback
            traceback.print_exc()
            print(f"Trial {trial.number}: EXCEPTION - {e}")
            trial.set_user_attr("status", "exception")
            return 0.0
        else:
            return accuracy

    return objective


def derive_names(readout_key: str, dataset_name: str):
    """Derive DB filename and study name from config components."""
    base = CLASSICAL_RESERVOIR_PRESET
    
    # Preprocess
    preprocess_name = type(base.preprocess).__name__.replace("Config", "")
    if "Standard" in preprocess_name:
        prep_tag = "Standard"
    elif "MinMax" in preprocess_name:
        prep_tag = "MinMaX"
    else:
        prep_tag = preprocess_name

    # Projection
    proj = base.projection
    if isinstance(proj, RandomProjectionConfig):
        proj_tag = f"Random{proj.n_units}"
    else:
        proj_tag = type(proj).__name__.replace("Config", "")

    # Study Name: optimize_rc_{Dataset}_{Preprocess}_{Projection}_{Readout}
    study_name = f"optimize_rc_{dataset_name.upper()}_{prep_tag}_{proj_tag}_{readout_key}_kai3"
    db_name = "optimize_rc.db" # Shared DB for RC optimization
    
    return study_name, db_name


def main():
    parser = argparse.ArgumentParser(description="Optuna RC Hyperparameter Search (MNIST)")
    parser.add_argument("--n-trials", type=int, default=500,
                        help="Number of optimization trials (default: 500)")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist"],
                        help="Dataset to optimize on (default: mnist)")
    parser.add_argument("--readout", type=str, default=None,
                        choices=list(READOUT_MAP.keys()),
                        help="Readout type (default: ridge)")
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

    # --- Readout ---
    if args.readout is not None:
        readout_key = args.readout
        readout_config = READOUT_MAP[readout_key]
    else:
        readout_config = CLASSICAL_RESERVOIR_PRESET.readout
        readout_key = "ridge"
        if isinstance(readout_config, PolyRidgeReadoutConfig):
             readout_key = "poly_full" if readout_config.mode == "full" else "poly_square"

    # --- Derive Names ---
    study_name, db_name = derive_names(readout_key, dataset_name)

    if args.study_name is not None:
        study_name = args.study_name
    
    if args.storage is None:
        db_path = Path(__file__).parent / db_name
        storage = f"sqlite:///{db_path}"
    else:
        storage = args.storage

    # --- Create Study ---
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="maximize", # Maximize Accuracy
        load_if_exists=True
    )

    historical_params = [
        # (min, max, input_scale, input_connectivity, bias_scale, spectral_radius, leak_rate, rc_connectivity)
        (-1.0, 1.0,
         0.03203273936326348, 0.7789498820486052, 0.6664704836440828,
         1.4068899032773172, 0.6854757236951003, 0.6213282741686085), # 0.8759999871253967 before warmup integration

        (0.0, 1.0,
         0.11524935267101843, 0.1381581069782872, 0.7862943891668603,
         1.4537351265687286, 0.7352675282751304, 0.4402710000522045), #0.8798999786376953 before warmup integration

        (-0.07768410112268466, 0.08160917176536134,
         1.0, 0.11458754901458218, 0.8295811429210161,
         1.45, 0.66,0.457758485877939), #0.877 before warmup integration

        (-0.7675280665952444, 0.35849784076318864,
         0.3543930218531782, 0.21745075681282766, 0.1725142451754484,
         1.921291918880454, 0.36449529864842045,0.6784641706491135), #id15 0.8753

    ]

    for min, max, input_scale, input_connectivity, bias_scale, spectral_radius, leak_rate, rc_connectivity in historical_params:
        bound = getattr(CLASSICAL_RESERVOIR_PRESET.preprocess, "bound", 1.0)
        maxminusmin = max - min
        maxplusmin = max + min
        scale = maxminusmin / (2.0 * bound)
        rs_denom = 2.0 * bound - maxminusmin
        rs = maxplusmin / rs_denom if rs_denom != 0 else 0.0

        # あなたが見つけたベストな値を登録
        study.enqueue_trial({
            "scale":scale,
            "relative_shift": rs,
            "input_scale": input_scale,
            "input_connectivity": input_connectivity,
            "bias_scale": bias_scale,
            "spectral_radius": spectral_radius,
            "leak_rate": leak_rate,
            "rc_connectivity": rc_connectivity,
        })




    print("=" * 60)
    print("Optuna RC Hyperparameter Search (MNIST)")
    print("=" * 60)
    print(f"  Study:   {study_name}")
    print(f"  Storage: {storage}")
    print(f"  Dataset: {dataset_name} ({dataset_enum})")
    print(f"  Trials:  {args.n_trials}")
    print(f"  Readout: {readout_key}")
    print("=" * 60)

    # --- Run ---
    objective_fn = make_objective(readout_config, dataset_enum)
    study.optimize(objective_fn, n_trials=args.n_trials)

    # --- Report ---
    print("\n" + "=" * 60)
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
