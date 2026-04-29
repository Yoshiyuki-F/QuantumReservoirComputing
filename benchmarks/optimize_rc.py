#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for Classical Reservoir Computing.

Optimizes:
- Preprocess feature_min
- Preprocess feature_max
- Projection input_scale
- Reservoir spectral_radius
- Reservoir leak_rate
- Reservoir rc_connectivity

Target Preset: TIME_CLASSICAL_RESERVOIR_PRESET

Usage:
uv run python benchmarks/optimize_rc.py
uv run python benchmarks/optimize_rc.py --n-trials 100
Visualization:
uv run optuna-dashboard  sqlite:////home/yoshi/PycharmProjects/Reservoir/benchmarks/optimize_rc.db
"""
from __future__ import annotations

import argparse
import dataclasses
import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import optuna
import jax

jax.config.update("jax_enable_x64", True)

from reservoir.pipelines import run_pipeline  # noqa: E402
from reservoir.pipelines.strategies import DivergenceError  # noqa: E402
from reservoir.utils import check_gpu_available  # noqa: E402
from reservoir.models.presets import (  # noqa: E402
    TIME_CLASSICAL_RESERVOIR_PRESET,
    DEFAULT_RIDGE_READOUT,
)
from reservoir.models.config import (  # noqa: E402
    ClassicalReservoirConfig,
    ReadoutConfig,
    MinMaxScalerConfig,
    PolyRidgeReadoutConfig,
    RandomProjectionConfig,
)
from reservoir.data.identifiers import Dataset  # noqa: E402

if TYPE_CHECKING:
    from reservoir.core.types import ResultDict, TestMetrics, TrainMetrics


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
        feature_min: float,
        feature_max: float,
        input_scale: float,
        input_connectivity: float,
        bias_scale: float,
        spectral_radius: float,
        leak_rate: float,
        rc_connectivity: float,
        readout_config: ReadoutConfig | None,
):
    """
    Build a PipelineConfig with dynamically updated parameters.
    """
    base = TIME_CLASSICAL_RESERVOIR_PRESET
    base_model = base.model
    if not isinstance(base_model, ClassicalReservoirConfig):
        raise TypeError("TIME_CLASSICAL_RESERVOIR_PRESET.model must be ClassicalReservoirConfig")

    # Update Preprocess (MinMaxScaler)
    if isinstance(base.preprocess, MinMaxScalerConfig):
        new_prep = dataclasses.replace(
            base.preprocess,
            feature_min=feature_min,
            feature_max=feature_max,
        )
    else:
        new_prep = base.preprocess

    # Update Projection (input_scale)
    # Ensure base projection is RandomProjectionConfig
    if isinstance(base.projection, RandomProjectionConfig):
        new_proj = dataclasses.replace(
            base.projection, 
            input_scale=input_scale,
            input_connectivity=input_connectivity,
            bias_scale=bias_scale
        )
    else:
        # Fallback if changed, but expected to be RP
        new_proj = base.projection

    # Update Reservoir (spectral_radius, leak_rate, connectivity)
    new_model = dataclasses.replace(
        base_model,
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



def make_objective(readout_config: ReadoutConfig | None, dataset_enum: Dataset):
    """Factory that returns an Optuna objective."""

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        """

        
        # === 1. Suggest Parameters ===
        
        # Preprocess
        feature_min = trial.suggest_float("feature_min", -4.0, 0.0)
        feature_max = trial.suggest_float("feature_max", 0.0, 4.0)

        # Projection
        input_scale = trial.suggest_float("input_scale", 0.1, 1.0)
        # input_scale = trial.suggest_float("input_scale", 1.0, 1.0)

        # input_connectivity = trial.suggest_float("input_connectivity", 0.0, 1.0)
        input_connectivity = trial.suggest_float("input_connectivity", 0.48684898425332923, 0.48684898425332923)

        bias_scale = trial.suggest_float("bias_scale", 0.0, 1.0)
        # bias_scale = trial.suggest_float("bias_scale", 0.6664704836440828, 0.6664704836440828)

        # Reservoir
        # spectral_radius = trial.suggest_float("spectral_radius", 0.5, 2.5)
        spectral_radius = trial.suggest_float("spectral_radius", 1.198057566868823, 1.198057566868823)

        leak_rate = trial.suggest_float("leak_rate", 0.0, 1.0)
        # leak_rate = trial.suggest_float("leak_rate", 0.66, 0.66)

        rc_connectivity = trial.suggest_float("rc_connectivity", 0.0, 1.0)
        # rc_connectivity = trial.suggest_float("rc_connectivity", 0.6784641706491135,  0.6784641706491135)


        # === 2. Build Config ===
        config = build_config(
            feature_min=feature_min,
            feature_max=feature_max,
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
            results: ResultDict = run_pipeline(config, dataset_enum)

            # === 4. Extract & Store ALL Metrics ===
            test_results_raw = results.get("test")
            if test_results_raw is None:
                test_results: TestMetrics = {}
            else:
                test_results = test_results_raw
            train_results_raw = results.get("train")
            if train_results_raw is None:
                train_results: TrainMetrics = {}
            else:
                train_results = train_results_raw
            chaos_raw = test_results.get("chaos_metrics")
            if chaos_raw is None:
                chaos: dict[str, float] = {}
            else:
                chaos = chaos_raw

            vpt_lt = test_results.get("vpt_lt", 0.0)
            best_lambda = train_results.get("best_lambda", None)

            # Store best_lambda
            if best_lambda is not None:
                trial.set_user_attr("best_lambda", float(best_lambda))

            # Store ALL chaos metrics
            for key in ["mse", "nmse", "nrmse", "mase", "ndei",
                        "var_ratio", "correlation", "vpt_steps", "vpt_lt", "vpt_threshold"]:
                val = chaos.get(key, None)
                if val is not None:
                    trial.set_user_attr(key, float(val))

            if vpt_lt is None or math.isnan(vpt_lt) or vpt_lt <= 0:
                print(f"Trial {trial.number}: FAILED (VPT=0) λ={best_lambda}")
                trial.set_user_attr("status", "failed")
                return -1.0

            print(f"Trial {trial.number}: VPT={vpt_lt:.2f} LT, MSE={chaos.get('mse',0):.5f}, λ={best_lambda:.2e} "
                  f"(min={feature_min:.2f}, max={feature_max:.2f}, in={input_scale:.2f}, ic={input_connectivity:.2f}, bs={bias_scale:.2f}, sr={spectral_radius:.2f}, lr={leak_rate:.2f}, rc={rc_connectivity:.2f})")

            trial.set_user_attr("status", "success")

        except DivergenceError as e:
            if hasattr(e, "stats") and isinstance(e.stats, dict):
                print(f"    [Divergence Stats] {e.stats}")
                for key, val in e.stats.items():
                    trial.set_user_attr(key, float(val))
            
            print(f"Trial {trial.number}: FAILED (Diverged) - {e}")
            trial.set_user_attr("status", "diverged")
            trial.set_user_attr("error", str(e))
            return -0.2

        except (ValueError, RuntimeError) as e:
            if hasattr(e, "stats") and isinstance(e.stats, dict):
                print(f"    [Exception Stats] {e.stats}")
                for key, val in e.stats.items():
                    trial.set_user_attr(key, float(val))
            
            err_msg = str(e)
            err_msg_lower = err_msg.lower()
            if "nan detected" in err_msg_lower:
                 print(f"Trial {trial.number}: FAILED (NaN) - {e}")
                 trial.set_user_attr("status", "nan_error")
                 trial.set_user_attr("error", err_msg)
                 return -0.5
            elif "validation nmse too high" in err_msg_lower:
                 try:
                     nmse_val = float(err_msg.split(":")[-1].strip())
                 except (ValueError, IndexError):
                     nmse_val = 1.0
                 print(f"Trial {trial.number}: FAILED (NMSE high) - {e}")
                 trial.set_user_attr("status", "failed_nmse")
                 trial.set_user_attr("nmse", nmse_val)
                 trial.set_user_attr("error", err_msg)
                 return -nmse_val
            else:
                 print(f"Trial {trial.number}: EXCEPTION - {e}")
                 trial.set_user_attr("status", "exception")
                 trial.set_user_attr("error", err_msg)
                 return -1.0
        else:
            return vpt_lt

    return objective


def derive_names(readout_key: str, dataset_name: str):
    """Derive DB filename and study name from config components."""
    base = TIME_CLASSICAL_RESERVOIR_PRESET
    
    # Preprocess
    preprocess_name = type(base.preprocess).__name__.replace("Config", "")
    if "Standard" in preprocess_name:
        prep_tag = "Standard"
    elif "MinMax" in preprocess_name:
        prep_tag = "MinMaX"
    else:
        prep_tag = preprocess_name

    # Custom MinMax tag
    if isinstance(base.preprocess, MinMaxScalerConfig):
        prep_tag = "MinMax"

    # Projection
    proj = base.projection
    if isinstance(proj, RandomProjectionConfig):
        proj_tag = f"Random{proj.n_units}"
    else:
        proj_tag = type(proj).__name__.replace("Config", "")

    # Study Name: optimize_rc_{Dataset}_{Preprocess}_{Projection}_{Readout}
    study_name = f"optimize_rc_{dataset_name.upper()}_{prep_tag}_{proj_tag}_{readout_key}_kai2"
    db_name = "optimize_rc.db" # Shared DB for RC optimization
    
    return study_name, db_name


def main():
    parser = argparse.ArgumentParser(description="Optuna RC Hyperparameter Search")
    parser.add_argument("--n-trials", type=int, default=5000,
                        help="Number of optimization trials (default: 500)")
    parser.add_argument("--dataset", type=str, default="mackey_glass",
                        choices=["mackey_glass", "lorenz", "lorenz96"],
                        help="Dataset to optimize on (default: mackey_glass)")
    parser.add_argument("--readout", type=str, default=None,
                        choices=list(READOUT_MAP.keys()),
                        help="Readout type (default: ridge)")
    parser.add_argument("--study-name", type=str, default=None,
                        help="Override Optuna study name")
    parser.add_argument("--storage", type=str, default=None,
                        help="Override Optuna storage URL")
    args = parser.parse_args()

    try:
        check_gpu_available()
    except RuntimeError as exc:
        raise ValueError(f"Warning: GPU check failed ({exc}). Continuing...") from exc

    # --- Dataset ---
    dataset_name = args.dataset
    if dataset_name == "mackey_glass":
        dataset_enum = Dataset.MACKEY_GLASS
    elif dataset_name == "lorenz":
        dataset_enum = Dataset.LORENZ
    elif dataset_name == "lorenz96":
        dataset_enum = Dataset.LORENZ96
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # --- Readout ---
    if args.readout is not None:
        readout_key = args.readout
        readout_config = READOUT_MAP[readout_key]
    else:
        readout_config = TIME_CLASSICAL_RESERVOIR_PRESET.readout
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
        direction="maximize",
        load_if_exists=True
    )

    print("=" * 60)
    print("Optuna RC Hyperparameter Search")
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
        print(f"  {'Best VPT':20s}: {study.best_value:.2f} LT")
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
