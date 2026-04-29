#!/usr/bin/env python3
"""
Optuna Hyperparameter Search for Classical Reservoir Computing (Multi-Seed).

Optimizes:
- Preprocess feature_min
- Preprocess feature_max
- Projection input_scale
- Reservoir spectral_radius
- Reservoir leak_rate
- Reservoir rc_connectivity

Target Preset: TIME_CLASSICAL_RESERVOIR_PRESET

Usage:
uv run python benchmarks/optimize_rc_multi_seed.py --dataset lorenz
uv run python benchmarks/optimize_rc_multi_seed.py --dataset mackey_glass --trials 100
Visualization:
uv run optuna-dashboard sqlite:////home/yoshi/PycharmProjects/Reservoir/benchmarks/optimize_rc_mean_vpt.db

uv run python benchmarks/optimize_rc_multi_seed.py --dataset mackey_glass --enqueue-csv benchmarks/filtered_optuna_results.csv

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
    ReadoutConfig,
    BoundedAffineScalerConfig,
    PolyRidgeReadoutConfig,
    RandomProjectionConfig,
    ClassicalReservoirConfig,
)
from reservoir.data.identifiers import Dataset  # noqa: E402

if TYPE_CHECKING:
    from reservoir.core.types import EvalMetrics, ResultDict, TestMetrics, TrainMetrics


def _set_trial_float_attr(trial: optuna.Trial, key: str, value: float | None) -> None:
    if value is not None:
        trial.set_user_attr(key, float(value))


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
        bound: float,
        input_scale: float,
        input_connectivity: float,
        bias_scale: float,
        spectral_radius: float,
        leak_rate: float,
        rc_connectivity: float,
        readout_config: ReadoutConfig | None,
        seed: int,
):
    """
    Build a PipelineConfig with dynamically updated parameters.
    """
    base = TIME_CLASSICAL_RESERVOIR_PRESET

    # Update Preprocess (BoundedAffineScaler)
    new_prep = BoundedAffineScalerConfig(
        scale=scale,
        relative_shift=relative_shift,
        bound=bound,
    )

    # Update Projection (input_scale, seed)
    if isinstance(base.projection, RandomProjectionConfig):
        new_proj = dataclasses.replace(
            base.projection, 
            input_scale=input_scale,
            input_connectivity=input_connectivity,
            bias_scale=bias_scale,
            seed=seed,
        )
    else:
        new_proj = base.projection

    # Update Reservoir (spectral_radius, leak_rate, connectivity, seed)
    if isinstance(base.model, ClassicalReservoirConfig):
        new_model = dataclasses.replace(
            base.model,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            rc_connectivity=rc_connectivity,
            seed=seed,
        )
    else:
        new_model = base.model

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
        Searches for optimal RC parameters to maximize mean_vpt across 5 seeds.
        """

        
        # === 1. Suggest Parameters ===
        
        # ======================== Preprocessing (Bounded Affine) ====================
        scale = trial.suggest_float("scale", 0.001, 1.0)
        relative_shift = trial.suggest_float("relative_shift", -1.0, 1.0)
        bound = np.pi  # Fixed to pi to maintain bijective mapping

        # Projection
        input_scale = trial.suggest_float("input_scale", 0.1, 1.0)
        input_connectivity = trial.suggest_float("input_connectivity", 0.0, 1.0)
        bias_scale = trial.suggest_float("bias_scale", 0.0, 1.0)

        # Reservoir
        spectral_radius = trial.suggest_float("spectral_radius", 0.0, 2.0)
        leak_rate = trial.suggest_float("leak_rate", 0.0, 1.0)
        rc_connectivity = trial.suggest_float("rc_connectivity", 0.0, 1.0)

        # === 2. Run Pipeline over multiple seeds ===
        seeds = [40, 41, 42, 43, 44]
        vpts: list[float] = []
        
        print(f"Trial {trial.number}: Starting (scale={scale:.3f}, shift={relative_shift:.3f}, bound={bound:.3f}, in={input_scale:.2f}, ic={input_connectivity:.2f}, bs={bias_scale:.2f}, sr={spectral_radius:.2f}, lr={leak_rate:.2f}, rc={rc_connectivity:.2f})")

        for seed in seeds:
            config = build_config(
                scale=scale,
                relative_shift=relative_shift,
                bound=bound,
                input_scale=input_scale,
                input_connectivity=input_connectivity,
                bias_scale=bias_scale,
                spectral_radius=spectral_radius,
                leak_rate=leak_rate,
                rc_connectivity=rc_connectivity,
                readout_config=readout_config,
                seed=seed,
            )

            try:
                results: ResultDict = run_pipeline(config, dataset_enum)

                test_results_raw = results.get("test")
                if test_results_raw is None:
                    test_results: TestMetrics = {}
                else:
                    test_results = test_results_raw
                val_results_raw = results.get("validation")
                if val_results_raw is None:
                    val_results: EvalMetrics = {}
                else:
                    val_results = val_results_raw
                train_results_raw = results.get("train")
                if train_results_raw is None:
                    train_results: TrainMetrics = {}
                else:
                    train_results = train_results_raw

                test_chaos_raw = test_results.get("chaos_metrics")
                if test_chaos_raw is None:
                    test_chaos: dict[str, float] = {}
                else:
                    test_chaos = test_chaos_raw

                vpt_lt = test_results.get("vpt_lt", 0.0)
                val_vpt_lt = val_results.get("vpt_lt", 0.0)
                best_lambda = train_results.get("best_lambda", None)

                if best_lambda is not None:
                    trial.set_user_attr(f"best_lambda_seed{seed}", float(best_lambda))

                # Store TEST chaos metrics per seed
                for key in ["mse", "nmse", "nrmse", "mase", "ndei",
                            "var_ratio", "correlation", "vpt_steps", "vpt_lt", "vpt_threshold"]:
                    val = test_chaos.get(key, None)
                    if val is not None:
                        trial.set_user_attr(f"{key}_seed{seed}", float(val))

                # Store VALIDATION chaos metrics per seed (prefix with val_)
                _set_trial_float_attr(trial, f"val_mse_seed{seed}", val_results.get("mse"))
                _set_trial_float_attr(trial, f"val_nmse_seed{seed}", val_results.get("nmse"))
                _set_trial_float_attr(trial, f"val_nrmse_seed{seed}", val_results.get("nrmse"))
                _set_trial_float_attr(trial, f"val_mase_seed{seed}", val_results.get("mase"))
                _set_trial_float_attr(trial, f"val_ndei_seed{seed}", val_results.get("ndei"))
                _set_trial_float_attr(trial, f"val_var_ratio_seed{seed}", val_results.get("var_ratio"))
                _set_trial_float_attr(trial, f"val_correlation_seed{seed}", val_results.get("correlation"))
                _set_trial_float_attr(trial, f"val_vpt_steps_seed{seed}", val_results.get("vpt_steps"))
                _set_trial_float_attr(trial, f"val_vpt_lt_seed{seed}", val_results.get("vpt_lt"))

                if vpt_lt is None or math.isnan(vpt_lt) or vpt_lt <= 0:
                    print(f"    Seed {seed}: FAILED (VPT=0). Failing early.")
                    trial.set_user_attr(f"vpt_lt_seed{seed}", 0.0)
                    trial.set_user_attr("status", "failed_early_vpt0")
                    return -1.0

                vpts.append(vpt_lt)
                trial.set_user_attr(f"vpt_lt_seed{seed}", float(vpt_lt))
                print(f"    Seed {seed}: VPT={vpt_lt:.2f} LT (Val VPT={val_vpt_lt:.2f}), MSE={test_chaos.get('mse',0):.5f}, λ={best_lambda:.2e}")

            except DivergenceError as e:
                print(f"    Seed {seed}: FAILED (Diverged) - {e}. Failing early.")
                trial.set_user_attr(f"error_seed{seed}", str(e))
                trial.set_user_attr(f"status_seed{seed}", "diverged")
                trial.set_user_attr("status", "failed_early_diverged")
                return -0.2

            except (ValueError, RuntimeError) as e:
                err_msg = str(e)
                err_msg_lower = err_msg.lower()
                trial.set_user_attr(f"error_seed{seed}", err_msg)
                
                if "nan detected" in err_msg_lower:
                    print(f"    Seed {seed}: FAILED (NaN) - {e}. Failing early.")
                    trial.set_user_attr(f"status_seed{seed}", "nan_error")
                    trial.set_user_attr("status", "failed_early_nan")
                    return -0.5
                elif "validation nmse too high" in err_msg_lower:
                    try:
                        nmse_val = float(err_msg.split(":")[-1].strip())
                    except (ValueError, IndexError):
                        nmse_val = 1.0
                    print(f"    Seed {seed}: FAILED (NMSE high) - {e}. Failing early.")
                    trial.set_user_attr(f"status_seed{seed}", "failed_nmse")
                    trial.set_user_attr(f"nmse_seed{seed}", nmse_val)
                    trial.set_user_attr("status", "failed_early_nmse")
                    return -nmse_val
                else:
                    print(f"    Seed {seed}: EXCEPTION - {e}. Failing early.")
                    trial.set_user_attr(f"status_seed{seed}", "exception")
                    trial.set_user_attr("status", "failed_early_exception")
                    return -1.0
        
        # === 3. Evaluate Mean ===
        mean_vpt = sum(vpts) / len(vpts)
        trial.set_user_attr("mean_vpt", float(mean_vpt))
        
        # Determine overall status
        if any(v <= 0 for v in vpts):
             trial.set_user_attr("status", "failed_or_partial")
             print(f"Trial {trial.number}: Mean VPT={mean_vpt:.2f} (Contains Failures)")
        else:
             trial.set_user_attr("status", "success")
             print(f"Trial {trial.number}: SUCCESS, Mean VPT={mean_vpt:.2f} LT")

        return float(mean_vpt)

    return objective


def derive_names(readout_key: str, dataset_name: str):
    """Derive DB filename and study name from config components."""
    base = TIME_CLASSICAL_RESERVOIR_PRESET
    
    # Preprocess
    prep_tag = "BoundedAffine"

    # Projection
    proj = base.projection
    if isinstance(proj, RandomProjectionConfig):
        proj_tag = f"Random{proj.n_units}"
    else:
        proj_tag = type(proj).__name__.replace("Config", "")

    # Study Name: optimize_rc_{Dataset}_{Preprocess}_{Projection}_{Readout}_mean_vpt
    study_name = f"optimize_rc_{dataset_name.upper()}_{prep_tag}_{proj_tag}_{readout_key}_mean_vpt"
    db_name = "optimize_rc_mean_vpt.db"
    
    return study_name, db_name


def main():
    parser = argparse.ArgumentParser(description="Optuna RC Hyperparameter Search (Multi-Seed Mean VPT)")
    parser.add_argument("--trials", type=int, default=500,
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
    parser.add_argument("--enqueue-csv", type=str, default=None,
                        help="Path to a CSV file to enqueue trials from")
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
    print("Optuna RC Hyperparameter Search (Multi-Seed Mean VPT)")
    print("=" * 60)
    print(f"  Study:   {study_name}")
    print(f"  Storage: {storage}")
    print(f"  Dataset: {dataset_name} ({dataset_enum})")
    print(f"  Trials:  {args.trials}")
    print(f"  Readout: {readout_key}")
    print("=" * 60)

    # --- Enqueue from CSV if provided ---
    if args.enqueue_csv:
        import csv
        csv_path = Path(args.enqueue_csv)
        if csv_path.exists():
            print(f"\n[+] Enqueueing trials from {csv_path}...")
            with open(csv_path, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    params: dict[str, float | int] = {}
                    for k, v in row.items():
                        if k.startswith('params_'):
                            param_name = k.replace('params_', '')
                            try:
                                params[param_name] = float(v)
                            except ValueError:
                                pass
                    
                    # Convert legacy MinMax params (feature_min/max) to BoundedAffine params (scale/shift)
                    if 'feature_max' in params:
                        f_max = params.pop('feature_max')
                        # feature_min が CSV にない場合はデフォルトの 0 (または -1) を想定
                        f_min = params.pop('feature_min', 0.0)
                        bound = np.pi
                        
                        # Mathematical conversion to BoundedAffine space:
                        # scale = (max - min) / (2 * bound)
                        # shift = (max + min) / (2 * bound * (1 - scale))
                        scale = (f_max - f_min) / (2 * bound)
                        if scale < 1.0:
                            relative_shift = (f_max + f_min) / (2 * bound * (1.0 - scale))
                        else:
                            relative_shift = 0.0
                        
                        # 探索範囲内にクリップしてエラーを防止
                        params['scale'] = max(0.001, min(1.0, scale))
                        params['relative_shift'] = max(-1.0, min(1.0, relative_shift))
                    
                    if params:
                        study.enqueue_trial(params, skip_if_exists=True)
                        count += 1
            print(f"[+] Successfully enqueued {count} trials.\n")
        else:
            print(f"\n[!] Warning: CSV file not found at {csv_path}\n")

    # --- Run ---
    objective_fn = make_objective(readout_config, dataset_enum)
    study.optimize(objective_fn, n_trials=args.trials)

    # --- Report ---
    print("\n" + "=" * 60)
    print("=== BEST PARAMETERS ===")
    print("=" * 60)
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v:.6f}")
    if study.best_value is not None:
        print(f"  {'Best Mean VPT':20s}: {study.best_value:.2f} LT")
    print("-" * 60)
    print("  [Stored Metrics (Best Trial)]")
    for k, v in study.best_trial.user_attrs.items():
        if isinstance(v, float):
            print(f"  {k:20s}: {v:.6f}")
        else:
            print(f"  {k:20s}: {v}")
    print("=" * 60)


if __name__ == "__main__":
    main()
