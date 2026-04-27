"""
Optuna Hyperparameter Search for Quantum Reservoir Computing (Multi-Seed).
Usage:
uv run python benchmarks/optimize_qrc_multi_seed.py --dataset mackey_glass
uv run python benchmarks/optimize_qrc_multi_seed.py --dataset mackey_glass --trials 500 --enqueue-csv benchmarks/filtered_optuna_results.csv
uv run optuna-dashboard sqlite:////home/yoshi/PycharmProjects/Reservoir/benchmarks/optuna_qrc_nonetype_mean_vpt.db
"""
import os
# Force 64-bit precision before ANY other imports
os.environ["JAX_ENABLE_X64"] = "True"

import argparse
import dataclasses
import math
from pathlib import Path

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import optuna

from reservoir.utils import check_gpu_available
from reservoir.pipelines import run_pipeline
from reservoir.pipelines.strategies import DivergenceError
from reservoir.models.presets import (
    TIME_QUANTUM_RESERVOIR_PRESET,
    DEFAULT_RIDGE_READOUT,
)
from reservoir.models.config import (
    PolyRidgeReadoutConfig,
)
from reservoir.data.identifiers import Dataset


# ---------------------------------------------------------------------------
# Readout Lookup  (ridge / poly_full / poly_square)
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
        bound: float,
        n_layers: int,
        feedback_scale: float,
        leak_rate: float,
        use_reuploading: bool,
        seed: int,
        *,
        measurement_basis: str,
        readout_config,
):
    """
    Build a PipelineConfig with dynamically updated parameters.
    
    scale, relative_shift, bound are applied via BoundedAffineScalerConfig (preprocessing).
    projection is None (Step 3 skipped).
    """
    from reservoir.models.config import BoundedAffineScalerConfig
    base = TIME_QUANTUM_RESERVOIR_PRESET

    # Update preprocessing (BoundedAffineScaler)
    new_preprocess = BoundedAffineScalerConfig(
        scale=scale, 
        relative_shift=relative_shift,
        bound=bound
    )

    # Update model (feedback_scale, leak_rate, measurement_basis, seed)
    new_model = dataclasses.replace(
        base.model,
        n_layers=n_layers,
        feedback_scale=feedback_scale,
        leak_rate=leak_rate,
        measurement_basis=measurement_basis,
        use_reuploading=use_reuploading,
        seed=seed,
    )

    # Construct final config (projection=None, no Step 3)
    return dataclasses.replace(
        base,
        preprocess=new_preprocess,
        model=new_model,
        readout=readout_config,
    )


def make_objective(measurement_basis: str, readout_config, use_reuploading: bool, dataset_enum: Dataset):
    """Factory that returns an Optuna objective closed over the study variant."""

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        Searches for optimal QRC parameters to maximize mean_vpt across 5 seeds.
        """
        # === 1. Suggest Parameters ===

        # ======================== Preprocessing (Bounded Affine) ====================
        scale = trial.suggest_float("scale", 0.001, 1.0)
        relative_shift = trial.suggest_float("relative_shift", -1.0, 1.0)
        bound = np.pi  # Fixed to pi to maintain bijective mapping

        # ======================== Reservoir ==================================
        # n_layers = trial.suggest_categorical("n_layers", [1, 2, 3,4,5,6,7])

        feedback_scale = trial.suggest_float("feedback_scale", 0, np.pi)
        leak_rate = trial.suggest_float("leak_rate", 0, 1)

        # === 2. Run Pipeline over multiple seeds ===
        seeds = [40, 41, 42, 43, 44]
        vpts = []
        
        print(f"Trial {trial.number}: Starting (scale={scale:.3f}, shift={relative_shift:.3f}, bound={bound:.3f}, fb={feedback_scale:.3f}, lr={leak_rate:.3f})")

        for seed in seeds:
            config = build_config(
                scale,
                relative_shift,
                bound,
                n_layers,
                feedback_scale,
                leak_rate,
                use_reuploading,
                seed,
                measurement_basis=measurement_basis,
                readout_config=readout_config,
            )

            try:
                results: dict[str] = run_pipeline(config, dataset_enum)

                test_results = results.get("test", {})
                val_results = results.get("validation", {}) # Reporter uses "validation"
                train_results = results.get("train", {})
                
                test_chaos = test_results.get("chaos_metrics", {})
                # For validation, metrics are flat in val_results due to Strategy logic
                val_chaos = val_results

                vpt_lt = test_results.get("vpt_lt", 0.0)
                val_vpt_lt = val_results.get("vpt_lt", 0.0)
                best_lambda = train_results.get("best_lambda", None)
                
                if best_lambda is not None:
                    trial.set_user_attr(f"best_lambda_seed{seed}", float(best_lambda))

                # Store TEST chaos metrics per seed
                for key in ["var_ratio", "vpt_steps", "vpt_lt"]:
                    val = test_chaos.get(key, None)
                    if val is not None:
                        trial.set_user_attr(f"{key}_seed{seed}", float(val))

                # Store VALIDATION chaos metrics per seed (prefix with val_)
                for key in ["mse", "nmse", "nrmse", "mase", "ndei",
                            "var_ratio", "correlation", "vpt_steps", "vpt_lt"]:
                    val = val_chaos.get(key, None)
                    if val is not None:
                        trial.set_user_attr(f"val_{key}_seed{seed}", float(val))

                if vpt_lt is None or math.isnan(vpt_lt) or vpt_lt <= 0:
                    print(f"    Seed {seed}: FAILED (VPT=0). Failing early.")
                    trial.set_user_attr(f"vpt_lt_seed{seed}", 0.0)
                    trial.set_user_attr("status", "failed_early_vpt0")
                    return -1.0
                
                vpts.append(vpt_lt)
                trial.set_user_attr(f"vpt_lt_seed{seed}", float(vpt_lt))
                print(f"    Seed {seed}: VPT={vpt_lt:.2f} LT (Val VPT={val_vpt_lt:.2f})")

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


def derive_names(dataset_enum: Dataset, measurement_basis: str, readout_key: str, proj_type: str, n_qubits: int, scaler_type: str, use_reuploading: bool):
    """Derive DB filename and study name from the variant combination."""
    reupload_str = "reupTrue" if use_reuploading else "reupFalse"
    dataset_str = dataset_enum.value
    study_name = f"qrc_{dataset_str}_vpt_{scaler_type}0_{proj_type}_q{n_qubits}_{measurement_basis}_{readout_key}_{reupload_str}_mean_vpt_fb<3.5"
    db_name = f"optuna_qrc_{proj_type}_mean_vpt.db"          # one DB per projection type
    return study_name, db_name


def main():
    parser = argparse.ArgumentParser(description="Optuna QRC Hyperparameter Search (Multi-Seed Mean VPT)")
    parser.add_argument("--trials", type=int, default=500,
                        help="Number of optimization trials (default: 500)")
    parser.add_argument("--dataset", type=str, default="lorenz",
                        choices=["mackey_glass", "lorenz", "lorenz96"],
                        help="Dataset to optimize on (default: lorenz)")
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
    parser.add_argument("--enqueue-csv", type=str, default=None,
                        help="Path to a CSV file to enqueue trials from (e.g. filtered_optuna_results.csv)")
    args = parser.parse_args()

    try:
        check_gpu_available()
    except RuntimeError as exc:
        raise ValueError(f"Warning: GPU check failed ({exc}). Continuing...")

    # --- Dataset ---
    dataset_name = args.dataset
    if dataset_name == "mackey_glass":
        dataset_enum = Dataset.MACKEY_GLASS
    elif dataset_name == "lorenz":
        dataset_enum = Dataset.LORENZ
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # --- Resolve variant from args or preset defaults ---
    base = TIME_QUANTUM_RESERVOIR_PRESET

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
    proj_type_name = type(base.projection).__name__
    proj_tag = proj_type_name.lower().replace("config", "")

    # Updated scaler tag for BoundedAffineScaler
    scaler_tag = "bounded_affine"

    study_name, db_name = derive_names(dataset_enum, measurement_basis, readout_key, proj_tag, n_qubits, scaler_tag, use_reuploading)

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

    print("=" * 60)
    print("Optuna QRC Hyperparameter Search (Multi-Seed Mean VPT)")
    print("=" * 60)
    print(f"  Study            : {study_name}")
    print(f"  Storage          : {storage}")
    print(f"  Dataset          : {dataset_name}")
    print(f"  Trials           : {args.trials}")
    print(f"  Measurement Basis: {measurement_basis}")
    print(f"  Readout          : {readout_key}")
    print(f"  Re-uploading     : {use_reuploading}")
    print("=" * 60)

    # --- Enqueue from CSV if provided ---
    if args.enqueue_csv:
        import csv
        csv_path = Path(args.enqueue_csv)
        if csv_path.exists():
            print(f"\\n[+] Enqueueing trials from {csv_path}...")
            with open(csv_path, encoding='utf-8') as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    params = {}
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
                        # If feature_min was fixed and not in the CSV, use the known fixed value for Lorenz/MG
                        f_min = params.pop('feature_min', 0)
                        bound = np.pi
                        
                        # Mathematical conversion from MinMax [-1, 1] output bounds to BoundedAffine space
                        # BoundedAffine formula: output = X_norm * (scale * bound) + relative_shift * bound * (1 - scale)
                        # So max_out = scale * bound + relative_shift * bound * (1 - scale) = f_max
                        #    min_out = -scale * bound + relative_shift * bound * (1 - scale) = f_min
                        scale = (f_max - f_min) / (2 * bound)
                        if scale < 1.0:
                            relative_shift = (f_max + f_min) / (2 * bound * (1.0 - scale))
                        else:
                            relative_shift = 0.0
                        
                        # Ensure values are within Optuna search space bounds to avoid enqueue errors
                        params['scale'] = max(0.001, min(1.0, scale))
                        params['relative_shift'] = max(-1.0, min(1.0, relative_shift))
                        
                    # If n_layers is missing from the old CSV, default it to 1
                    if 'n_layers' not in params:
                        params['n_layers'] = 1
                    else:
                        params['n_layers'] = int(params['n_layers'])
                        
                    if params:
                        study.enqueue_trial(params, skip_if_exists=True)
                        count += 1
            print(f"[+] Successfully enqueued {count} trials.\\n")
        else:
            print(f"\\n[!] Warning: CSV file not found at {csv_path}\\n")

    # --- Run ---
    objective_fn = make_objective(measurement_basis, readout_config, use_reuploading, dataset_enum)
    study.optimize(objective_fn, n_trials=args.trials)
    # --- Report ---
    print("\n" + "=" * 60)
    print("=== BEST PARAMETERS ===")
    print("=" * 60)
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"  {'Best Mean VPT':20s}: {study.best_value:.2f} LT")
    print("=" * 60)


if __name__ == "__main__":
    main()