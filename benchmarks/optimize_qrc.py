"""
Optuna Hyperparameter Search for Quantum Reservoir Computing.
Usage:
uv run python benchmarks/optimize_qrc.py --dataset lorenz
uv run python benchmarks/optimize_qrc.py --trials 100
Visualization:
uv run optuna-dashboard  sqlite:////home/yoshi/PycharmProjects/Reservoir/benchmarks/optuna_qrc_nonetype.db
"""
from __future__ import annotations

import os
# Force 64-bit precision before ANY other imports
os.environ["JAX_ENABLE_X64"] = "True"

import argparse
import dataclasses
import math
from pathlib import Path
from typing import TYPE_CHECKING

import jax
jax.config.update("jax_enable_x64", True)

import numpy as np  # noqa: E402
import optuna  # noqa: E402

from reservoir.utils import check_gpu_available  # noqa: E402
from reservoir.pipelines import run_pipeline  # noqa: E402
from reservoir.pipelines.strategies import DivergenceError  # noqa: E402
from reservoir.models.presets import (  # noqa: E402
    TIME_QUANTUM_RESERVOIR_PRESET,
    DEFAULT_RIDGE_READOUT,
)
from reservoir.models.config import (  # noqa: E402
    ReadoutConfig,
    PolyRidgeReadoutConfig,
    QuantumReservoirConfig,
)
from reservoir.data.identifiers import Dataset  # noqa: E402

if TYPE_CHECKING:
    from typing import Literal

    from reservoir.core.types import ResultDict, TestMetrics, TrainMetrics


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


def _parse_measurement_basis(value: str) -> Literal["Z", "ZZ", "Z+ZZ"]:
    if value == "Z":
        return "Z"
    if value == "ZZ":
        return "ZZ"
    if value == "Z+ZZ":
        return "Z+ZZ"
    raise ValueError(f"Unknown measurement basis: {value}")


def _require_n_qubits(n_qubits: int | None, preset_name: str) -> int:
    if n_qubits is None:
        raise ValueError(f"{preset_name}.model.n_qubits must be set for benchmark naming")
    return n_qubits


def build_config(
        feature_min: float,
        feature_max: float,
        n_layers: int,
        feedback_scale: float,
        leak_rate: float,
        use_reuploading: bool,
        *,
        measurement_basis: Literal["Z", "ZZ", "Z+ZZ"],
        readout_config: ReadoutConfig | None,
):
    """
    Build a PipelineConfig with dynamically updated parameters.
    
    feature_min/max is applied via MinMaxScalerConfig (preprocessing).
    projection is None (Step 3 skipped).
    """
    from reservoir.models.config import MinMaxScalerConfig
    base = TIME_QUANTUM_RESERVOIR_PRESET
    base_model = base.model
    if not isinstance(base_model, QuantumReservoirConfig):
        raise TypeError("TIME_QUANTUM_RESERVOIR_PRESET.model must be QuantumReservoirConfig")

    # Update preprocessing (MinMaxScaler)
    new_preprocess = MinMaxScalerConfig(
        feature_min=feature_min, 
        feature_max=feature_max,
    )

    # Update model (feedback_scale, leak_rate, measurement_basis)
    new_model = dataclasses.replace(
        base_model,
        n_layers=n_layers,
        feedback_scale=feedback_scale,
        leak_rate=leak_rate,
        measurement_basis=measurement_basis,
        use_reuploading=use_reuploading,
    )

    # Construct final config (projection=None, no Step 3)
    return dataclasses.replace(
        base,
        preprocess=new_preprocess,
        model=new_model,
        readout=readout_config,
    )


def make_objective(
        measurement_basis: Literal["Z", "ZZ", "Z+ZZ"],
        readout_config: ReadoutConfig | None,
        use_reuploading: bool,
        dataset_enum: Dataset,
):
    """Factory that returns an Optuna objective closed over the study variant."""

    def objective(trial: optuna.Trial) -> float:
        """
        Optuna objective function.
        Searches for optimal QRC parameters to maximize VPT.
        """
        # === 1. Suggest Parameters ===

        # ======================== Preprocessing (MinMax) ====================
        # feature_min = trial.suggest_float("feature_min", -np.pi, 0.0)
        feature_max = trial.suggest_float("feature_max", 0.0, 1.0)
        feature_min = trial.suggest_float("feature_min",  -9.735743764947846e-05,  -9.735743764947846e-05) #MG-T
        # feature_max = trial.suggest_float("feature_max", 0.0, 0.15)
        # feature_max = trial.suggest_float("feature_max", 0.04387396511208059, 0.04387396511208059)



        # feature_max = trial.suggest_float("delta", 0.6516478039657447, 0.6516478039657447) #MG-F
        # ======================== Reservoir ==================================
        n_layers = trial.suggest_int("n_layers", 1, 1)
        # n_layers = trial.suggest_int("n_layers", 7, 7) #MG-T
        # n_layers = trial.suggest_int("n_layers", 5, 5) #MG-F
        #
        feedback_scale = trial.suggest_float("feedback_scale", 0, 3.5) #theoretically can be just till np.pi, but we see divergence beyond 3.5 in practice
        # feedback_scale = trial.suggest_float("feedback_scale", 0.0, 2.0) #MG-T
        # feedback_scale = trial.suggest_float("feedback_scale", np.pi, np.pi) #MG-T

        # feedback_scale = trial.suggest_float("feedback_scale", 2.4272583655991578, 2.4272583655991578) #MG-F
        #
        leak_rate = trial.suggest_float("leak_rate", 0, 1)
        # leak_rate = trial.suggest_float("leak_rate",  0.11967302052818608,  0.11967302052818608) #MG-T
        # leak_rate = trial.suggest_float("leak_rate",  0.23260873230871418,  0.23260873230871418) #MG-F

        # === 2. Build Config ===
        config = build_config(
            feature_min,
            feature_max,
            n_layers,
            feedback_scale,
            leak_rate,
            use_reuploading,
            measurement_basis=measurement_basis,
            readout_config=readout_config,
        )

        # === 3. Run Pipeline ===
        try:
            results: ResultDict = run_pipeline(config, dataset_enum)

            # === 4. Extract Metrics ===
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

            # Store ALL chaos metrics (including new stats from strategies.py)
            success_stats: dict[str, float] = {}
            for key in ["mse", "nmse", "nrmse", "mase", "ndei",
                        "var_ratio", "correlation", "vpt_steps", "vpt_lt", "vpt_threshold",
                        "pred_mean", "pred_std", "pred_min", "pred_max",
                        "truth_mean", "truth_std", "truth_min", "truth_max"]:
                val = chaos.get(key, None)
                if val is not None:
                    trial.set_user_attr(key, float(val))
                    if "pred" in key or "truth" in key:
                        success_stats[key] = float(val)

            if success_stats:
                print(f"    [Success Stats] {success_stats}")

            # For backward compatibility / printing convenience, ensure these locals exist
            var_ratio = chaos.get("var_ratio", 0.0)

            if vpt_lt is None or math.isnan(vpt_lt) or vpt_lt <= 0:
                print(f"Trial {trial.number}: FAILED (VPT=0) "
                      f"(min={feature_min:.3f}, max={feature_max:.3f}, fb={feedback_scale:.3f})")
                trial.set_user_attr("status", "failed")
                return -1.0  # Return a negative value to indicate failure


            print(f"Trial {trial.number}: VPT={vpt_lt:.2f} LT, Var={var_ratio:.3f} "
                  f"(min={feature_min:.3f}, max={feature_max:.3f}, fb={feedback_scale:.3f})")

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
            return -0.2 # Specific negative value for divergence

        except (ValueError, RuntimeError) as e:
            # Fallback stats retrieval
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
                 return -0.5 # Return a negative value to indicate failure
            elif "validation nmse too high" in err_msg_lower:
                 try:
                     # Extract nmse from message: "Validation NMSE too high: 0.0002358972"
                     nmse_val = float(err_msg.split(":")[-1].strip())
                 except (ValueError, IndexError):
                     nmse_val = 1.0
                 print(f"Trial {trial.number}: FAILED (NMSE high) - {e}")
                 trial.set_user_attr("status", "failed_nmse")
                 trial.set_user_attr("nmse", nmse_val)
                 trial.set_user_attr("error", err_msg)
                 return -nmse_val
            else:
                 print(f"Trial {trial.number}: EXCEPTION (VPT=0) - {e}")
                 trial.set_user_attr("status", "exception")
                 trial.set_user_attr("error", err_msg)
                 return -1.0  # Return a negative value to indicate failure
        else:
            return vpt_lt

    return objective


def derive_names(
        dataset_enum: Dataset,
        measurement_basis: Literal["Z", "ZZ", "Z+ZZ"],
        readout_key: str,
        proj_type: str,
        n_qubits: int,
        scaler_type: str,
        use_reuploading: bool,
):
    """Derive DB filename and study name from the variant combination."""
    reupload_str = "reupTrue" if use_reuploading else "reupFalse"
    dataset_str = dataset_enum.value
    study_name = f"qrc_{dataset_str}_vpt_{scaler_type}0_{proj_type}_q{n_qubits}_{measurement_basis}_{readout_key}_{reupload_str}_kai7_fb<2"
    db_name = f"optuna_qrc_{proj_type}.db"          # one DB per projection type
    return study_name, db_name


def main():
    parser = argparse.ArgumentParser(description="Optuna QRC Hyperparameter Search")
    parser.add_argument("--trials", type=int, default=500,
                        help="Number of optimization trials (default: 500)")
    parser.add_argument("--dataset", type=str, default="mackey_glass",
                        choices=["mackey_glass", "lorenz", "lorenz96"],
                        help="Dataset to optimize on (default: mackey_glass)")
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
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # --- Resolve variant from args or preset defaults ---
    base = TIME_QUANTUM_RESERVOIR_PRESET
    base_model = base.model
    if not isinstance(base_model, QuantumReservoirConfig):
        raise TypeError("TIME_QUANTUM_RESERVOIR_PRESET.model must be QuantumReservoirConfig")

    # Measurement basis
    if args.measurement_basis is not None:
        measurement_basis = _parse_measurement_basis(args.measurement_basis)
    else:
        measurement_basis = base_model.measurement_basis
    
    # Qubits
    n_qubits = _require_n_qubits(base_model.n_qubits, "TIME_QUANTUM_RESERVOIR_PRESET")

    # Re-uploading
    use_reuploading = base_model.use_reuploading

    # Preprocessing
    # feature_min is tuned dynamically now

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

    # Updated scaler tag for MinMaxScaler
    scaler_tag = "minmax"

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
    print("Optuna QRC Hyperparameter Search")
    print("=" * 60)
    print(f"  Study            : {study_name}")
    print(f"  Storage          : {storage}")
    print(f"  Dataset          : {dataset_name}")
    print(f"  Trials           : {args.trials}")
    print(f"  Measurement Basis: {measurement_basis}")
    print(f"  Readout          : {readout_key}")
    print(f"  Re-uploading     : {use_reuploading}")
    print("=" * 60)

    # --- Run ---
    objective_fn = make_objective(measurement_basis, readout_config, use_reuploading, dataset_enum)
    study.optimize(objective_fn, n_trials=args.trials)
    # --- Report ---
    print("\n" + "=" * 60)
    print("=== BEST PARAMETERS ===")
    print("=" * 60)
    for k, v in study.best_params.items():
        print(f"  {k:20s}: {v:.4f}")
    print(f"  {'Best VPT':20s}: {study.best_value:.2f} LT")
    print("=" * 60)


if __name__ == "__main__":
    main()
