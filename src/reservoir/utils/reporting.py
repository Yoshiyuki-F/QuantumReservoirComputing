"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/utils/reporting.py
Reporting utilities for post-run analysis: metrics, logging, and file outputs Draw and save no recalculation.
"""
from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING
from reservoir.utils.metrics import vpt_score

if TYPE_CHECKING:
    from reservoir.core.types import NpF64, ResultDict, TrainLogs, EvalMetrics, FitResultDict, TestMetrics, TopologyMeta, TrainMetrics
    from reservoir.models.generative import ClosedLoopModel
    from reservoir.models.config import PipelineConfig
    from reservoir.data.config import DatasetPreset
    from reservoir.training.config import TrainingConfig
    from reservoir.readout.base import ReadoutModule
    from reservoir.layers.preprocessing import Preprocessor
    from reservoir.pipelines.config import FrontendContext, DatasetMetadata
    from collections.abc import Sequence

# --- Array Formatting Helpers ---
def _format_cls_array(arr: NpF64 | None) -> NpF64 | None:
    if arr is None:
        return None
    res = arr
    if res.ndim > 1 and res.shape[-1] > 1:
        res = np.argmax(res, axis=-1)
    return res.ravel()

def _calc_acc(y_true: NpF64 | None, y_pred: NpF64 | None) -> float:
    if y_true is None or y_pred is None:
        return 0.0
    y_t = y_true.ravel()
    y_p = y_pred.ravel()
    if len(y_t) == 0:
        return 0.0
    return float(np.mean(y_t == y_p))

def _get_seq_len(arr: NpF64 | None) -> int:
    if arr is None:
        return 0
    if arr.ndim == 3:
        return int(arr.shape[1])
    if arr.ndim >= 1:
        return int(arr.shape[0])
    return 0

def _to_2d(arr: NpF64) -> NpF64:
    if arr.ndim == 3:
        return arr.reshape(-1, arr.shape[-1])
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

# --- Metrics Calculation ---



def print_chaos_metrics(metrics: EvalMetrics, header: str | None = None) -> None:
    """
    Print chaos metrics to console.
    """
    if header:
        print(f"{header}")
    else:
        print("=== Chaos Prediction Metrics ===")
    
    # Direct access from strictly typed EvalMetrics
    # Optional fields use 0.0 or inf as defaults to prevent crashes if not present
    print(f"MSE       : {metrics.get('mse', 0.0):}")
    print(f"NMSE      : {metrics.get('nmse', float('inf')):}")
    print(f"NRMSE     : {metrics.get('nrmse', float('inf')):}")
    print(f"MASE      : {metrics.get('mase', float('inf')):}")
    print(f"NDEI      : {metrics.get('ndei', float('inf')):} (Target < 0.1)")
    print(f"Var Ratio : {metrics.get('var_ratio', 0.0):} (Target ~ 1.0)")
    print(f"Corr      : {metrics.get('correlation', 0.0):} (Target > 0.95)")
    
    vpt_steps = int(metrics.get("vpt_steps", 0.0))
    vpt_lt = float(metrics.get("vpt_lt", 0.0))
    vpt_threshold = float(metrics.get("vpt_threshold", 0.4))
    print(f"VPT       : {vpt_steps} steps ({vpt_lt} LT) @ threshold={vpt_threshold}")




# --- Logging / Printing ---

def print_feature_stats(features: NpF64, file:str, stage: str, backend: str = "numpy") -> None:
    """Internal implementation handling concrete numpy arrays."""
    # 基本統計量
    stats = {
        "shape": features.shape,
        "dtype": f"{backend}.{features.dtype.name}",
        "mean": float(np.mean(features)),
        "std": float(np.std(features)),
        "min": float(np.min(features)),
        "max": float(np.max(features)),
        "nans": int(np.isnan(features).sum()),
    }

    print(
        f"[{file} FeatureStats:{stage}] dtype={stats['dtype']}, shape={stats['shape']}, "
        f"mean={stats['mean']:.4f}, std={stats['std']:.4f}, "
        f"min={stats['min']:.4f}, max={stats['max']:.4f}, nans={stats['nans']}"
    )
    if stats["std"] < 1e-6:
        print("Feature matrix has near-zero variance. Model output may be inactive.")


def print_ridge_search_results(train_res: FitResultDict, metric_name: str = "MSE") -> None:
    history_raw = train_res.get("search_history")
    if history_raw is None or len(history_raw) == 0:
        return
    history: dict[float, float] = history_raw
    best_lam = train_res.get("best_lambda")
    
    weight_norms_raw = train_res.get("weight_norms")
    weight_norms: dict[float, float]
    if weight_norms_raw is None:
        weight_norms = {}
    else:
        weight_norms = weight_norms_raw
    
    metric_label = metric_name

    # Decide best logic for marking
    # Both minimize score internally (MSE is min, -VPT is min)
    best_by_metric = min(history.items(), key=lambda item: float(item[1]))[0]

    best_marker = best_lam if best_lam is not None else best_by_metric

    print("\n" + "=" * 40)
    print(f"Ridge Hyperparameter Search ({metric_label})")
    print("-" * 40)
    sorted_lambdas = sorted(history.keys())
    for lam in sorted_lambdas:
        if lam is None:
            continue
        try:
             score = float(history[lam])
             lam_val = float(lam)
        except (ValueError, TypeError):
             continue
        
        # Format score for display
        score_disp = score
        label = f"Val {metric_name}"
        
        # Legacy VPT handling: if metric is exactly "VPT", we assume stored as negative
        if metric_name == "VPT":
            score_disp = -score
            label = "Val VPT"
            
        norm = weight_norms.get(lam)
        norm_str = f"(Norm: {norm})" if norm is not None else "(Norm: n/a)"
        marker = ""
        if best_marker is not None:
             try:
                 bm_val = float(best_marker)
                 if abs(lam_val - bm_val) < 1e-12:
                     marker = " <= best"
             except (ValueError, TypeError):
                 pass
        print(f"   λ = {lam_val:.2e} : {label} = {score_disp:.6e} {norm_str}{marker}")
    print("=" * 40 + "\n")


def plot_distillation_loss(training_logs: TrainLogs, save_path: str, title: str, learning_rate: float | None = None) -> None:
    loss_history = training_logs.get("loss_history")
    if not loss_history:
        return
    try:
        from reservoir.utils.plotting import plot_loss_history
    except ImportError as exc:  # pragma: no cover
        print(f"Skipping distillation loss plotting due to import error: {exc}")
        return
    loss_list = list(loss_history)
    plot_loss_history(loss_list, save_path, title=title, learning_rate=learning_rate)


def plot_classification_report(
    train_y: NpF64 | None,
    test_y: NpF64 | None,
    val_y: NpF64 | None,
    filename: str,
    model_type_str: str,
    dataset_meta: DatasetMetadata,
    results: ResultDict,
    training_obj: TrainingConfig,
    train_pred: NpF64 | None = None,
    test_pred: NpF64 | None = None,
    val_pred: NpF64 | None = None,
    selected_lambda: float | None = None,
    class_names: Sequence[str] | None = None,
) -> None:
    try:
        from reservoir.utils.plotting import plot_classification_results
    except ImportError as exc:  # pragma: no cover
        print(f"Skipping plotting due to import error: {exc}")
        return

    int(getattr(training_obj, "batch_size", 0) or 0)

    # ---------------------------------------------------------
    # 1. Labels Preparation
    # ---------------------------------------------------------
    train_labels_np = _format_cls_array(train_y)
    test_labels_np = _format_cls_array(test_y)
    val_labels_np = _format_cls_array(val_y)

    # ---------------------------------------------------------
    # 2. Predictions Preparation
    # ---------------------------------------------------------
    train_pred_np = _format_cls_array(train_pred)
    test_pred_np = _format_cls_array(test_pred)
    val_pred_np = _format_cls_array(val_pred)

    # ---------------------------------------------------------
    # 3. Plot
    # ---------------------------------------------------------
    acc_train = None
    acc_test = None
    acc_val = None

    if results is not None:
         train_res = results.get("train", {})
         val_res = results.get("validation", {})
         test_res = results.get("test", {})
         
         acc_train = float(train_res.get("accuracy", 0.0))
         acc_val = float(val_res.get("accuracy", 0.0))
         acc_test = float(test_res.get("accuracy", 0.0))

    if acc_train == 0.0:
        acc_train = _calc_acc(train_labels_np, train_pred_np)
    
    if acc_test == 0.0:
        acc_test = _calc_acc(test_labels_np, test_pred_np)
    
    if acc_val == 0.0:
        acc_val = _calc_acc(val_labels_np, val_pred_np) if val_labels_np is not None else 0.0
    
    print("\n[Report] Accuracy Check (Pre-Plot):")
    print(f"  Train: {acc_train:.4%}")
    print(f"  Val  : {acc_val:.4%}")
    print(f"  Test : {acc_test:.4%}")

    # Extract lambda_norm from weight_norms for the selected lambda
    lambda_norm = None
    if selected_lambda is not None and results is not None:
        train_res = results.get("train", {})
        weight_norms = train_res.get("weight_norms", {})
        lambda_norm = float(weight_norms.get(selected_lambda, 0.0))

    empty_test_metrics: TestMetrics = {}
    metrics_test = results.get("test", empty_test_metrics) if results is not None else empty_test_metrics
    metrics_payload: dict[str, str] = {str(k): str(v) for k, v in metrics_test.items()}
    if train_labels_np is None or test_labels_np is None or train_pred_np is None or test_pred_np is None:
        print("    [Reporter] Missing data for classification plot, skipping.")
        return

    plot_classification_results(
        train_labels=train_labels_np,
        test_labels=test_labels_np,
        train_predictions=train_pred_np,
        test_predictions=test_pred_np,
        val_labels=val_labels_np,
        val_predictions=val_pred_np,
        title=f"{model_type_str.upper()} on {dataset_meta.dataset_name}",
        filename=filename,
        metrics_info=metrics_payload,
        best_lambda=selected_lambda,
        lambda_norm=lambda_norm,
        class_names=class_names,
    )


def get_preprocess_label(topo_meta: TopologyMeta, config: PipelineConfig | None) -> str:
    # Use config.preprocess.label (single source of truth)
    if config is not None and hasattr(config, "preprocess") and hasattr(config.preprocess, "label"):
        return config.preprocess.label

    # Fallback for legacy topo_meta
    details = topo_meta.get("details")
    if details is None:
        return "raw"
    raw_label = str(details.get("preprocess", ""))
    return raw_label if raw_label else "raw"


def get_projection_label(config: PipelineConfig, topo_meta: TopologyMeta) -> str | None:
    _ = topo_meta
    if not hasattr(config, 'projection') or config.projection is None:
        return None

    # Use config.projection.label (single source of truth)
    if hasattr(config.projection, "label"):
        return config.projection.label

    # Fallback
    return type(config.projection).__name__.replace("Config", "")


def infer_filename_parts(topo_meta: TopologyMeta, training_obj: TrainingConfig, model_type_str: str, readout: ReadoutModule | None = None, config: PipelineConfig | None = None) -> list[str]:
    # 1. Model Type & Parameters
    model_str = str(model_type_str)
    
    if config is not None:
        # Some configs might be nested or we just need the underlying model config
        curr_cfg = config
        while hasattr(curr_cfg, 'model') and curr_cfg.model is not None:
            curr_cfg = curr_cfg.model
            
        if hasattr(curr_cfg, 'label'):
            model_str = curr_cfg.label
            
        # If n_qubits was omitted in QRC config, pull it from projection
        if "q?" in model_str and hasattr(config, "projection") and config.projection:
            n_qubits = getattr(config.projection, "n_units", None)
            if n_qubits is not None:
                model_str = model_str.replace("q?", f"q{n_qubits}")

    filename_parts: list[str] = [model_str]

    # 2. Preprocessing
    preprocess_label = get_preprocess_label(topo_meta, config)
    filename_parts.append(preprocess_label)

    # 3. Projection
    if config is not None:
        proj_lbl = get_projection_label(config, topo_meta)
        if proj_lbl:
            filename_parts.append(proj_lbl)

    # 4. Readout
    if config is not None and config.readout is not None:
        readout_config = config.readout
        readout_lbl = getattr(readout_config, "label", readout_config.__class__.__name__)
        readout_hidden_layers = getattr(readout_config, "hidden_layers", None)
        if readout_hidden_layers is not None:
            lr = float(getattr(training_obj, 'learning_rate', 0.0)) if training_obj else 0.0
            if lr > 0:
                filename_parts.append(f"{readout_lbl}_LR{lr:.0e}")
            else:
                filename_parts.append(readout_lbl)
        else:
            filename_parts.append(readout_lbl)
    elif readout is not None:
        readout_type = readout.__class__.__name__
        hidden_layers = readout.hidden_layers
        if hidden_layers:
            layers_str = "-".join(str(int(v)) for v in hidden_layers)
            lr = float(getattr(training_obj, 'learning_rate', 0.0)) if training_obj else 0.0
            if lr > 0:
                filename_parts.append(f"{readout_type}{layers_str}_LR{lr:.0e}")
            else:
                filename_parts.append(f"{readout_type}{layers_str}")
        else:
            filename_parts.append(f"{readout_type}RO")

    # 5. NN Epochs marker
    type_lower = str(model_type_str).lower()
    topo_type = str(topo_meta.get("type", "")).lower()
    is_fnn = "fnn" in type_lower or "fnn" in topo_type or "rnn" in topo_type or "nn" in topo_type
    
    if is_fnn and training_obj:
        filename_parts.append(f"epochs{int(getattr(training_obj, 'epochs', 0) or 0)}")
        
    return filename_parts


def generate_report(
    results: ResultDict,
    config: PipelineConfig,
    topo_meta: TopologyMeta,
    readout: ReadoutModule | None,
    train_y: NpF64 | None,
    test_y: NpF64 | None,
    val_y: NpF64 | None,
    training_obj: TrainingConfig,
    dataset_meta: DatasetMetadata,
    model_type_str: str,
    classification: bool = False,
    # preprocessors removed
    dataset_preset: DatasetPreset | None = None,  # DatasetPreset for dt/lyapunov_time_unit
    model_obj: ClosedLoopModel | None = None, # New Argument
) -> None:
    """
    Coordinator for generating all report elements (plots, logs).
    Delegates specific plotting tasks to specialized functions.
    """
    # 1. Common: Distillation Loss (if available)
    _plot_distillation_section(results, topo_meta, training_obj, model_type_str, readout, config, dataset_meta)

    # 2. Main Task Plots (Classification vs Regression using MSE)
    if classification:
        _plot_classification_section(
            results, config, topo_meta, training_obj, dataset_meta, model_type_str, readout,
            train_y, test_y, val_y
        )
    else:
        _plot_regression_section(
            results, config, topo_meta, training_obj, dataset_meta, model_type_str, readout,
            train_y, val_y, test_y, dataset_preset
        )

    # 3. Quantum Dynamics (if available)
    _plot_quantum_section(results, topo_meta, training_obj, dataset_meta, model_type_str, readout, config, model_obj)


def _plot_distillation_section(results: ResultDict, topo_meta: TopologyMeta, training_obj: TrainingConfig, model_type_str: str, readout: ReadoutModule | None, config: PipelineConfig, dataset_meta: DatasetMetadata) -> None:
    training_logs = results.get("training_logs")
    if training_logs is not None:
        dataset_name = dataset_meta.dataset_name
        filename_parts = infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
        loss_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_loss.png"
        lr = float(getattr(training_obj, 'learning_rate', 0.0))
        plot_distillation_loss(training_logs, loss_filename, title=f"{model_type_str.upper()} Distillation Loss", learning_rate=lr if lr > 0 else None)


def _plot_classification_section(
    results: ResultDict, config: PipelineConfig, topo_meta: TopologyMeta, training_obj: TrainingConfig, dataset_meta: DatasetMetadata, model_type_str: str, readout: ReadoutModule | None,
    train_y: NpF64 | None, test_y: NpF64 | None, val_y: NpF64 | None
) -> None:
    dataset_name = dataset_meta.dataset_name
    filename_parts = infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
    confusion_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_confusion.png"
    
    train_res_raw = results.get("train")
    train_res: TrainMetrics
    if train_res_raw is None:
        train_res = {}
    else:
        train_res = train_res_raw
    selected_lambda = None
    lam_val = train_res.get("best_lambda")
    if lam_val is not None:
        selected_lambda = float(str(lam_val))

    # Extract predictions from ResultDict and ensure Host Domain (NpF64)
    outputs_raw = results.get("outputs")
    outputs: dict[str, NpF64 | None]
    if outputs_raw is None:
        outputs = {}
    else:
        outputs = outputs_raw
    train_pred_raw = outputs.get("train_pred")
    test_pred_raw = outputs.get("test_pred")
    val_pred_raw = outputs.get("val_pred")
    
    train_p = train_pred_raw
    test_p = test_pred_raw
    val_p = val_pred_raw

    plot_classification_report(
        train_y=train_y,
        test_y=test_y,
        val_y=val_y,
        filename=confusion_filename,
        model_type_str=model_type_str,
        dataset_meta=dataset_meta,
        # metric removed
        selected_lambda=selected_lambda,
        results=results,
        training_obj=training_obj,
        train_pred=train_p,
        test_pred=test_p,
        val_pred=val_p,
    )
    
    # FNN Readout Loss Plot
    if readout is not None and hasattr(readout, 'training_logs') and readout.training_logs:
        fnn_loss_history = readout.training_logs.get("loss_history")
        if fnn_loss_history:
            loss_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_loss.png"
            lr = float(getattr(training_obj, 'learning_rate', 0.0))
            plot_distillation_loss(readout.training_logs, loss_filename, title=f"{model_type_str.upper()} FNN Readout Loss", learning_rate=lr if lr > 0 else None)


def _plot_regression_section(
    results: ResultDict, config: PipelineConfig, topo_meta: TopologyMeta, training_obj: TrainingConfig, dataset_meta: DatasetMetadata, model_type_str: str, readout: ReadoutModule | None,
    train_y: NpF64 | None, val_y: NpF64 | None, test_y: NpF64 | None, dataset_preset: DatasetPreset | None
) -> None:
    dataset_name = dataset_meta.dataset_name
    filename_parts = infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
    prediction_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_prediction.png"
    
    test_results = results.get("test")
    test_mse = 0.0
    if test_results is not None and test_results.get("mse") is not None:
        test_mse = float(test_results["mse"])
        
    scaler = results.get("scaler")
    is_closed_loop = bool(results.get("is_closed_loop", False))

    # Extract predictions from ResultDict and ensure Host Domain (NpF64)
    outputs_raw = results.get("outputs")
    outputs: dict[str, NpF64 | None]
    if outputs_raw is None:
        outputs = {}
    else:
        outputs = outputs_raw
    test_pred_raw = outputs.get("test_pred")

    test_p = test_pred_raw

    # Get dt and lyapunov_time_unit for VPT calculation
    dt = None
    ltu = None
    if dataset_preset is not None:
        ds_config = getattr(dataset_preset, 'config', None)
        if ds_config is not None:
            dt = float(getattr(ds_config, 'dt', 1.0))
            ltu = float(getattr(ds_config, 'lyapunov_time_unit', 1.0))

    if readout is None:
        print("    [Reporter] Missing Readout module for regression plot. Skipping.")
        return

    plot_regression_report(
            train_y=train_y,
            val_y=val_y,
            test_y=test_y,
            filename=prediction_filename,
            model_type_str=model_type_str,
            mse=test_mse if test_mse > 0 else None,
            test_pred=test_p,
            scaler=scaler,
            is_closed_loop=is_closed_loop,
            dt=dt,
            lyapunov_time_unit=ltu,
        )

    # Note: Lambda Search BoxPlot is now handled in strategies.py (Step 7.5)


def plot_ridgecv_intermediates(
    residuals_hist: dict[float, NpF64] | None,
    weight_norms: dict[float, float] | None,
    best_lambda: float | None,
    best_score: float,
    val_y: NpF64 | None,
    val_pred_np: NpF64 | None,
    frontend_ctx: FrontendContext,
    topo_meta: TopologyMeta,
    pipeline_config: PipelineConfig,
    dataset_meta: DatasetMetadata,
    model_type_str: str,
    readout: ReadoutModule | None,
    metric_name: str = "NMSE",
) -> None:
    """Standardized intermediate plotting for Step 7 (Validation phase)."""
    try:
        from reservoir.utils.plotting import plot_lambda_search_boxplot, plot_timeseries_comparison
        
        training_obj = dataset_meta.training
        dataset_name = dataset_meta.dataset_name
        filename_parts = infer_filename_parts(topo_meta, training_obj, model_type_str, readout, pipeline_config)
        
        # 1. Lambda Search BoxPlot
        if residuals_hist:
            boxplot_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_lambda_boxplot.png"
            plot_lambda_search_boxplot(
                residuals_hist, boxplot_filename,
                title=f"Step 7: Lambda Search Residuals ({model_type_str})",
                best_lambda=best_lambda,
                metric_name=metric_name,
                weight_norms=weight_norms
            )

        # 2. Validation Prediction Plot (Open-Loop) - ONLY for Regression
        if val_pred_np is not None and val_y is not None and metric_name.lower() != "accuracy":
            val_plot_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_val_prediction.png"
            
            # Unscale for plotting if possible
            scaler = frontend_ctx.preprocessor
            def _inv(arr: NpF64) -> NpF64:
                if scaler is None:
                    return arr
                try:
                    v = arr.reshape(-1, 1) if arr.ndim == 1 else arr
                    return scaler.inverse_transform(v).reshape(arr.shape)
                except (ValueError, TypeError):
                    return arr

            val_y_raw = _inv(val_y)
            val_p_raw = _inv(val_pred_np)
            
            best_norm = 0.0
            if weight_norms is not None and best_lambda is not None:
                best_norm = float(weight_norms.get(best_lambda, 0.0))
            
            plot_timeseries_comparison(
                targets=val_y_raw,
                predictions=val_p_raw,
                filename=val_plot_filename,
                title=f"Step 7: Val Open-Loop ({metric_name}: {best_score:.2e}, ||w||: {best_norm:.2e})"
            )
    except (ImportError, RuntimeError, ValueError, OSError, TypeError, AttributeError) as e:
        print(f"    [Warning] Intermediate plotting failed in reporting.py: {e}")


def _plot_quantum_section(results: ResultDict, topo_meta: TopologyMeta, training_obj: TrainingConfig, dataset_meta: DatasetMetadata, model_type_str: str, readout: ReadoutModule | None, config: PipelineConfig, model_obj: ClosedLoopModel | None) -> None:
    quantum_trace = results.get("quantum_trace")
    if quantum_trace is not None:
        try:
            from reservoir.utils.quantum_plotting import plot_qubit_dynamics

            dataset_name = dataset_meta.dataset_name
            filename_parts = infer_filename_parts(topo_meta, training_obj, model_type_str, readout, config)
            dynamics_filename = f"outputs/{dataset_name}/{'_'.join(filename_parts)}_quantum_dynamics.png"

            trace_np = quantum_trace
            feature_names = None
            if model_obj is not None and hasattr(model_obj, "get_observable_names"):
                    feature_names = model_obj.get_observable_names()
            elif hasattr(training_obj, "get_observable_names"):
                    # Fallback but unlikely
                    feature_names = training_obj.get_observable_names()
            
            if trace_np is not None:
                plot_qubit_dynamics(trace_np, dynamics_filename, title=f"{model_type_str.upper()} Dynamics ({dataset_name})", feature_names=feature_names)

        except ImportError:
            pass # Skipping quantum plotting (ImportError)
        except (RuntimeError, ValueError) as e:
            print(f"Skipping quantum plotting (Error: {e})")


def plot_regression_report(
    *,
    train_y: NpF64 | None,
    val_y: NpF64 | None = None, # New Argument
    test_y: NpF64 | None,
    filename: str,
    model_type_str: str,
    mse: float | None = None,
    test_pred: NpF64 | None = None,
    scaler: Preprocessor | None = None,
    is_closed_loop: bool = False,
    dt: float | None = None,
    lyapunov_time_unit: float | None = None,
    vpt_threshold: float = 0.4,
) -> None:
    try:
        from reservoir.utils.plotting import plot_timeseries_comparison
    except ImportError as exc:  # pragma: no cover
        print(f"Skipping plotting due to import error: {exc}")
        return

    # Generate Test Predictions
    test_pred_final = test_pred

    # Infer global time offset
    # Offset = Length(Train) + Length(Val)
    offset = _get_seq_len(train_y) + _get_seq_len(val_y)

    # Align lengths if predictions are shorter (e.g. TimeDelayEmbedding)
    if test_y is not None and test_pred_final is not None:
        len_t = _get_seq_len(test_y)
        len_p = _get_seq_len(test_pred_final)
        
        if len_p < len_t:
             diff = len_t - len_p
             # print(f"  [Report] Aligning plot targets: slicing first {diff} steps.")
             test_y_np = test_y
             if test_y_np.ndim == 3:
                 test_y = test_y_np[:, diff:, :]
             else:
                 test_y = test_y_np[diff:]

    # Prepare for plotting (Inverse Transform to Raw Domain)
    if test_pred_final is not None:
        test_pred_plot = _to_2d(test_pred_final)
    else:
        test_pred_plot = None
        
    test_y_plot = _to_2d(test_y) if test_y is not None else None

    if scaler is not None:
        try:
            if test_pred_plot is not None:
                test_pred_plot = scaler.inverse_transform(test_pred_plot)
            if test_y_plot is not None:
                test_y_plot = scaler.inverse_transform(test_y_plot)
        except (ValueError, TypeError) as e:
            print(f"  [Report] Scaler inverse transform failed: {e}")

    # Update variables for plotting
    test_pred = test_pred_plot
    test_y = test_y_plot

    title_str = f"Test Predictions ({model_type_str})"
    if is_closed_loop:
        title_str = f"{title_str} closed-loop"
    
    # Calculate VPT using shared metric function
    vpt_lt = None
    if dt is not None and lyapunov_time_unit is not None and test_y is not None and test_pred is not None:
        # vpt_score handles multivariate logic correctly (Euclidean norm).
        # It expects (Time, Features), which we have after inverse transform.
        vpt_steps = vpt_score(test_y, test_pred, threshold=vpt_threshold)
        steps_per_lt = int(lyapunov_time_unit / dt) if dt > 0 else 1
        vpt_lt = float(vpt_steps) / steps_per_lt if steps_per_lt > 0 else 0.0

    # Display VPT if calculated, otherwise fallback to MSE
    if vpt_lt is not None:
        title_str += f" | VPT: {vpt_lt:.2f} LT"
    elif mse is not None:
        title_str += f" | MSE: {mse:.4f} (Scaled)"

    if test_y is not None and test_pred is not None:
        plot_timeseries_comparison(
            targets=test_y,
            predictions=test_pred,
            filename=filename,
            title=title_str,
            time_offset=offset,
        )
