"""
src/reservoir/utils/metrics.py
Refactored metrics calculation logic.
Pure Numpy implementation for efficiency (avoids CPU->GPU transfer for metrics).
"""
from beartype import beartype
from reservoir.core.types import NpF64, EvalMetrics
import numpy as np

# --- Standard Metric Functions ---

@beartype
def mse(y_true: NpF64, y_pred: NpF64) -> float:
    """Mean Squared Error."""
    # Inputs are guaranteed to be NpF64 by beartype
    # Align if needed
    if y_pred.shape != y_true.shape and y_pred.size == y_true.size:
        y_pred = y_pred.reshape(y_true.shape)
    return float(np.mean((y_true - y_pred) ** 2))

@beartype
def rmse(y_true: NpF64, y_pred: NpF64) -> float:
    """Root Mean Squared Error."""
    return float(np.sqrt(mse(y_true, y_pred)))

@beartype
def nmse(y_true: NpF64, y_pred: NpF64) -> float:
    """
    Normalized Mean Squared Error (User Definition).
    NMSE = sum((y - y_hat)^2) / sum(y^2)
    Normalization by Second Moment (Energy).
    """
    if y_pred.shape != y_true.shape and y_pred.size == y_true.size:
        y_pred = y_pred.reshape(y_true.shape)

    numerator = float(np.sum((y_true - y_pred) ** 2))
    denominator = float(np.sum(y_true ** 2))
    
    return float(numerator / denominator) if denominator > 1e-9 else float('inf')

@beartype
def nrmse(y_true: NpF64, y_pred: NpF64) -> float:
    """
    Normalized Root Mean Squared Error.
    NRMSE = RMSE / std(y)
    """
    rmse_val = rmse(y_true, y_pred)
    std_true = float(np.std(y_true))
    return float(rmse_val / std_true) if std_true > 1e-9 else float('inf')

@beartype
def ndei(y_true: NpF64, y_pred: NpF64) -> float:
    """
    Non-Dimensional Error Index.
    Defined as RMSE / std(y). Same as NRMSE within this context.
    """
    return nrmse(y_true, y_pred)

@beartype
def mase(y_true: NpF64, y_pred: NpF64) -> float:
    """
    Mean Absolute Scaled Error.
    MAE / Mean Absolute Error of Naive Forecast (on true data).
    """
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Ensure same length
    min_len = min(len(y_true_flat), len(y_pred_flat))
    y_true_flat = y_true_flat[:min_len]
    y_pred_flat = y_pred_flat[:min_len]

    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    
    if len(y_true_flat) > 1:
        naive_mae = np.mean(np.abs(np.diff(y_true_flat)))
        return float(mae / naive_mae) if naive_mae > 1e-9 else float('inf')
    return float('inf')

@beartype
def var_ratio(y_true: NpF64, y_pred: NpF64) -> float:
    """Variance Ratio: std(pred) / std(true)."""
    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    return std_pred / std_true if std_true > 1e-9 else 0.0

@beartype
def correlation(y_true: NpF64, y_pred: NpF64) -> float:
    """Pearson Correlation Coefficient."""
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    min_len = min(len(y_true_flat), len(y_pred_flat))
    y_true_flat = y_true_flat[:min_len]
    y_pred_flat = y_pred_flat[:min_len]

    std_true = float(np.std(y_true_flat))
    std_pred = float(np.std(y_pred_flat))
    
    if std_true > 1e-9 and std_pred > 1e-9:
        matrix = np.corrcoef(y_true_flat, y_pred_flat)
        return float(matrix[0, 1])
    return 0.0

@beartype
def accuracy(y_true: NpF64, y_pred: NpF64) -> float:
    """Classification Accuracy."""
    pred_labels = y_pred if y_pred.ndim == 1 else np.argmax(y_pred, axis=-1)
    true_labels = y_true if y_true.ndim == 1 else np.argmax(y_true, axis=-1)
    
    return float(np.mean(pred_labels == true_labels))

@beartype
def vpt_score(y_true: NpF64, y_pred: NpF64, threshold: float) -> int:
    """
    Valid Prediction Time (VPT).
    Calculated based on normalized error at each time step.
    For multivariate data, we use the Euclidean norm of the error vector.
    """
    # Ensure shape (Time, Features) or (Time,)
    # Working with copies to avoid side effects on reshapes if referenced
    if y_true.ndim == 1:
        y_true_rs = y_true.reshape(-1, 1)
        y_pred_rs = y_pred.reshape(-1, 1)
    else:
        y_true_rs = y_true
        y_pred_rs = y_pred

    min_len = min(len(y_true_rs), len(y_pred_rs))
    y_true_rs = y_true_rs[:min_len]
    y_pred_rs = y_pred_rs[:min_len]

    # Calculate error at each step (Euclidean norm over features)
    errors = np.linalg.norm(y_pred_rs - y_true_rs, axis=-1)

    # Calculate scale (Global standard deviation magnitude)
    stds = np.std(y_true_rs, axis=0)
    scale = np.linalg.norm(stds)

    if scale > 1e-9:
        normalized_errors = errors / scale
        # Find first index where error exceeds threshold
        is_exceeded = normalized_errors > threshold
        if np.any(is_exceeded):
            return int(np.argmax(is_exceeded))
        else:
             return len(y_true_rs)
    return 0

# --- Dispatcher and Aggregator ---

@beartype
def compute_score(preds: NpF64, targets: NpF64, metric_name: str) -> float:
    """
    Compute generic score (MSE, NMSE, or Accuracy).
    Dispatcher to specific functions.
    """
    name = metric_name.lower()
    
    if name == "accuracy":
        return accuracy(targets, preds)
    elif name == "nmse":
        return nmse(targets, preds)
    elif name == "mse":
        return mse(targets, preds)
    elif name == "rmse":
        return rmse(targets, preds)
    elif name == "nrmse":
        return nrmse(targets, preds)
    elif name == "mase":
        return mase(targets, preds)
    
    # Default to MSE if unknown (or raise error, but maintaining old behavior)
    return mse(targets, preds)

@beartype
def calculate_chaos_metrics(
    y_true: NpF64,
    y_pred: NpF64,
    dt: float,
    lyapunov_time_unit: float,
    vpt_threshold: float = 0.4,
) -> EvalMetrics:
    """
    Calculate Chaos prediction metrics (Mackey-Glass, etc).
    Wrapper around detailed metric functions.
    """
    # We use flattened arrays for chaos metrics usually
    # Helpers handle flattening internally if needed (MASE, Correlation, VPT)
    # MSE/NMSE handle logic internally using jnp
    
    # Calculate all metrics
    val_mse = mse(y_true, y_pred)
    val_nmse = nmse(y_true, y_pred)
    val_nrmse = nrmse(y_true, y_pred)
    val_mase = mase(y_true, y_pred)
    val_ndei = ndei(y_true, y_pred)
    val_var_ratio = var_ratio(y_true, y_pred)
    val_corr = correlation(y_true, y_pred)
    val_vpt_steps = vpt_score(y_true, y_pred, vpt_threshold)
    
    # Convert VPT to Lyapunov time
    steps_per_lt = int(lyapunov_time_unit / dt) if dt > 0 else 1
    val_vpt_lt = float(val_vpt_steps) / steps_per_lt if steps_per_lt > 0 else 0.0

    return {
        "mse": val_mse,
        "nmse": val_nmse,
        "nrmse": val_nrmse,
        "mase": val_mase,
        "ndei": val_ndei,
        "var_ratio": val_var_ratio,
        "correlation": val_corr,
        "vpt_steps": float(val_vpt_steps),
        "vpt_lt": val_vpt_lt,
        "vpt_threshold": float(vpt_threshold),
    }
