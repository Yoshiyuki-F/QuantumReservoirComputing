"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/utils/plotting.py
Classification visualization utilities with validation support.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from reservoir.core.types import NpF64


def _find_project_root() -> Path:
    """Locate the project root (directory containing pyproject.toml)."""
    path = Path(__file__).resolve()
    for parent in path.parents:
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path(__file__).parent.parent


PROJECT_ROOT = _find_project_root()


def _resolve_output_path(filename: str) -> Path:
    """Resolve output path under the project root."""
    path = Path(filename)
    if path.is_absolute() or (len(path.parts) > 0 and path.parts[0] == "outputs"):
        return PROJECT_ROOT / path
    return PROJECT_ROOT / "outputs" / path


def plot_classification_results(
    train_labels: NpF64,
    test_labels: NpF64,
    train_predictions: NpF64,
    test_predictions: NpF64,
    title: str,
    filename: str,
    metrics_info: dict[str, str] | None = None,
    val_labels: NpF64 | None = None,
    val_predictions: NpF64 | None = None,
    best_lambda: float | None = None,
    lambda_norm: float | None = None,
    class_names: Sequence[str] | None = None,
) -> None:
    """
    Visualize classification results with confusion matrix and accuracy bars, with optional validation.
    Annotates the selected ridge lambda and weight norm when provided.
    """

    train_labels_np = train_labels
    test_labels_np = test_labels
    train_predictions_np = train_predictions
    test_predictions_np = test_predictions

    val_labels_np = val_labels
    val_predictions_np = val_predictions

    def _safe_max(array: NpF64) -> int:
        return int(array.max()) if array.size > 0 else -1

    if class_names is None:
        inferred_max = max(
            _safe_max(train_labels_np),
            _safe_max(test_labels_np),
            _safe_max(train_predictions_np),
            _safe_max(test_predictions_np),
            _safe_max(val_labels_np) if val_labels_np is not None else -1,
            _safe_max(val_predictions_np) if val_predictions_np is not None else -1,
        )
        num_classes = max(inferred_max + 1, 1)
        class_names = [str(idx) for idx in range(num_classes)]
    else:
        num_classes = len(class_names)

    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    valid_mask = (
        (test_labels_np >= 0)
        & (test_labels_np < num_classes)
        & (test_predictions_np >= 0)
        & (test_predictions_np < num_classes)
    )
    np.add.at(
        confusion_matrix,
        (test_labels_np[valid_mask], test_predictions_np[valid_mask]),
        1,
    )

    def _calc_acc(true_y: NpF64, pred_y: NpF64) -> float:
        if true_y.size == 0:
            return 0.0
        return float(np.mean(true_y == pred_y))

    train_accuracy = _calc_acc(train_labels_np, train_predictions_np)
    test_accuracy = _calc_acc(test_labels_np, test_predictions_np)
    val_accuracy = None
    if val_labels_np is not None and val_predictions_np is not None:
        val_accuracy = _calc_acc(val_labels_np, val_predictions_np)
    elif metrics_info and "Val Acc" in metrics_info:
        try:
            val_accuracy = float(metrics_info["Val Acc"])
            if val_accuracy > 1.0:
                val_accuracy = val_accuracy / 100.0
        except (TypeError, ValueError):
            val_accuracy = None

    fig, (ax_conf, ax_bar) = plt.subplots(1, 2, figsize=(12, 6))

    im = ax_conf.imshow(confusion_matrix, interpolation='nearest', cmap='Blues')
    cbar = fig.colorbar(im, ax=ax_conf, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel('Count', rotation=-90, va='bottom')

    ax_conf.set_title('Test Confusion Matrix')
    ax_conf.set_xlabel('Predicted Label')
    ax_conf.set_ylabel('True Label')
    ax_conf.set_xticks(range(num_classes))
    ax_conf.set_yticks(range(num_classes))
    ax_conf.set_xticklabels(class_names, rotation=45, ha='right')
    ax_conf.set_yticklabels(class_names)

    threshold = (confusion_matrix.max() / 2.0) if confusion_matrix.max() > 0 else 0.5
    for i in range(num_classes):
        for j in range(num_classes):
            value = confusion_matrix[i, j]
            color = 'white' if value > threshold else 'black'
            ax_conf.text(j, i, str(int(value)), ha='center', va='center', color=color, fontsize=9)

    # Plot from Bottom to Top, so for Top-to-Bottom order (Train, Val, Test),
    # we need to supply them as [Test, Val, Train].
    accuracy_labels = ['Test', 'Train']
    accuracy_values = [test_accuracy * 100.0, train_accuracy * 100.0]
    bar_colors = ['tab:orange', 'tab:green']

    if val_accuracy is not None:
        # Insert Val in the middle
        accuracy_labels.insert(1, 'Val')
        accuracy_values.insert(1, val_accuracy * 100.0)
        bar_colors.insert(1, 'tab:purple')

    bars = ax_bar.barh(accuracy_labels, accuracy_values, color=bar_colors)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel('Accuracy (%)')
    ax_bar.set_title('Accuracy Overview')
    ax_bar.grid(True, axis='x', alpha=0.3, linestyle='--')

    for bar, accuracy in zip(bars, accuracy_values, strict=False):
        display_value = min(accuracy + 1.0, 99.9)
        ax_bar.text(
            display_value,
            bar.get_y() + bar.get_height() / 2,
            f"{accuracy:.2f}%",
            va='center',
            fontsize=10,
        )

    summary_parts: list[str] = []
    if best_lambda is not None:
        lambda_str = f"lambda*: {best_lambda:.3e}"
        norm_str = f" ||w||={float(lambda_norm):.3e}" if lambda_norm is not None else ""
        summary_parts.append(lambda_str + norm_str)
    if summary_parts:
        fig.text(0.5, 0.92, " | ".join(summary_parts), ha='center', fontsize=10, color='gray')

    fig.suptitle(title, fontsize=16)
    fig.tight_layout(rect=(0, 0.05, 1, 0.92))

    output_path = _resolve_output_path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"分類結果を '{output_path}' に保存しました。")


def plot_loss_history(history: Sequence[float], filename: str, title: str = "Loss Curve", learning_rate: float | None = None) -> None:
    """Plot a generic loss history curve and save to outputs."""
    output_path = _resolve_output_path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(history) + 1), history, marker="o")
    plt.yscale("log")
    
    # Add learning rate to title if provided
    full_title = title
    if learning_rate is not None:
        full_title = f"{title} (LR={learning_rate:.4f})"
    plt.title(full_title)
    
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Loss curve saved to '{output_path}'.")



def plot_timeseries_comparison(
    targets: NpF64,
    predictions: NpF64,
    filename: str,
    title: str = "TimeSeries Prediction",
    max_features: int = 3,
    start_step: int = 0,
    end_step: int | None = None,
    time_offset: int = 0,
) -> None:
    """
    Plot predicted vs actual time series trajectories.
    Handles 3D (Batch, Time, Feat) or 2D (Time, Feat) inputs.
    Plots the first sample (if batched) and up to `max_features` features.
    """
    targets_np = targets
    preds_np = predictions

    # Standardize to (Time, Feat)
    if targets_np.ndim == 3:
        # Take first sample in batch
        target_seq = targets_np[0]
        pred_seq = preds_np[0]
    elif targets_np.ndim == 2:
        target_seq = targets_np
        pred_seq = preds_np
    else:
        print(f"Skipping plot: expected 2D/3D arrays, got {targets_np.shape}")
        return

    time_steps, n_feats = target_seq.shape
    plot_feats = min(n_feats, max_features)
    
    if end_step is None:
        end_step = time_steps

    # Slice time (0-based for array indexing)
    target_slice = target_seq[start_step:end_step]
    pred_slice = pred_seq[start_step:end_step]
    
    # Time axis (shifted by global offset)
    t_start = start_step + time_offset
    t_end = t_start + (end_step - start_step)
    t_axis = np.arange(t_start, t_end)



    fig, axes = plt.subplots(plot_feats, 1, figsize=(12, 3 * plot_feats), sharex=True)

    axes_list = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    for i in range(plot_feats):
        ax = axes_list[i]
        ax.plot(t_axis, target_slice[:, i], label="Actual", color="black", alpha=0.7)
        ax.plot(t_axis, pred_slice[:, i], label="Predicted", color="tab:blue", alpha=0.9, linestyle="--")
        ax.set_ylabel(f"Feature {i}")
        ax.grid(True, linestyle=":", alpha=0.5)
        if i == 0:
            ax.legend(loc="upper right")
        if i == plot_feats - 1:
            ax.set_xlabel("Time Step")

    output_path = _resolve_output_path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Prediction plot saved to '{output_path}'.")


def plot_lambda_search_boxplot(
    residuals_history: dict[float, NpF64],
    filename: str,
    title: str = "Lambda Search: Residuals Distribution",
    best_lambda: float | None = None,
    metric_name: str = "NMSE",
    weight_norms: dict[float, float] | None = None,
) -> None:
    """
    Plot boxplot of per-sample squared errors for each lambda.
    Highlights the selected (best) lambda box in red.
    """
    if not residuals_history:
        return

    sorted_lambdas = sorted(residuals_history.keys())
    data = []
    labels = []
    
    best_idx = None  # 1-based index of best lambda box

    for i, lam in enumerate(sorted_lambdas):
        res = residuals_history[lam]
        res = res[np.isfinite(res)]
        data.append(res)
        
        # Create label with Lambda and optionally Weight Norm
        lam_label = f"{lam:.1e}"
        if weight_norms and lam in weight_norms:
            norm_val = weight_norms[lam]
            lam_label += f"\n{norm_val:.1e}"
        labels.append(lam_label)
        
        if best_lambda is not None and abs(lam - best_lambda) < 1e-15:
            best_idx = i + 1  # boxplot uses 1-based indexing

    output_path = _resolve_output_path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 6))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)
    
    # Style all boxes light gray
    for patch in bp['boxes']:
        patch.set_facecolor('#E8E8E8')
        patch.set_alpha(0.8)
    
    # Highlight best lambda box in red
    if best_idx is not None and best_idx <= len(bp['boxes']):
        bp['boxes'][best_idx - 1].set_facecolor('#FF6B6B')
        bp['boxes'][best_idx - 1].set_edgecolor('#CC0000')
        bp['boxes'][best_idx - 1].set_linewidth(2)
        bp['boxes'][best_idx - 1].set_alpha(1.0)

    ax.set_title(title, fontsize=13)
    ax.set_xlabel("Lambda (Regularization Strength)")
    ax.set_ylabel(f"Normalized Residuals (Individual contribution to {metric_name})")
    ax.set_yscale("log")
    
    # Plot line connecting means for better visibility of convexity/humps
    means = [np.mean(d) for d in data]
    x_pos = np.arange(1, len(data) + 1)
    ax.plot(x_pos, means, color='#2c3e50', linestyle='--', marker='o', 
            markersize=4, alpha=0.6, label='Mean residuals')
    
    plt.xticks(rotation=45, ha='right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Lambda Search BoxPlot saved to '{output_path}'.")
