"""
Visualization utilities for Quantum Reservoir Computing.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from reservoir.core.types import NpF64


from reservoir.utils.plotting import _resolve_output_path


def plot_qubit_dynamics(
    states: NpF64,
    filename: str,
    title: str = "Quantum Reservoir Dynamics",
    feature_names: list[str] | None = None,
    cmap: str = "RdBu_r",
    vmin: float = -1.0,
    vmax: float = 1.0,
) -> None:
    """
    Plots a heatmap of qubit expectation values over time.
    
    Args:
        states: NpF64 of shape (Time, Features). If batch dim exists, take first.
        filename: Output filename (saved to outputs/).
        title: Plot title.
        feature_names: Optional list of labels for Y-axis (e.g., ["Z0", "Z1", "Z0Z1"]).
        cmap: Colormap (default RdBu_r for -1 to 1 values).
        vmin: Min value for colorbar.
        vmax: Max value for colorbar.
    """
    # Handle Batch Dimension: (Batch, Time, Feat) -> (Time, Feat)
    if states.ndim == 3:
        states = states[0]
        
    if states.ndim != 2:
        raise ValueError(f"Expected 2D (Time, Feat) or 3D (Batch, Time, Feat) array, got {states.shape}")
        
    time_steps, n_features = states.shape
    
    # Setup Figure
    # Adaptive height based on number of features
    height = max(4, n_features * 0.3)
    fig, ax = plt.subplots(figsize=(10, height))
    
    # Transpose for (Features, Time) heatmap
    im = ax.imshow(
        states.T, 
        aspect="auto", 
        cmap=cmap, 
        interpolation="nearest",
        vmin=vmin, 
        vmax=vmax,
        extent=(0.0, float(time_steps), float(n_features) - 0.5, -0.5) # Align indices
    )
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Expectation Value <O>")
    
    # Labels
    ax.set_title(title)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Observable Index")
    
    # Y-Ticks
    if feature_names:
        if len(feature_names) != n_features:
            print(f"Warning: feature_names length ({len(feature_names)}) != n_features ({n_features}). Ignoring labels.")
        else:
            ax.set_yticks(range(n_features))
            ax.set_yticklabels(feature_names)
    else:
        # If too many features, sparse ticks
        if n_features > 20:
             step = n_features // 10
             ax.set_yticks(np.arange(0, n_features, step))
        else:
             ax.set_yticks(np.arange(n_features))
             
    # Grid (optional, maybe distracting for heatmap)
    # ax.grid(False)
    
    # Save
    output_path = _resolve_output_path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Quantum dynamics heatmap saved to '{output_path}'.")
