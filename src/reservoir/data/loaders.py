"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/loaders.py
Dataset loader registrations and preparation helpers."""

from __future__ import annotations

from collections.abc import Callable  # noqa: TC003
from typing import TYPE_CHECKING, TypeVar, cast
from beartype import beartype

import numpy as np

from reservoir.data.identifiers import Dataset
from reservoir.data.generators import (
    generate_sine_data,
    generate_mnist_sequence_data,
    generate_mackey_glass_data,
    generate_lorenz_data,
    generate_lorenz96_data,
)
from reservoir.data.presets import get_dataset_preset
from reservoir.training.presets import TrainingConfig, get_training_preset
from reservoir.data.structs import SplitDataset
from reservoir.core.types import NpF64
from reservoir.data.config import (
    BaseDatasetConfig,
    SineWaveConfig,
    MackeyGlassConfig,
    LorenzConfig,
    Lorenz96Config,
    MNISTConfig,
)

if TYPE_CHECKING:
    from reservoir.data.config import ChaosDatasetConfig


type LoaderFunc = (
    Callable[[BaseDatasetConfig], tuple[NpF64, NpF64] | SplitDataset]
)
type SpecificLoaderFunc = (
    Callable[[SineWaveConfig], tuple[NpF64, NpF64]]
    | Callable[[MNISTConfig], SplitDataset]
    | Callable[[MackeyGlassConfig], tuple[NpF64, NpF64]]
    | Callable[[LorenzConfig], tuple[NpF64, NpF64]]
    | Callable[[Lorenz96Config], tuple[NpF64, NpF64]]
)


LOADER_REGISTRY: dict[Dataset, LoaderFunc] = {}
F = TypeVar("F", bound=SpecificLoaderFunc)

def register_loader(dataset: Dataset) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        LOADER_REGISTRY[dataset] = cast("LoaderFunc", fn)
        return fn

    return decorator


@register_loader(Dataset.SINE_WAVE)
@beartype
def load_sine_wave(config: SineWaveConfig) -> tuple[NpF64, NpF64]:
    """Load or generate sine wave data and return as (N, T, F) sequences."""
    X_arr, y_arr = generate_sine_data(config)
    if X_arr.dtype != np.float64:
        raise ValueError(f"Sine wave X_arr must be float64, got {X_arr.dtype}")
    if y_arr.dtype != np.float64:
        raise ValueError(f"Sine wave y_arr must be float64, got {y_arr.dtype}")

    # Ensure 3D shape (N, T, F). Treat each timestep as a length-1 sequence.
    if X_arr.ndim == 2:
        X_arr = X_arr[:, None, :]

    return X_arr, y_arr


@register_loader(Dataset.MNIST)
def load_mnist(config: MNISTConfig) -> SplitDataset:
    """Load MNIST sequence dataset as canonical train/val/test splits."""
    if generate_mnist_sequence_data is None:
        raise ImportError("MNIST sequence loader requires torch/torchvision.")
    train_arr, train_labels = generate_mnist_sequence_data(config, split=config.split)
    test_arr , test_labels = generate_mnist_sequence_data(config, split="test")

    if train_arr.dtype != np.float64:
        raise ValueError(f"MNIST train_arr must be float64, got {train_arr.dtype}")
    if test_arr.dtype != np.float64:
        raise ValueError(f"MNIST test_arr must be float64, got {test_arr.dtype}")

    # Ensure (N, T, F)
    if train_arr.ndim == 2:
        train_arr = train_arr[..., None]
    if test_arr.ndim == 2:
        test_arr = test_arr[..., None]

    # Flatten any spatial dims into feature dim while preserving time length.
    train_arr = train_arr.reshape(train_arr.shape[0], train_arr.shape[1], -1)
    test_arr = test_arr.reshape(test_arr.shape[0], test_arr.shape[1], -1)

    num_classes = int(config.n_output)
    
    # Memory-efficient Numpy-based one-hot encoding (avoiding np.eye broadcasting)
    train_labels_arr = np.zeros((len(train_labels), num_classes), dtype=np.float64)
    train_labels_arr[np.arange(len(train_labels)), np.int32(train_labels)] = 1.0
    
    test_labels_arr = np.zeros((len(test_labels), num_classes), dtype=np.float64)
    test_labels_arr[np.arange(len(test_labels)), np.int32(test_labels)] = 1.0

    # Create validation split from training data (10%)
    val_ratio = 0.1
    n_train = train_arr.shape[0]
    n_val = max(1, int(n_train * val_ratio))
    n_train_new = n_train - n_val

    val_arr = train_arr[n_train_new:]
    val_labels_arr = train_labels_arr[n_train_new:]
    train_arr = train_arr[:n_train_new]
    train_labels_arr = train_labels_arr[:n_train_new]

    return SplitDataset(
        train_X=train_arr,
        train_y=train_labels_arr,
        test_X=test_arr,
        test_y=test_labels_arr,
        val_X=val_arr,
        val_y=val_labels_arr,
    )


@register_loader(Dataset.MACKEY_GLASS)
@beartype
def load_mackey_glass(config: MackeyGlassConfig) -> tuple[NpF64, NpF64]:
    """Generate Mackey-Glass samples (N, 1) compatible with sequence models."""

    # 1. Generate (returns jnp arrays)
    X_gen, y_gen = generate_mackey_glass_data(config)
    if X_gen.dtype != np.float64:
        raise ValueError(f"Mackey-Glass X_gen must be float64, got {X_gen.dtype}")
    if y_gen.dtype != np.float64:
        raise ValueError(f"Mackey-Glass y_gen must be float64, got {y_gen.dtype}")

    # 2. Reconstruct full sequence (N+1)
    # X_gen: (T, 1), y_gen: (T, 1)
    # y is X shifted by 1. full = [X[0], X[1]... X[T-1], y[T-1]]
    seq = np.append(X_gen.flatten(), y_gen[-1])
    
    # 3. Downsampling (Parameterized)
    step = getattr(config, "downsample", 1)
    if step > 1:
        seq = seq[::step]
        print(f"[loader.py] Applied downsampling: step={step}")

    # 4. Normalize (Disabled: Pipeline handles preprocessing)

    # 5. Naive MSE Baseline
    # "Naive MSE = mean((y[1:] - y[:-1])**2)"
    naive_loss = np.mean((seq[1:] - seq[:-1]) ** 2)
    print(f"[loader.py] Naive Prediction MSE (Baseline): {naive_loss:.6f}")
    
    # 6. Re-create X, y
    X_new = seq[:-1].reshape(-1, 1)
    y_new = seq[1:].reshape(-1, 1)
    
    X_arr = X_new
    y_arr = y_new
    
    return X_arr, y_arr


@register_loader(Dataset.LORENZ)
@beartype
def load_lorenz(config: LorenzConfig) -> tuple[NpF64, NpF64]:
    """Generate Lorenz attractor sequences."""
    X, y = generate_lorenz_data(config)
    if X.dtype != np.float64:
        raise ValueError(f"Lorenz X must be float64, got {X.dtype}")
    if y.dtype != np.float64:
        raise ValueError(f"Lorenz y must be float64, got {y.dtype}")
    # Return (T, F) so that splitting happens along the time axis (axis 0).
    # We will reshape this to (1, T, F) in load_dataset_with_validation_split.
    return X, y


@register_loader(Dataset.LORENZ96)
@beartype
def load_lorenz96(config: Lorenz96Config) -> tuple[NpF64, NpF64]:
    """Generate Lorenz 96 sequences."""
    X, y = generate_lorenz96_data(config)
    if X.dtype != np.float64:
        raise ValueError(f"Lorenz96 X must be float64, got {X.dtype}")
    if y.dtype != np.float64:
        raise ValueError(f"Lorenz96 y must be float64, got {y.dtype}")
    # Return (T, F) so that splitting happens along the time axis (axis 0).
    # We will reshape this to (1, T, F) in load_dataset_with_validation_split.
    # if X.ndim == 2:
    #     X = X[:, None, :]
    # if y.ndim == 2:
    #     y = y[:, None, :]
    return X, y


def load_dataset_with_validation_split(
    dataset_enum: Dataset,
    training_cfg: TrainingConfig | None = None,
) -> SplitDataset:
    """
    Load dataset via registry, apply task-specific preprocessing, and split into train/val/test.
    """
    if training_cfg is None:
        training_cfg = get_training_preset("standard")

    preset = get_dataset_preset(dataset_enum)
    if preset is None:
        raise ValueError(f"Dataset preset '{dataset_enum}' is not registered.")
    loader = LOADER_REGISTRY.get(dataset_enum)
    if loader is None:
        raise ValueError(f"No loader registered for dataset '{dataset_enum}'.")

    print(f"[loader.py] Loading dataset: {dataset_enum.value}...")
    dataset = loader(preset.config)

    # val is needed for both tasks
    val_size = float(training_cfg.val_size)

    def _split_validation(features: NpF64, labels: NpF64) -> tuple[NpF64, NpF64, NpF64, NpF64]:
        # Always create validation split (minimum 1 sample)
        val_count = max(1, int(len(features) * val_size)) if val_size > 0 else 1
        train_count = len(features) - val_count
        if train_count < 1:
            train_count = len(features) - 1

        val_features = features[train_count:]
        val_labels = labels[train_count:]
        return features[:train_count], labels[:train_count], val_features, val_labels

    train_X: NpF64
    train_y: NpF64
    test_X: NpF64
    test_y: NpF64

    if isinstance(dataset, SplitDataset):
        train_X = dataset.train_X
        train_y = dataset.train_y
        test_X = dataset.test_X
        test_y = dataset.test_y

        if dataset.val_X is not None or dataset.val_y is not None:
            val_X = dataset.val_X
            val_y = dataset.val_y
        else:
            train_X, train_y, val_X, val_y = _split_validation(train_X, train_y)
    else:
        try:
            X, y = dataset
        except (TypeError, ValueError):
            raise ValueError(f"Loader for dataset '{dataset_enum}' must return (X, y) tuple or SplitDataset.") from None

        total = len(X)
        if total < 2:
            raise ValueError(f"Dataset '{dataset_enum}' is too small to split (size={total}).")

        # Check if this is a chaotic dataset with LT-based splitting
        config = preset.config
        has_lt_split = (
            hasattr(config, 'train_lt') and 
            hasattr(config, 'val_lt') and 
            hasattr(config, 'test_lt') and
            hasattr(config, 'lyapunov_time_unit') and
            hasattr(config, 'dt')
        )

        if has_lt_split and getattr(config, "lyapunov_time_unit", 0) > 0:
            from typing import cast
            chaos_config = cast("ChaosDatasetConfig", config)
            # LT-based splitting for chaotic datasets
            # steps_per_lt = lyapunov_time_unit / dt
            steps_per_lt = int(chaos_config.lyapunov_time_unit / chaos_config.dt)
            train_count = int(chaos_config.train_lt * steps_per_lt)
            val_count = int(chaos_config.val_lt * steps_per_lt)
            test_count = int(chaos_config.test_lt * steps_per_lt)
            
            required_total = train_count + val_count + test_count
            if total < required_total:
                raise ValueError(
                    f"Dataset '{dataset_enum}' has {total} samples but LT-based split requires "
                    f"{required_total} (train={train_count}, val={val_count}, test={test_count})."
                )
            
            # Use the first (train+val+test) samples, ignore any extra
            train_X, train_y = X[:train_count], y[:train_count]
            val_X, val_y = X[train_count:train_count + val_count], y[train_count:train_count + val_count]
            test_X, test_y = X[train_count + val_count:train_count + val_count + test_count], y[train_count + val_count:train_count + val_count + test_count]
            
            print(f"[loader.py] LT-based split: train={train_count} ({chaos_config.train_lt} LT), "
                  f"val={val_count} ({chaos_config.val_lt} LT), test={test_count} ({chaos_config.test_lt} LT)")
        else:
            # Percentage-based splitting (original logic)
            train_ratio = float(training_cfg.train_size)
            test_ratio = float(training_cfg.test_ratio)

            if not (0.0 < train_ratio < 1.0):
                raise ValueError(f"train_size must be in (0,1), got {train_ratio}.")
            if not (0.0 <= test_ratio < 1.0):
                raise ValueError(f"test_ratio must be in [0,1), got {test_ratio}.")

            # Correct 8:1:1 split: calculate all counts from total
            import math
            train_count = max(1, math.ceil(total * train_ratio))
            val_count = max(1, round(total * val_size)) if val_size > 0 else 0
            test_count = total - train_count - val_count

            if train_count < 1 or test_count < 1:
                raise ValueError(f"Invalid split sizes: train={train_count}, val={val_count}, test={test_count} for total={total}.")

            train_X, train_y = X[:train_count], y[:train_count]
            # Always create validation split (minimum 1 sample)
            if val_count < 1:
                val_count = 1
                train_count = train_count - 1 if train_count > 1 else train_count
            val_X, val_y = X[train_count:train_count + val_count], y[train_count:train_count + val_count]
            test_X, test_y = X[train_count + val_count:], y[train_count + val_count:]

    #TODO why is Val_X has warning??
    return SplitDataset(train_X, train_y, test_X, test_y, val_X, val_y)
