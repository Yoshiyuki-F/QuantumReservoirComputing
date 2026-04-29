"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/generators.py
Reservoir Computing用のデータ生成関数。
"""

from beartype import beartype
from reservoir.core.types import NpF64

import numpy as np

try:
    import torch 
    from reservoir.data.mnist_loader import get_mnist_datasets, image_to_sequence
except ModuleNotFoundError:  # pragma: no cover - torch optional
    torch = None
    get_mnist_datasets = None 
    image_to_sequence = None 

from reservoir.data.config import (
    SineWaveConfig,
    LorenzConfig,
    Lorenz96Config,
    MackeyGlassConfig,
    MNISTConfig,
)


@beartype
def generate_sine_data(config: SineWaveConfig) -> tuple[NpF64, NpF64]:
    """複数周波数のサイン波を合成した時系列データを生成。
    
    複数の正弦波を重ね合わせて合成信号を作成し、ガウシアンノイズを
    追加した時系列データを生成します。Reservoir Computingの訓練・
    テスト用データとして使用されます。
    
    Args:
        config: SineWaveConfig オブジェクト
            
    Returns:
        tuple: (入力データ, 目標データ) のペア
            - 入力データ: 形状 (time_steps-1, 1) の現在値
            - 目標データ: 形状 (time_steps-1, 1) の次時刻値
    """

    t = np.arange(config.time_steps)
    
    # 複数の周波数のサインwave合成
    signal = np.zeros(config.time_steps)
    frequencies = config.frequencies
    if not frequencies:
        raise ValueError("sine_wave requires non-empty frequencies.")

    for freq in frequencies:
        signal += np.sin(2 * np.pi * freq * t)
    
    # ノイズを追加
    noise = np.random.normal(0, config.noise_level, config.time_steps)
    signal += noise
    
    # 入力は現在の値、ターゲットは次の値
    input_data = signal[:-1].reshape(-1, 1)   # 形状: (time_steps-1, 1)
    target_data = signal[1:].reshape(-1, 1)   # 形状: (time_steps-1, 1)
    
    return input_data, target_data


@beartype
def generate_lorenz_data(config: LorenzConfig) -> tuple[NpF64, NpF64]:
    """Lorenz方程式による決定論的カオス時系列データを生成。
    
    Lorenzアトラクターは気象学から生まれた有名なカオス系で、
    決定論的でありながら複雑で予測困難な時系列を生成します。
    Reservoir Computingのテストに適した非線形動力学系です。
    
    微分方程式系:
        dx/dt = σ(y - x)
        dy/dt = x(ρ - z) - y  
        dz/dt = xy - βz
        
    Args:
        config: LorenzConfig オブジェクト
        
    Returns:
        tuple: (入力データ, 目標データ) のペア
            - 入力データ: 形状 (time_steps-1, 3) の現在状態 [x, y, z]
            - 目標データ: 形状 (time_steps-1, 3) の次時刻状態

        
    Note:
        数値積分にはオイラー法を使用。より高精度が必要な場合は
        Runge-Kutta法の実装を検討してください。
        washup_lt * lt でウォームアップステップ数を計算します。
    """
    # 初期値（互換性のためオプショナル）
    lorenz_x = 1.0
    lorenz_y = 1.0
    lorenz_z = 1.0
    
    # Calculate steps from LT parameters
    # steps_per_lt = lyapunov_time_unit / dt
    # Add +1 to account for input/target shift (X=data[:-1], y=data[1:])
    steps_per_lt = int(config.lyapunov_time_unit / config.dt)
    warmup_steps = int(config.washup_lt * steps_per_lt)
    data_steps = int((config.train_lt + config.val_lt + config.test_lt) * steps_per_lt) + 1
    total_steps = warmup_steps + data_steps
    
    data = np.zeros((total_steps, 3))
    
    for i in range(total_steps):
        # Lorenz方程式の数値積分（オイラー法）
        dx = config.sigma * (lorenz_y - lorenz_x)
        dy = lorenz_x * (config.rho - lorenz_z) - lorenz_y
        dz = lorenz_x * lorenz_y - config.beta * lorenz_z
        
        lorenz_x += dx * config.dt
        lorenz_y += dy * config.dt
        lorenz_z += dz * config.dt
        
        data[i] = [lorenz_x, lorenz_y, lorenz_z]
    
    # Discard warmup steps
    data = data[warmup_steps:]
    
    # 入力は現在の状態、ターゲットは次の状態
    input_data = data[:-1]
    target_data = data[1:]
    
    return input_data, target_data


@beartype
def generate_lorenz96_data(config: Lorenz96Config) -> tuple[NpF64, NpF64]:
    """Generates Lorenz 96 chaotic time series data.
    
    The Lorenz 96 model is defined by:
    dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F
    
    Args:
        config: Lorenz96Config object
        
    Returns:
        tuple: (input_data, target_data)
            - input_data: Shape (time_steps, N)
            - target_data: Shape (time_steps, N)
    """
    N = config.n_input
    F = config.F
    dt = config.dt
    
    # Initialization
    if config.seed is not None:
        np.random.seed(config.seed)
    
    x0 = np.full(N, F, dtype=np.float64)
    x0 += np.random.normal(0, 0.01, N)
    
    steps_per_lt = int(config.lyapunov_time_unit / config.dt)
    warmup_steps = int(config.washup_lt * steps_per_lt)
    data_steps = int((config.train_lt + config.val_lt + config.test_lt) * steps_per_lt) + 1
    total_steps = warmup_steps + data_steps

    # Pre-compute indices for cyclic boundary conditions
    indices = np.arange(N)
    idx_minus_2 = (indices - 2) % N
    idx_minus_1 = (indices - 1) % N
    idx_plus_1 = (indices + 1) % N

    def lorenz96_deriv(state_vec: NpF64) -> NpF64:
        # dx/dt = (x[i+1] - x[i-2]) * x[i-1] - x[i] + F
        return (state_vec[idx_plus_1] - state_vec[idx_minus_2]) * state_vec[idx_minus_1] - state_vec + F

    # RK4 Integration (pure NumPy loop — runs once during data generation)
    data = np.empty((total_steps, N), dtype=np.float64)
    state_vec = x0
    for t in range(total_steps):
        k1 = lorenz96_deriv(state_vec)
        k2 = lorenz96_deriv(state_vec + k1 * dt / 2)
        k3 = lorenz96_deriv(state_vec + k2 * dt / 2)
        k4 = lorenz96_deriv(state_vec + k3 * dt)
        state_vec = state_vec + (k1 + 2*k2 + 2*k3 + k4) * dt / 6.0
        data[t] = state_vec
        
    # Discard warmup steps
    data = data[warmup_steps:]
    
    # Input is current state, Target is next state
    input_data = data[:-1]
    target_data = data[1:]
    
    return input_data, target_data


@beartype
def generate_mackey_glass_data(config: MackeyGlassConfig) -> tuple[NpF64, NpF64]:
    """
    Mackey-Glassカオス時系列データを生成します。

    Args:
        config: MackeyGlassConfig オブジェクト

    Returns:
        (input_data, target_data): 入力データとターゲットデータのタプル

    Note:
        トランジェント除去が不要な場合は`washup_lt`を0に設定できます。
        washup_lt * lt でウォームアップステップ数を計算します。
        tauは連続時間の遅延として解釈され、離散化ステップdtで割ることでサンプル遅延数に変換されます。
    """
    tau = int(config.tau)
    beta = float(config.beta)
    gamma = float(config.gamma)
    n = float(config.n)

    dt = float(config.dt)
    if dt <= 0:
        raise ValueError("dt must be positive for Mackey-Glass generation")

    # tauの単位をdtに合わせた遅延ステップ数へ変換
    delay_steps = int(np.round(float(tau) / dt))
    if delay_steps < 1:
        delay_steps = 1

    # 初期化とトランジェント除去用のウォームアップ（省略可）
    # steps_per_lt = lyapunov_time_unit / dt
    # Add +1 to account for input/target shift (X=data[:-1], y=data[1:])
    history_length = delay_steps + 1
    steps_per_lt = int(config.lyapunov_time_unit / config.dt)
    warmup_steps = max(int(config.washup_lt * steps_per_lt), 0)
    data_steps = int((config.train_lt + config.val_lt + config.test_lt) * steps_per_lt) + 1
    total_steps = data_steps + history_length + warmup_steps

    series = np.full(total_steps, fill_value=0.0)
    initial_value = 1.2
    series[:history_length] = initial_value

    for i in range(history_length, total_steps):
        x_prev = series[i - 1]
        x_tau = series[i - delay_steps]
        dx = (beta * x_tau) / (1 + x_tau**n) - gamma * x_prev
        series[i] = x_prev + dt * dx

    # トランジェント除去
    start_idx = history_length + warmup_steps
    end_idx = start_idx + data_steps
    if end_idx > len(series):
        raise ValueError("Warmup and history exceed generated sequence length")
    series = series[start_idx:end_idx]

    # ノイズを追加
    if config.noise_level > 0:
        # Use config seed for deterministic noise if provided
        if config.seed is not None:
             np.random.seed(config.seed)
        noise = np.random.normal(0, config.noise_level, len(series))
        series += noise

    # 入力は現在の値、ターゲットは次の値
    input_data = series[:-1].reshape(-1, 1)
    target_data = series[1:].reshape(-1, 1)

    return input_data, target_data


def generate_mnist_sequence_data(
    config: MNISTConfig,
    *,
    split: str | None = None,
) -> tuple[NpF64, NpF64]:
    """
    Generate MNIST-based sequence data by scanning images as time series.

    Args:
        config: MNIST generation configuration.
        split: Dataset split override. Uses ``config.split`` when omitted.

    Returns:
        Tuple (input_sequences, labels) where both are numpy arrays.

    Raises:
        ImportError: If torch/torchvision are not available.
    """
    if get_mnist_datasets is None or image_to_sequence is None or torch is None:
        raise ImportError(
            "MNIST generation requires torch and torchvision. "
            "Install them to enable this feature."
        )

    split_name = split or config.split

    total_pixels = 28 * 28
    n_steps = int(config.time_steps)
    if n_steps <= 0:
        raise ValueError(f"time_steps must be positive, got {n_steps}")
    if total_pixels % n_steps != 0:
        raise ValueError(
            f"time_steps={n_steps} must evenly divide {total_pixels} to reshape MNIST images"
        )

    train_set, test_set = get_mnist_datasets()
    if split_name not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got '{split_name}'")
    dataset = train_set if split_name == "train" else test_set

    max_available = len(dataset)
    fraction_key = f"{split_name}_fraction"
    fraction_param = config.train_fraction if split_name == "train" else config.test_fraction
    fraction_val = float(fraction_param)
    if fraction_val <= 0 or fraction_val > 1:
        raise ValueError(f"{fraction_key} must be in (0, 1], got {fraction_val}")
    limit = max(1, min(max_available, int(max_available * fraction_val)))

    sequences = []
    labels = []

    for idx in range(limit):
        img_tensor, label = dataset[idx]
        img_np = img_tensor.double().numpy()
        seq = image_to_sequence(img_np, n_steps=n_steps)
        sequences.append(seq)
        labels.append(label)

    input_data = np.stack(sequences, axis=0)
    return input_data, np.array(labels, dtype=np.float64)
