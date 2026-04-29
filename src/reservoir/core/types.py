"""
reservoir/core/types.py — Central Type Definitions & Domain Gateway

厳格な型エイリアス定義。AnyやUnionは一切禁止。
各ドメインのファイルはここからimportし、迷わず正しい型を使う。

NUMPY Domain → NpF64 (import numpy のみ使うファイル用)
JAX Domain   → JaxF64 (import jax のみ使うファイル用)

CPU↔GPU転送 → to_jax_f64() / to_np_f64() を必ず通す（関所）

NOTE: このファイルは MAPPER として登録済み（lint_imports.py）。
      型定義の橋渡し（Bridge）+ ドメイン転送の関所という責任を持つ。
"""
from beartype import beartype
from jaxtyping import Float64, UInt32, jaxtyped
import jax
from jax import Array
import jax.numpy as jnp
import numpy as np

from typing import TypedDict, TYPE_CHECKING, Protocol, runtime_checkable
from collections.abc import Iterator

if TYPE_CHECKING:
    import reservoir.readout.base
    import reservoir.layers.preprocessing

# ==========================================
# 型エイリアス定義
# ==========================================
NpF64 = Float64[np.ndarray, "..."]
JaxF64 = Float64[Array, "..."]
JaxKey = UInt32[Array, "..."]  # JAX PRNG key (uint32)

BatchData = tuple[JaxF64, JaxF64 | None]
BatchIterator = Iterator[BatchData]
ModelState = JaxF64 | tuple[JaxF64, JaxF64 | None]
type FlaxParamTree = JaxF64 | dict[str, FlaxParamTree]
ArrayResult = JaxF64 | NpF64

@runtime_checkable
class DataLoaderProtocol(Protocol):
    X: NpF64 | None
    y: NpF64 | None
    num_samples: int
    batch_size: int
    def __iter__(self) -> BatchIterator: ...

class TrainLogs(TypedDict, total=False):
    """Strictly typed training logs to replace Dict[str, object]."""
    loss_history: list[float]
    final_loss: float
    distill_mse: float
    accuracy: float
    # Add other specific keys as they emerge.


def empty_train_logs() -> TrainLogs:
    """Return an explicitly typed empty training log."""
    logs: TrainLogs = {}
    return logs

class RegressionMetrics(TypedDict, total=False):
    """Scalar regression metrics."""
    mse: float
    mae: float


class ClassificationMetrics(TypedDict, total=False):
    """Scalar classification metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float


class ChaosMetrics(TypedDict, total=False):
    """Scalar chaos and forecasting metrics."""
    nmse: float
    nrmse: float
    mase: float
    ndei: float
    var_ratio: float
    correlation: float
    vpt_steps: float 
    vpt_lt: float
    vpt_threshold: float


class EvalMetrics(RegressionMetrics, ClassificationMetrics, ChaosMetrics, total=False):
    """Strictly typed evaluation metrics to replace Dict[str, float]."""

class TrainMetrics(EvalMetrics, total=False):
    weight_norms: dict[float, float]
    search_history: dict[float, float]
    best_lambda: float | None

class TestMetrics(EvalMetrics, total=False):
    chaos_metrics: dict[str, float]

# ==========================================
# Config Domain Types (Nesting, No Recursion to satisfy beartype)
# ==========================================

# 値になりうる基本型
PrimitiveValue = str | float | int | bool | None

# Recursive config values for serialization boundaries.
type ConfigValue = PrimitiveValue | tuple[ConfigValue, ...] | list[ConfigValue] | dict[str, ConfigValue]
type ConfigDict = dict[str, ConfigValue]

# ==========================================
# Topology Metadata Types (Model Builder Output)
# ==========================================

class ShapesMeta(TypedDict, total=False):
    """Step shapes through the pipeline."""
    input: tuple[int, ...] | None
    preprocessed: tuple[int, ...] | None
    projected: tuple[int, ...] | None
    adapter: tuple[int, ...] | None
    internal: tuple[int, ...] | None
    feature: tuple[int, ...] | None
    output: tuple[int, ...] | None

class DetailsMeta(TypedDict, total=False):
    """Pipeline component details."""
    preprocess: str | None
    agg_mode: str | None
    readout: str | None
    adapter: str | None
    student_layers: tuple[int, ...] | None
    student_structure: str | None
    window_size: int | None
    structure: str | None

class TopologyMeta(TypedDict, total=False):
    """Topology metadata produced by model factories and enriched by ModelBuilder."""
    type: str
    shapes: ShapesMeta
    details: DetailsMeta

# ==========================================
# Result Domain Types (Execution Outputs)
# ==========================================

class FitResultMetrics(TypedDict, total=False):
    train: EvalMetrics
    val: EvalMetrics
    test: EvalMetrics

class FitResultDict(TypedDict, total=False):
    train_pred: ArrayResult | None
    val_pred: ArrayResult | None
    test_pred: ArrayResult | None
    metrics: FitResultMetrics
    best_lambda: float | None
    best_score: float | None
    search_history: dict[float, float]
    weight_norms: dict[float, float]
    residuals_history: dict[float, NpF64] | None
    closed_loop_pred: ArrayResult | None
    closed_loop_history: ArrayResult | None
    closed_loop_truth: ArrayResult | None
    chaos_results: dict[str, float] | None
    outputs: dict[str, ArrayResult | None]
    aligned_test_y: ArrayResult | None

class ResultDict(TypedDict, total=False):
    fit_result: FitResultDict
    train_logs: TrainLogs
    quantum_trace: NpF64 | None
    train: TrainMetrics
    test: TestMetrics
    validation: EvalMetrics
    outputs: dict[str, NpF64 | None]
    readout: reservoir.readout.base.ReadoutModule | None
    preprocessor: reservoir.layers.preprocessing.Preprocessor | None
    scaler: reservoir.layers.preprocessing.Preprocessor | None
    training_logs: TrainLogs
    meta: dict[str, float | str]
    is_closed_loop: bool
    residuals_history: dict[float, NpF64] | None

# **kwargs 用の厳格な型定義 (No Any)
KwargsDict = dict[str, PrimitiveValue | JaxF64 | NpF64 | tuple[PrimitiveValue, ...] | list[PrimitiveValue] | ConfigDict | ResultDict]


# ==========================================
# Domain Gateway（関所）— CPU ↔ GPU 転送
# ==========================================

#takes only NpF64 and returns JaxF64, checks for NaN/Inf, and uses jax.device_put to ensure it's on GPU
@jaxtyped(typechecker=beartype)
def to_jax_f64(x: NpF64) -> JaxF64:
    """NumPy(CPU) → JAX(GPU) 変換の関所。

    - beartype が NpF64 (numpy.float64) のみ受け付ける
    - NaN/Inf が混入していたら即クラッシュ
    - jax.device_put で明示的にGPUへ転送
    """
    if np.any(np.isnan(x)):
        raise ValueError(f"NaN detected at CPU→GPU boundary! shape={x.shape}")
    if np.any(np.isinf(x)):
        raise ValueError(f"Inf detected at CPU→GPU boundary! shape={x.shape}")
    ret = jax.device_put(jnp.array(x, dtype=jnp.float64))
    if ret.dtype != jnp.float64:
        print(f"DEBUG to_jax_f64: {x.dtype.name} -> {ret.dtype.name}, config={jax.config.read('jax_enable_x64')}")
    return ret

#takes only JaxF64 and returns NpF64, checks for NaN/Inf, and uses np.asarray to ensure it's on CPU
@jaxtyped(typechecker=beartype)
def to_np_f64(x: JaxF64) -> NpF64:
    """JAX(GPU) → NumPy(CPU) 変換の関所。

    - beartype が JaxF64 (jax.Array float64) のみ受け付ける
    - NaN/Inf が混入していたら即クラッシュ
    - np.asarray で明示的にCPUへ回収
    """
    result = np.asarray(x, dtype=np.float64)
    if np.any(np.isnan(result)):
        raise ValueError(f"NaN detected at GPU→CPU boundary! shape={result.shape}")
    if np.any(np.isinf(result)):
        raise ValueError(f"Inf detected at GPU→CPU boundary! shape={result.shape}")
    return result
