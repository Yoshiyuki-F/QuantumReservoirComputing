from reservoir.core.types import JaxF64, EvalMetrics, to_np_f64
from reservoir.utils.metrics import calculate_chaos_metrics
from reservoir.utils.reporting import print_chaos_metrics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reservoir.data.config import BaseDatasetConfig
    from reservoir.layers.preprocessing import Preprocessor

class Evaluator:
    """Encapsulates evaluation logic including inverse transformation and metric calculation."""

    @staticmethod
    def compute_chaos_metrics(
        truth: JaxF64,
        pred: JaxF64,
        scaler: Preprocessor | None,
        dataset_config: BaseDatasetConfig,
        global_start: int,
        global_end: int,
        verbose: bool
    ) -> EvalMetrics | None:
        """Compute VPT, NDEI, and other chaos metrics with inverse transform."""
        if scaler is None:
            return None

        shape_pred = pred.shape
        shape_truth = truth.shape
        # Flatten -> Inverse Transform -> Reshape back
        pred_np = to_np_f64(pred)
        truth_np = to_np_f64(truth)
        pred_raw = scaler.inverse_transform(pred_np.reshape(-1, shape_pred[-1])).reshape(shape_pred)
        truth_raw = scaler.inverse_transform(truth_np.reshape(-1, shape_truth[-1])).reshape(shape_truth)

        if verbose:
            print(f"\n[evaluation.py] [Closed-Loop Metrics] (Global Steps {global_start} -> {global_end})")

        dt = float(getattr(dataset_config, 'dt', 1.0))
        ltu = float(getattr(dataset_config, 'lyapunov_time_unit', 1.0))

        # Flatten to 2D (Time, Features) for metric calculation
        # If shape was (1, T, F), it becomes (T, F). If (T, F), stays (T, F).
        pred_raw_2d = pred_raw.reshape(-1, shape_pred[-1])
        truth_raw_2d = truth_raw.reshape(-1, shape_truth[-1])

        metrics = calculate_chaos_metrics(truth_raw_2d, pred_raw_2d, dt=dt, lyapunov_time_unit=ltu)
        if verbose:
            print_chaos_metrics(metrics)
        return metrics

    @staticmethod
    def align_targets(features: JaxF64 | None, targets: JaxF64 | None) -> JaxF64 | None:
        """Align target length (dim 0) to match feature length (warmup handling)."""
        if features is None or targets is None:
            return targets

        # 2D only: length is shape[0]
        len_f = features.shape[0]
        len_t = targets.shape[0]

        if len_f < len_t:
            diff = len_t - len_f
            return targets[diff:]
        return targets
