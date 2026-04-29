"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/components/reporter.py"""
import time
from typing import cast, TYPE_CHECKING

from reservoir.utils.reporting import generate_report
from reservoir.core.types import ArrayResult, ResultDict, FitResultDict, FitResultMetrics, TrainMetrics, TestMetrics, EvalMetrics, to_np_f64, NpF64

if TYPE_CHECKING:
    from reservoir.core.types import JaxF64
    from reservoir.pipelines.config import DatasetMetadata, FrontendContext, ModelStack
    from reservoir.models.presets import PipelineConfig


def _metric_value(metrics: EvalMetrics, metric_name: str) -> float | None:
    if metric_name == "mse":
        return metrics.get("mse")
    if metric_name == "mae":
        return metrics.get("mae")
    if metric_name == "accuracy":
        return metrics.get("accuracy")
    if metric_name == "precision":
        return metrics.get("precision")
    if metric_name == "recall":
        return metrics.get("recall")
    if metric_name == "f1":
        return metrics.get("f1")
    if metric_name == "nmse":
        return metrics.get("nmse")
    if metric_name == "nrmse":
        return metrics.get("nrmse")
    if metric_name == "mase":
        return metrics.get("mase")
    if metric_name == "ndei":
        return metrics.get("ndei")
    if metric_name == "var_ratio":
        return metrics.get("var_ratio")
    if metric_name == "correlation":
        return metrics.get("correlation")
    if metric_name == "vpt_steps":
        return metrics.get("vpt_steps")
    if metric_name == "vpt_lt":
        return metrics.get("vpt_lt")
    if metric_name == "vpt_threshold":
        return metrics.get("vpt_threshold")
    return None


def _set_eval_metric(metrics: EvalMetrics, metric_name: str, value: float) -> None:
    if metric_name == "mse":
        metrics["mse"] = value
    elif metric_name == "mae":
        metrics["mae"] = value
    elif metric_name == "accuracy":
        metrics["accuracy"] = value
    elif metric_name == "precision":
        metrics["precision"] = value
    elif metric_name == "recall":
        metrics["recall"] = value
    elif metric_name == "f1":
        metrics["f1"] = value
    elif metric_name == "nmse":
        metrics["nmse"] = value
    elif metric_name == "nrmse":
        metrics["nrmse"] = value
    elif metric_name == "mase":
        metrics["mase"] = value
    elif metric_name == "ndei":
        metrics["ndei"] = value
    elif metric_name == "var_ratio":
        metrics["var_ratio"] = value
    elif metric_name == "correlation":
        metrics["correlation"] = value
    elif metric_name == "vpt_steps":
        metrics["vpt_steps"] = value
    elif metric_name == "vpt_lt":
        metrics["vpt_lt"] = value
    elif metric_name == "vpt_threshold":
        metrics["vpt_threshold"] = value


class ResultReporter:
    """
    Handles result aggregation, metric calculation, and report generation.
    """

    def __init__(self, stack: ModelStack, frontend_ctx: FrontendContext, dataset_meta: DatasetMetadata):
        self.stack = stack
        self.frontend_ctx = frontend_ctx
        self.dataset_meta = dataset_meta
        self.start_time = time.time()

    def compile_and_save(self, execution_results: ResultDict, config: PipelineConfig) -> ResultDict:
        """
        Compile final results and trigger report generation.
        """
        fit_result: FitResultDict = execution_results["fit_result"]
        train_logs = execution_results["train_logs"]
        quantum_trace = execution_results.get("quantum_trace") # New
        processed = self.frontend_ctx.processed_split
        
        results: ResultDict = {}
        metric_name = self.stack.metric
        test_y = processed.test_y

        # Use aligned_test_y from fit_result if available (for FNN windowed mode)
        aligned_test_y = fit_result.get("aligned_test_y", test_y)

        def _safe_to_np(val: ArrayResult | None) -> NpF64 | None:
            if val is None:
                return None
            if hasattr(val, "block_until_ready") or hasattr(val, "device_buffer"):
                return to_np_f64(cast("JaxF64", val))
            return cast("NpF64", val)

        if fit_result["closed_loop_pred"] is not None:
            # Predictions from strategies might be JaxF64, convert to NpF64 for reporting
            test_pred = _safe_to_np(fit_result["closed_loop_pred"])
            
            _safe_to_np(fit_result["closed_loop_truth"])
            results["is_closed_loop"] = True
        else:
            test_pred = _safe_to_np(fit_result.get("test_pred"))
            _safe_to_np(aligned_test_y)

        # Try to use pre-calculated metrics from Strategy
        metrics: FitResultMetrics
        metrics_raw = fit_result.get("metrics")
        if metrics_raw is None:
            empty_metrics: FitResultMetrics = {}
            metrics = empty_metrics
        else:
            metrics = metrics_raw

        # Test Score
        test_score = 0.0
        test_metrics_raw = metrics.get("test")
        if test_metrics_raw is None:
            test_metrics: TestMetrics = {}
        else:
            test_metrics = cast("TestMetrics", test_metrics_raw)
        metric_value = _metric_value(test_metrics, metric_name)
        if metric_value is not None:
             test_score = metric_value

        # Train Score 
        train_metrics_raw = metrics.get("train")
        if train_metrics_raw is None:
            empty_train_metrics: EvalMetrics = {}
            train_metrics_from_strat = empty_train_metrics
        else:
            train_metrics_from_strat = train_metrics_raw
        train_result_metrics: TrainMetrics = {
            "search_history": fit_result["search_history"],
            "weight_norms": fit_result["weight_norms"],
            **train_metrics_from_strat
        }
        results["train"] = train_result_metrics
        if fit_result["best_lambda"] is not None:
            results["train"]["best_lambda"] = fit_result["best_lambda"]
        
        # Propagate residuals history for plotting
        if "residuals_history" in fit_result:
            results["residuals_history"] = fit_result["residuals_history"]

        test_payload: TestMetrics = {**test_metrics}
        _set_eval_metric(test_payload, metric_name, test_score)
        results["test"] = test_payload
        chaos_raw = fit_result.get("chaos_results")
        if chaos_raw is None:
            chaos: dict[str, float] = {}
        else:
            chaos = chaos_raw
        if chaos:
            results["test"]["chaos_metrics"] = chaos
            results["test"]["vpt_lt"] = chaos.get("vpt_lt", 0.0)
            results["test"]["ndei"] = chaos.get("ndei", float("inf"))
            results["test"]["var_ratio"] = chaos.get("var_ratio", 0.0)
            results["test"]["mse"] = chaos.get("mse", float("inf"))

        # Val Score
        val_score = 0.0
        val_metrics_raw = metrics.get("val")
        if val_metrics_raw is None:
            empty_val_metrics: EvalMetrics = {}
            val_metrics = empty_val_metrics
        else:
            val_metrics = val_metrics_raw
        val_metric_value = _metric_value(val_metrics, metric_name)
        if val_metric_value is not None:
            val_score = val_metric_value
        best_score = fit_result.get("best_score")
        if val_metric_value is None and best_score is not None:
             # Keep this fallback as best_score corresponds to validation during fit
             val_score = best_score

        validation_payload: EvalMetrics = {**val_metrics}
        _set_eval_metric(validation_payload, metric_name, val_score)
        results["validation"] = validation_payload

        # Ensure all predictions and outputs are moved to Host Domain (NpF64)
        def _to_np_recursive(val: dict[str, ArrayResult | None]) -> dict[str, NpF64 | None]:
            converted: dict[str, NpF64 | None] = {}
            for key, item in val.items():
                converted[key] = _safe_to_np(item)
            return converted

        outputs_raw_from_result = fit_result.get("outputs") # strategy might have returned them
        if outputs_raw_from_result is None:
            outputs_raw: dict[str, ArrayResult | None] = {}
        else:
            outputs_raw = outputs_raw_from_result
        if not outputs_raw:
             # Fallback: strategy returned them directly in fit_result keys
             outputs_raw = {
                 "train_pred": fit_result.get("train_pred"),
                 "test_pred": test_pred, # already converted above
                 "val_pred": fit_result.get("val_pred"),
             }

        results["outputs"] = _to_np_recursive(outputs_raw)

        results["readout"] = self.stack.readout
        results["preprocessor"] = self.frontend_ctx.preprocessor
        results["scaler"] = self.frontend_ctx.preprocessor  # Alias for reporting.py
        results["training_logs"] = train_logs
        results["quantum_trace"] = _safe_to_np(quantum_trace)
        results["meta"] = {
            "metric": metric_name,
            "elapsed_sec": time.time() - self.start_time,
        }
        
        # Trigger Report Generation
        self._generate_report(results, config)

        return results

    def _generate_report(self, results: ResultDict, config: PipelineConfig):
        processed = self.frontend_ctx.processed_split
        report_payload = dict(
            readout=self.stack.readout,
            train_y=processed.train_y,
            test_y=processed.test_y,
            val_y=processed.val_y,
            training_obj=self.dataset_meta.training,
            dataset_meta=self.dataset_meta,
            model_type_str=self.stack.model_label,
        )
        generate_report(
            results,
            config,
            self.stack.topo_meta,
            **report_payload,
            classification=self.dataset_meta.classification,
            dataset_preset=self.dataset_meta.preset,
            model_obj=self.stack.model,
        )
