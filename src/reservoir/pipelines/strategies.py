"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/strategies.py"""
from __future__ import annotations

from abc import ABC, abstractmethod
from reservoir.core.types import NpF64, to_np_f64, to_jax_f64, FitResultDict, JaxF64, ModelState
import jax
import jax.numpy as jnp
from typing import cast, TYPE_CHECKING
import numpy as np


from reservoir.utils.batched_compute import batched_compute
from reservoir.utils.reporting import print_chaos_metrics, print_ridge_search_results, plot_ridgecv_intermediates, print_feature_stats
from reservoir.utils.metrics import compute_score, calculate_chaos_metrics

if TYPE_CHECKING:
    from collections.abc import Callable
    from reservoir.models.generative import Predictable
    from reservoir.models.presets import PipelineConfig
    from reservoir.pipelines.config import FrontendContext, DatasetMetadata
    from reservoir.readout.base import ReadoutModule
    from reservoir.pipelines.evaluation import Evaluator
    from reservoir.models.generative import ClosedLoopGenerativeModel, ClosedLoopModel
    from reservoir.core.types import TopologyMeta


def _check_closed_loop_divergence(pred_std: float, threshold: float) -> None:
    """Raise ValueError if closed-loop prediction has diverged."""
    if pred_std > threshold:
        raise ValueError(f"Closed-loop prediction diverged! STD={pred_std:.2f} > {threshold}")


class DivergenceError(ValueError):
    """Raised when closed-loop prediction diverges, carrying diagnostic stats."""
    def __init__(self, message: str, stats: dict[str, float]) -> None:
        super().__init__(message)
        self.stats: dict[str, float] = stats


_RIDGE_FEATURE_BATCH_BUDGET_BYTES = 128 * 1024 * 1024
_RIDGE_SOLVE_BATCH_BUDGET_BYTES = 512 * 1024 * 1024


def _augment_design_matrix(features: JaxF64, use_intercept: bool) -> JaxF64:
    """Append an intercept column when requested."""
    if not use_intercept:
        return features
    ones = jnp.ones((features.shape[0], 1), dtype=features.dtype)
    return jnp.concatenate([ones, features], axis=1)


def _resolve_ridge_feature_batch_size(
    train_Z: JaxF64,
    batch_size: int,
    use_intercept: bool,
    feature_mapper: Callable[[JaxF64], JaxF64] | None,
) -> tuple[int, int]:
    """Estimate a safe sample chunk size for feature expansion on GPU."""
    if train_Z.shape[0] == 0:
        return max(1, batch_size), int(train_Z.shape[1])

    sample_features = train_Z[:1]
    if feature_mapper is not None:
        sample_features = feature_mapper(sample_features)

    expanded_dim = int(sample_features.shape[1])
    design_dim = expanded_dim + int(use_intercept)
    bytes_per_sample = max(1, design_dim * np.dtype(np.float64).itemsize)
    budget_limited = max(1, _RIDGE_FEATURE_BATCH_BUDGET_BYTES // bytes_per_sample)
    feature_batch_size = max(1, min(int(batch_size), int(budget_limited), int(train_Z.shape[0])))
    return feature_batch_size, expanded_dim


def _resolve_ridge_lambda_chunk_size(n_features: int, n_candidates: int) -> int:
    """Limit how many lambda systems are solved at once for large feature spaces."""
    matrix_bytes = max(1, n_features * n_features * np.dtype(np.float64).itemsize)
    chunk_size = max(1, _RIDGE_SOLVE_BATCH_BUDGET_BYTES // matrix_bytes)
    return max(1, min(int(n_candidates), int(chunk_size)))


def _accumulate_ridge_normal_equations(
    train_Z: JaxF64,
    train_y: JaxF64,
    use_intercept: bool,
    feature_mapper: Callable[[JaxF64], JaxF64] | None,
    sample_batch_size: int,
    expanded_dim: int,
) -> tuple[JaxF64, JaxF64]:
    """Build XtX and Xty without materializing the full expanded training matrix."""
    y_train_mat = train_y if train_y.ndim > 1 else train_y[:, None]
    n_targets = int(y_train_mat.shape[1])

    n_features = expanded_dim + int(use_intercept)
    pad_size = (-int(train_Z.shape[0])) % sample_batch_size
    if pad_size > 0:
        x_pad = [(0, pad_size)] + [(0, 0)] * (train_Z.ndim - 1)
        y_pad = [(0, pad_size)] + [(0, 0)] * (y_train_mat.ndim - 1)
        train_Z = jnp.pad(train_Z, tuple(x_pad))
        y_train_mat = jnp.pad(y_train_mat, tuple(y_pad))

    n_chunks = max(1, int(train_Z.shape[0]) // sample_batch_size)
    chunked_train_Z = train_Z.reshape((n_chunks, sample_batch_size, *train_Z.shape[1:]))
    chunked_train_y = y_train_mat.reshape((n_chunks, sample_batch_size, *y_train_mat.shape[1:]))

    XtX0 = jnp.zeros((n_features, n_features), dtype=train_Z.dtype)
    Xty0 = jnp.zeros((n_features, n_targets), dtype=train_Z.dtype)

    @jax.jit
    def scan_accumulate(x_chunks: JaxF64, y_chunks: JaxF64) -> tuple[JaxF64, JaxF64]:
        def scan_step(carry: tuple[JaxF64, JaxF64], batch: tuple[JaxF64, JaxF64]) -> tuple[tuple[JaxF64, JaxF64], None]:
            xtx_acc, xty_acc = carry
            x_batch, y_batch = batch
            mapped_batch = feature_mapper(x_batch) if feature_mapper is not None else x_batch
            x_aug = _augment_design_matrix(mapped_batch, use_intercept)
            next_carry = (xtx_acc + (x_aug.T @ x_aug), xty_acc + (x_aug.T @ y_batch))
            return next_carry, None

        (XtX, Xty), _ = jax.lax.scan(scan_step, (XtX0, Xty0), (x_chunks, y_chunks))
        return XtX, Xty

    return scan_accumulate(chunked_train_Z, chunked_train_y)


def optimize_ridge_vmap(
    lambda_candidates: tuple[float, ...],
    use_intercept: bool,
    train_Z: JaxF64,
    train_y: JaxF64,
    val_Z: JaxF64,
    val_y: NpF64,
    metric_name: str,
    batch_size: int,
    inverse_fn: Callable[[NpF64], NpF64] | None = None,
    norm_threshold: float | None = None,
    feature_mapper: Callable[[JaxF64], JaxF64] | None = None,
) -> tuple[float, float, dict[float, float], dict[float, float], NpF64, JaxF64, dict[float, NpF64] | None]:
    """
    Common vectorized RidgeCV optimization logic using batched_compute.
    
    Returns:
        best_lambda
        best_score
        search_history
        weight_norms
        best_val_pred_np (prediction for best lambda)
        all_weights (all candidate weights)
        residuals_history (optional, for regression analysis)
    """
    # 1. Precompute XtX and Xty without materializing the full expanded train design.
    y_train_mat = train_y
    if y_train_mat.ndim == 1:
        y_train_mat = y_train_mat[:, None]

    feature_batch_size, expanded_dim = _resolve_ridge_feature_batch_size(
        train_Z=train_Z,
        batch_size=batch_size,
        use_intercept=use_intercept,
        feature_mapper=feature_mapper,
    )
    if feature_mapper is not None:
        print(
            f"[strategies.py] RidgeCV normal equations: chunked feature expansion "
            f"(expanded_dim={expanded_dim}, chunk_size={feature_batch_size})."
        )

    XtX, Xty = _accumulate_ridge_normal_equations(
        train_Z=train_Z,
        train_y=y_train_mat,
        use_intercept=use_intercept,
        feature_mapper=feature_mapper,
        sample_batch_size=feature_batch_size,
        expanded_dim=expanded_dim,
    )

    mapped_val_Z = feature_mapper(val_Z) if feature_mapper is not None else val_Z
    val_Z_aug = _augment_design_matrix(mapped_val_Z, use_intercept)
    n_features = XtX.shape[0]
    identity = jnp.eye(n_features, dtype=XtX.dtype)

    # 2. Solve ridge systems, chunking lambda batches for large feature spaces.
    @jax.jit
    def solve_ridge_vectorized(lams: JaxF64) -> JaxF64:
        def solve_one(lam_l: float) -> JaxF64:
            return jax.scipy.linalg.solve(XtX + lam_l * identity, Xty)
        return jax.vmap(solve_one)(lams)

    lambdas_jax = jnp.array(lambda_candidates)
    lambda_chunk_size = _resolve_ridge_lambda_chunk_size(int(n_features), len(lambda_candidates))
    if lambda_chunk_size < len(lambda_candidates):
        print(
            f"[strategies.py] RidgeCV solve: splitting {len(lambda_candidates)} lambdas "
            f"into chunks of {lambda_chunk_size} for {int(n_features)} features."
        )

    weight_chunks: list[JaxF64] = []
    for start in range(0, len(lambda_candidates), lambda_chunk_size):
        stop = min(start + lambda_chunk_size, len(lambda_candidates))
        weight_chunks.append(solve_ridge_vectorized(lambdas_jax[start:stop]))

    all_weights = jnp.concatenate(weight_chunks, axis=0) # (N_lam, N_feat, N_out)

    # 3. Batched validation prediction over lambda chunks.
    def predict_batch_fn(weights_batch: JaxF64) -> JaxF64:
        return jnp.einsum("sf,bfo->bso", val_Z_aug, weights_batch)

    # Convert weights to numpy to pass to batched_compute (avoids beartype error)
    all_weights_np = to_np_f64(all_weights)
    
    all_val_preds_np = batched_compute(
        predict_batch_fn,
        all_weights_np,
        batch_size=lambda_chunk_size,
        desc="RidgeCV Search",
        file="strategies.py",
        step="7"
    )
    # 4. Score on CPU
    search_history: dict[float, float] = {}
    weight_norms: dict[float, float] = {}
    residuals_history: dict[float, NpF64] = {}
    
    # Determine optimization direction
    minimize = metric_name.lower() in ["nmse", "mse", "rmse", "nrmse", "mase"]
    
    abs_best_score = float('inf') if minimize else -float('inf')
    best_lambda = lambda_candidates[0]
    best_score_val = abs_best_score
    best_pred_idx = 0
    
    found_stable = False

    for i, lam in enumerate(lambda_candidates):
        lam_val = float(lam)
        vp_np = all_val_preds_np[i]
        
        # Apply inverse transform if provided (for regression metrics)
        p_eval = inverse_fn(vp_np) if inverse_fn else vp_np
        t_eval = inverse_fn(val_y) if inverse_fn else val_y
        
        score = compute_score(p_eval, t_eval, metric_name)
        search_history[lam_val] = float(score)
        
        w_coef = all_weights_np[i, 1:] if use_intercept else all_weights_np[i]
        norm = float(jnp.linalg.norm(w_coef))
        weight_norms[lam_val] = norm
        
        # Always save residuals for analysis (BoxPlot)
        energy = float(np.mean(t_eval**2))
        res_sq = (p_eval.ravel() - t_eval.ravel()) ** 2 / (energy + 1e-12)
        residuals_history[lam_val] = res_sq

        # Robust argmin with optional stability constraint (None => no cap)
        within_threshold = norm_threshold is None or norm <= norm_threshold
        if within_threshold:
            is_better = (score < abs_best_score) if minimize else (score > abs_best_score)
            if is_better:
                abs_best_score = score
                best_lambda = lam_val
                best_score_val = score
                best_pred_idx = i
                found_stable = True
                
    # Fallback if no candidate met the norm threshold
    if not found_stable:
        # Pick the largest lambda (last index) as it generally has the smallest norm
        best_pred_idx = len(lambda_candidates) - 1
        best_lambda = float(lambda_candidates[best_pred_idx])
        best_score_val = search_history[best_lambda]
        abs_best_score = best_score_val
        print(f"    [Warning] No RidgeCV candidate met the Norm <= {norm_threshold} threshold! Falling back to max lambda {best_lambda:.2e} (Norm: {weight_norms[best_lambda]:.2f})")
            
    best_val_pred_np = all_val_preds_np[best_pred_idx]
    
    print(f"[strategy.py] optimize_ridge_vmap best_idx={best_pred_idx}, best_lambda={best_lambda:.2e}, score={best_score_val:.8e} (abs_min_stable={abs_best_score:.8e})")
    print_feature_stats(best_val_pred_np, "strategies.py",":best_val_pred_np")
    
    return best_lambda, best_score_val, search_history, weight_norms, best_val_pred_np, all_weights, (residuals_history if residuals_history else None)


class ReadoutStrategy(ABC):
    """Abstract base class for readout fitting and evaluation strategies."""
    
    def __init__(self, evaluator: Evaluator, metric_name: str):
        self.evaluator = evaluator
        self.metric_name = metric_name

    @staticmethod
    def _flatten_3d_to_2d(arr: NpF64 | None, label: str = "array") -> NpF64 | None:
        """Flatten 3D states (Batch, Time, Features) -> 2D (Batch, Features)."""
        if arr is None:
            return None
        if arr.ndim == 3:
            print(f"[strategy.py] Flattening 3D {label} {arr.shape} -> 2D")
            return arr.reshape(arr.shape[0], -1)
        return arr

    @staticmethod
    def _get_seed_sequence(train_X: NpF64, val_X: NpF64 | None) -> JaxF64:
        """Prepare seed for closed-loop (concat train+val)."""
        if val_X is not None:
            axis = 1 if train_X.ndim == 3 else 0
            # Transition to Device domain
            return jnp.concatenate([to_jax_f64(train_X), to_jax_f64(val_X)], axis=axis)
        return to_jax_f64(train_X)

    @abstractmethod
    def fit_and_evaluate(
        self,
        model: ClosedLoopModel,
        readout: ReadoutModule | None,
        train_Z: NpF64,
        val_Z: NpF64 | None,
        test_Z: NpF64 | None,
        train_y: NpF64 | None,
        val_y: NpF64 | None,
        test_y: NpF64 | None,
        frontend_ctx: FrontendContext,
        dataset_meta: DatasetMetadata,
        pipeline_config: PipelineConfig,
        topo_meta: TopologyMeta,
        val_final_state: tuple[ModelState | None, JaxF64 | None] | None = None, # New Argument
    ) -> FitResultDict:
        """Fit readout and return predictions/metrics."""

    def _optimize_and_plot_ridge(
        self,
        readout: ReadoutModule,
        train_Z: NpF64,
        val_Z: NpF64,
        train_y: NpF64,
        val_y: NpF64,
        frontend_ctx: FrontendContext,
        dataset_meta: DatasetMetadata,
        pipeline_config: PipelineConfig,
        topo_meta: TopologyMeta,
        inverse_fn: Callable[[NpF64], NpF64] | None = None,
        metric_name: str = "NMSE",
    ) -> tuple[float | None, float | None, dict[float, float], dict[float, float], dict[float, NpF64] | None, NpF64 | None]:
        """
        Shared logic for RidgeCV optimization, model updating, and intermediate plotting.
        Returns: best_lambda, best_score, search_history, weight_norms, residuals_history, val_pred_np
        """
        from reservoir.readout.ridge import RidgeCV, RidgeRegression
        if not isinstance(readout, RidgeCV):
            print("[strategies.py] No hyperparameter search needed for this readout.")
            readout.fit(to_jax_f64(train_Z), to_jax_f64(train_y))
            val_pred_jax = readout.predict(to_jax_f64(val_Z))
            val_pred_np = to_np_f64(val_pred_jax) if val_pred_jax is not None else None
            return None, None, {}, {}, None, val_pred_np

        print(f"[strategies.py] Optimizing RidgeCV ({metric_name}) over {len(readout.lambda_candidates)} candidates (JAX Vectorized)...")
        
        # Prepare JAX inputs
        X_jax = to_jax_f64(train_Z)
        y_jax = to_jax_f64(train_y)
        if y_jax.ndim == 1:
            y_jax = y_jax[:, None]
        val_X_jax = to_jax_f64(val_Z)
        feature_mapper = getattr(readout, "map_features", None)
        
        batch_size_cfg = int(dataset_meta.training.batch_size) if dataset_meta.training and dataset_meta.training.batch_size else 32
        
        best_lambda, best_score, search_history, weight_norms, best_val_pred_np, all_weights, residuals_history = optimize_ridge_vmap(
            lambda_candidates=readout.lambda_candidates,
            use_intercept=readout.use_intercept,
            train_Z=X_jax,
            train_y=y_jax,
            val_Z=val_X_jax,
            val_y=val_y,
            metric_name=metric_name.lower(),
            inverse_fn=inverse_fn,
            batch_size=batch_size_cfg,
            norm_threshold=getattr(readout, "norm_threshold", 100.0),
            feature_mapper=feature_mapper,
        )
        
        print(f"[strategies.py] Best Lambda: {best_lambda:.5e} (Score: {best_score:.5f})")
        
        # Re-instantiate best model
        best_idx = list(readout.lambda_candidates).index(best_lambda)
        readout.best_model = RidgeRegression(ridge_lambda=best_lambda, use_intercept=readout.use_intercept)
        
        if readout.use_intercept:
            readout.best_model.intercept_ = all_weights[best_idx, 0].ravel()
            readout.best_model.coef_ = all_weights[best_idx, 1:]
        else:
            readout.best_model.intercept_ = jnp.zeros(all_weights.shape[-1])
            readout.best_model.coef_ = all_weights[best_idx]
        
        print_ridge_search_results(cast("FitResultDict", {
            "search_history": search_history,
            "best_lambda": best_lambda,
            "weight_norms": weight_norms,
            "residuals_history": residuals_history
        }), metric_name=metric_name)
        
        # Immediate Plotting
        try:
            plot_ridgecv_intermediates(
                residuals_hist=residuals_history,
                weight_norms=weight_norms,
                best_lambda=best_lambda,
                best_score=best_score,
                val_y=val_y,
                val_pred_np=best_val_pred_np,
                frontend_ctx=frontend_ctx,
                topo_meta=topo_meta,
                pipeline_config=pipeline_config,
                dataset_meta=dataset_meta,
                model_type_str=str(pipeline_config.model_type.value),
                readout=readout,
                metric_name=metric_name,
            )
        except (ImportError, RuntimeError, ValueError, OSError, TypeError, AttributeError) as e:
            print(f"[Warning] Failed to generate intermediate plots: {e}")
            
        return best_lambda, best_score, search_history, weight_norms, residuals_history, best_val_pred_np


class EndToEndStrategy(ReadoutStrategy):
    """Strategy for End-to-End models where features are predictions."""
    
    def fit_and_evaluate(
        self, model: ClosedLoopModel, 
        readout: ReadoutModule | None, 
        train_Z: NpF64, 
        val_Z: NpF64 | None, 
        test_Z: NpF64 | None, 
        train_y: NpF64 | None,
        val_y: NpF64 | None,
        test_y: NpF64 | None,
        frontend_ctx: FrontendContext, 
        dataset_meta: DatasetMetadata, 
        pipeline_config: PipelineConfig,
        topo_meta: TopologyMeta,
        val_final_state: tuple[ModelState | None, JaxF64 | None] | None = None,
    ) -> FitResultDict:
        print("Readout is None. End-to-End mode.")
        
        # For FNN windowed mode, align targets using the model's adapter
        aligned_test_y = test_y

        if hasattr(model, 'adapter') and test_y is not None:
            aligned_test_y = model.adapter.align_targets(test_y)

        result: FitResultDict = {
            "train_pred": train_Z,
            "val_pred": val_Z,
            "test_pred": test_Z,
            "aligned_test_y": aligned_test_y,  # Pass aligned y for scoring
            "best_lambda": None,
            "best_score": None,
            "search_history": {},
            "weight_norms": {},
            "closed_loop_pred": None,
            "closed_loop_truth": None,
            "chaos_results": None,
            "metrics": {},
        }

        # FNN Closed-Loop for regression check
        if not dataset_meta.classification and hasattr(model, 'window_size') and model.window_size is not None:
             # Delegate to helper if needed, copying logic from legacy Runner
             # Logic from _strategy_end_to_end line 150+
             print("\n=== Step 8: FNN Closed-Loop Generation ===")
             try:
                processed = frontend_ctx.processed_split
                generation_steps = processed.test_X.shape[1] if hasattr(processed.test_X, "shape") else 0
                if processed.test_X.ndim == 2:
                    generation_steps = processed.test_X.shape[0]

                seed_data = self._get_seed_sequence(processed.train_X, processed.val_X)
                # For E2E, readout is None or implicit, pass explicit None if needed, but signature says readout
                # EndToEnd typically has readout=None.
                readout_cast = cast("Predictable", readout)
                closed_loop_pred = cast(
                    "JaxF64",
                    model.generate_closed_loop(seed_data, steps=generation_steps, readout=readout_cast),
                )
                
                # Check for divergence
                pred_std = float(np.std(closed_loop_pred))
                _check_closed_loop_divergence(pred_std, threshold=50)

                print_feature_stats(to_np_f64(closed_loop_pred), "strategies.py", "8:fnn_closed_loop_prediction")

                global_start = processed.train_X.shape[1] + (processed.val_X.shape[1] if processed.val_X is not None else 0)
                global_end = global_start + generation_steps

                chaos_results = self.evaluator.compute_chaos_metrics(
                    jnp.array(processed.test_y), jnp.array(closed_loop_pred), frontend_ctx.preprocessor,
                    dataset_meta.preset.config, global_start, global_end, verbose=False)

                result["closed_loop_pred"] = closed_loop_pred
                result["closed_loop_truth"] = processed.test_y
                result["chaos_results"] = cast("dict[str, float]", chaos_results)
             except (ValueError, RuntimeError) as e:
                print(f"[Warning] FNN Closed-loop generation failed: {e}")
        
        return result



class ClassificationStrategy(ReadoutStrategy):
    """Open-Loop classification strategy with Accuracy optimization."""
    
    def fit_and_evaluate(
        self, model: ClosedLoopModel, readout: ReadoutModule | None, train_Z: NpF64, val_Z: NpF64 | None, test_Z: NpF64 | None, train_y: NpF64 | None, val_y: NpF64 | None, test_y: NpF64 | None, frontend_ctx: FrontendContext, dataset_meta: DatasetMetadata, pipeline_config: PipelineConfig,
        topo_meta: TopologyMeta,
        val_final_state: tuple[ModelState | None, JaxF64 | None] | None = None
    ) -> FitResultDict:
        print("[strategies.py] Classification task: Using Open-Loop evaluation.")
        if readout is None:
            raise ValueError("Readout must be provided for ClassificationStrategy")

        tf_reshaped = self._flatten_3d_to_2d(train_Z, "train states")
        vf_reshaped = self._flatten_3d_to_2d(val_Z, "val states")
        test_Z = self._flatten_3d_to_2d(test_Z, "test states")
        ty_reshaped = train_y

        if tf_reshaped is None or ty_reshaped is None:
            raise ValueError("train_Z and train_y must not be None for ClassificationStrategy")

        if vf_reshaped is None or val_y is None:
            raise ValueError("Validation data required for ClassificationStrategy readout optimization")

        best_lambda, best_score, search_history, weight_norms, _, val_pred_np = self._optimize_and_plot_ridge(
            readout=readout,
            train_Z=tf_reshaped,
            val_Z=vf_reshaped,
            train_y=ty_reshaped,
            val_y=val_y,
            frontend_ctx=frontend_ctx,
            dataset_meta=dataset_meta,
            pipeline_config=pipeline_config,
            topo_meta=topo_meta,
            inverse_fn=None,
            metric_name="Accuracy"
        )
        val_pred = to_jax_f64(val_pred_np) if val_pred_np is not None else None

        print("\n=== Step 8: Final Predictions (Classification):===")

        # Helper for batched prediction on Train/Test
        def predict_model_batch(x_batch: JaxF64) -> JaxF64:
            return readout.predict(x_batch)

        # Train Prediction (Batched)
        train_pred_np = batched_compute(
            predict_model_batch,
            train_Z,
            batch_size=32,
            desc="Train Pred",
            file="strategies.py",
            step="8"
        )
        train_pred = to_jax_f64(train_pred_np)

        # Test Prediction (Batched)
        test_pred = None
        test_pred_np = None
        if test_Z is not None:
            test_pred_np = batched_compute(
                predict_model_batch,
                test_Z,
                batch_size=32,
                desc="Test Pred",
                file="strategies.py",
                step="8"
            )
            test_pred = to_jax_f64(test_pred_np)

        # Train
        if train_y is None:
            raise ValueError("train_y must not be None for ClassificationStrategy")
        metrics = {"train": {
            self.metric_name: compute_score(train_pred_np, train_y, self.metric_name)
        }}
        

        # Val
        if val_pred_np is not None and val_y is not None:
             metrics["val"] = {
                 self.metric_name: compute_score(val_pred_np, val_y, self.metric_name)
             }
             
        # Test
        if test_pred_np is not None and test_y is not None:
             metrics["test"] = {
                 self.metric_name: compute_score(test_pred_np, test_y, self.metric_name)
             }

        return cast("FitResultDict", {
            "train_pred": train_pred,
            "val_pred": val_pred,
            "test_pred": test_pred,
            "metrics": metrics,
            "best_lambda": best_lambda,
            "best_score": best_score,
            "search_history": search_history,
            "weight_norms": weight_norms,
            "closed_loop_pred": None,
            "closed_loop_truth": None,
            "chaos_results": None,
        })


class ClosedLoopRegressionStrategy(ReadoutStrategy):
    """Closed-Loop regression strategy (VPT optimization)."""

    def fit_and_evaluate(
        self,
            model: ClosedLoopModel,
            readout: ReadoutModule | None,
            train_Z: NpF64,
            val_Z: NpF64 | None,
            test_Z: NpF64 | None,
            train_y: NpF64 | None,
            val_y: NpF64 | None,
            test_y: NpF64 | None,
            frontend_ctx: FrontendContext,
            dataset_meta: DatasetMetadata,
            pipeline_config: PipelineConfig,
            topo_meta: TopologyMeta,
            val_final_state: tuple[ModelState | None, JaxF64 | None] | None = None
    ) -> FitResultDict:
        if readout is None:
            raise ValueError("Readout must be provided for ClosedLoopRegressionStrategy")

        metrics: dict[str, dict[str, float]] = {}

        proj_fn: Callable[[JaxF64], JaxF64] | None = None
        # Check pipeline_config for projection, not dataset_meta
        if pipeline_config.projection is not None and frontend_ctx.projection_layer is not None:
            projection_layer = frontend_ctx.projection_layer

            def proj_fn(x: JaxF64) -> JaxF64:
                return projection_layer(x)

        tf_reshaped = self._flatten_3d_to_2d(train_Z, "train states")
        ty_reshaped = train_y
        
        if tf_reshaped is None or ty_reshaped is None:
            raise ValueError("train_Z and train_y must not be None for ClosedLoopRegressionStrategy")

        # Open-Loop Validation Prep
        vf_reshaped = self._flatten_3d_to_2d(val_Z, "val states")
        vy_reshaped = val_y

        if vf_reshaped is None or val_y is None:
            raise ValueError("Validation data required for ClosedLoopRegressionStrategy readout optimization")

        scaler = frontend_ctx.preprocessor
         
        def _inverse(arr: NpF64) -> NpF64:
            if scaler is None:
                return arr
            shape = arr.shape
            try:
                val = arr
                if val.ndim == 1:
                    val = val.reshape(-1, 1)
                inv = scaler.inverse_transform(val)
                return inv.reshape(shape)
            except (ValueError, TypeError):
                return arr

        best_lambda, best_score, search_history, weight_norms, residuals_history, val_pred_np = self._optimize_and_plot_ridge(
            readout=readout,
            train_Z=tf_reshaped,
            val_Z=vf_reshaped,
            train_y=ty_reshaped,
            val_y=val_y,
            frontend_ctx=frontend_ctx,
            dataset_meta=dataset_meta,
            pipeline_config=pipeline_config,
            topo_meta=topo_meta,
            inverse_fn=_inverse,
            metric_name="NMSE"
        )

        # Display Validation Metrics (Open Loop) immediately
        if val_pred_np is not None and val_y is not None:

            print("\n=== Validation Open Loop Metrics ===")
            dt = float(getattr(dataset_meta.preset.config, 'dt', 1.0))
            ltu = float(getattr(dataset_meta.preset.config, 'lyapunov_time_unit', 1.0))

            # _inverse is defined above inside the RidgeCV check,
            # we need a version available here too or move it.
            # I'll define it locally for safety if not defined.
            scaler = frontend_ctx.preprocessor
            def _inv_local(arr: NpF64) -> NpF64:
             if scaler is None:
                 return arr
             try:
                 val = arr
                 if val.ndim == 1:
                     val = val.reshape(-1, 1)
                 return scaler.inverse_transform(val).reshape(arr.shape)
             except (ValueError, TypeError):
                 return arr

            val_y_raw = _inv_local(val_y)
            val_pred_raw = _inv_local(val_pred_np)

            val_metrics_chaos = calculate_chaos_metrics(val_y_raw, val_pred_raw, dt=dt, lyapunov_time_unit=ltu)
            print_chaos_metrics(val_metrics_chaos)

            # Store validation chaos metrics
            if "val" not in metrics:
                metrics["val"] = dict[str, float]()
            mse = val_metrics_chaos.get("mse")
            mae = val_metrics_chaos.get("mae")
            nmse = val_metrics_chaos.get("nmse")
            nrmse = val_metrics_chaos.get("nrmse")
            mase = val_metrics_chaos.get("mase")
            ndei = val_metrics_chaos.get("ndei")
            var_ratio = val_metrics_chaos.get("var_ratio")
            correlation = val_metrics_chaos.get("correlation")
            vpt_steps = val_metrics_chaos.get("vpt_steps")
            vpt_lt_metric = val_metrics_chaos.get("vpt_lt")
            vpt_threshold = val_metrics_chaos.get("vpt_threshold")
            if mse is not None:
                metrics["val"]["mse"] = float(mse)
            if mae is not None:
                metrics["val"]["mae"] = float(mae)
            if nmse is not None:
                metrics["val"]["nmse"] = float(nmse)
            if nrmse is not None:
                metrics["val"]["nrmse"] = float(nrmse)
            if mase is not None:
                metrics["val"]["mase"] = float(mase)
            if ndei is not None:
                metrics["val"]["ndei"] = float(ndei)
            if var_ratio is not None:
                metrics["val"]["var_ratio"] = float(var_ratio)
            if correlation is not None:
                metrics["val"]["correlation"] = float(correlation)
            if vpt_steps is not None:
                metrics["val"]["vpt_steps"] = float(vpt_steps)
            if vpt_lt_metric is not None:
                metrics["val"]["vpt_lt"] = float(vpt_lt_metric)
            if vpt_threshold is not None:
                metrics["val"]["vpt_threshold"] = float(vpt_threshold)
            metrics["val"]["vpt_lt"] = val_metrics_chaos.get("vpt_lt", 0.0)

            if float(val_metrics_chaos.get("nmse", 0.0)) > 1e-5:
                raise ValueError(f"Validation NMSE too high: {val_metrics_chaos.get('nmse'):.10f}")

        # Test Generation ===============================================================================================================================
        print("\n=== Step 8: Final Predictions (Regression):===")
        closed_loop_pred = None
        closed_loop_truth = None
        chaos_results = None

        # Check model type for compatibility (simplified check)
        # Assuming "reservoir", "distillation", "passthrough" are all capable if they are passed here.
        # But we can safeguard.
        processed = frontend_ctx.processed_split
        if hasattr(processed.test_X, "shape"):
            if processed.test_X.ndim == 3:
                generation_steps = int(processed.test_X.shape[1])
            else:
                generation_steps = int(processed.test_X.shape[0])
        else:
            generation_steps = 0
        
        full_seed_data = self._get_seed_sequence(processed.train_X, processed.val_X)
        print(f"[strategies.py] Full Closed-Loop Test: Generating {generation_steps} steps.")
        
        # Prepare initial state if available to skip warmup
        init_state = None
        init_out = None
        if val_final_state is not None:
            init_state, init_out = val_final_state
            print("[strategies.py] Using captured Validation State to skip Warmup!")

        # seed_data is already JAX array from _get_seed_sequence
        readout_cast = cast("Predictable", readout)
        model_for_state = cast("ClosedLoopGenerativeModel[ModelState]", model)
        closed_loop_pred, closed_loop_hist = model_for_state.generate_closed_loop(
            full_seed_data, 
            steps=generation_steps, 
            readout=readout_cast, 
            projection_fn=proj_fn,
            initial_state=init_state,
            initial_output=init_out,
            return_history=True
        )
        print_feature_stats(to_np_f64(closed_loop_pred), "strategies.py", "8:closed_loop_prediction")

        closed_loop_truth = test_y
        if closed_loop_truth is not None:
            print_feature_stats(closed_loop_truth, "strategies.py", "8:closed_loop_truth")

        # Check for divergence
        pred_np = to_np_f64(closed_loop_pred)
        if test_y is None:
            raise ValueError("test_y is required for closed-loop regression statistics.")
        truth_np = test_y

        pred_std = np.std(pred_np)
        pred_max = np.max(pred_np)
        pred_min = np.min(pred_np)

        truth_std = np.std(truth_np)
        truth_max = np.max(truth_np)
        truth_min = np.min(truth_np)

        threshold = 1.5

        # Stats to return (No recalculation needed later)
        stats_dict = {
            "pred_mean": float(np.mean(pred_np)),
            "pred_std": float(pred_std),
            "pred_min": float(pred_min),
            "pred_max": float(pred_max),
            "truth_mean": float(np.mean(truth_np)),
            "truth_std": float(truth_std),
            "truth_min": float(truth_min),
            "truth_max": float(truth_max),
        }

        if pred_std > threshold * truth_std or truth_std > threshold * pred_std:
            print(f"    [Warning] Closed-loop prediction diverged! Pred STD={pred_std:.2f} > {threshold}x Truth STD={truth_std:.2f} (or collapsed)")
            # raise DivergenceError(
            #     f"Closed-loop prediction diverged! Pred STD={pred_std:.2f} > {threshold}x Truth STD={truth_std:.2f} (or collapsed)",
            #     stats=stats_dict,
            # )

        if pred_max > threshold + truth_max or truth_max > threshold + pred_max:
            print(f"    [Warning] Closed-loop prediction diverged! Pred Max={pred_max:.2f} > {threshold}x Truth Max={truth_max:.2f} (or collapsed)")
            # raise DivergenceError(
            #     f"Closed-loop prediction diverged! Pred Max={pred_max:.2f} > {threshold}x Truth Max={truth_max:.2f} (or collapsed)",
            #     stats=stats_dict,
            # )

        # Calculate global_start based on dimensions
        def get_time_steps(arr: NpF64 | None) -> int:
            if arr is None:
                return 0
            # If 3D (Batch, Time, Feat), return Time (shape[1])
            if arr.ndim == 3:
                return int(arr.shape[1])
            # If 2D (Time, Feat) or (Batch, Feat) - assuming Time for Series
            # For Reservoir time-series Time is usually axis 0 in 2D
            return int(arr.shape[0])

        train_steps = get_time_steps(processed.train_X)
        val_steps_count = get_time_steps(processed.val_X)
        global_start = train_steps + val_steps_count
        global_end = global_start + generation_steps
        
        if closed_loop_truth is not None:
            chaos_results = self.evaluator.compute_chaos_metrics(
                to_jax_f64(closed_loop_truth), closed_loop_pred, frontend_ctx.preprocessor,
                dataset_meta.preset.config, global_start, global_end, verbose=True
            )

        # Helper for batched prediction
        def predict_model_batch(x_batch: JaxF64) -> JaxF64:
            return readout.predict(x_batch)

        # Calculate Predictions (Open Loop) - Batched
        # Skip Train Prediction for Regression task to save time
        if train_y is not None:
             metrics["train"] = {self.metric_name: 0.0} # Placeholder or skip
        
        # Val (if needed, though Strategy optimized on it)
        # val_pred_np should already be set from RidgeCV or Else block above
        if val_pred_np is not None and val_y is not None:
            if "val" not in metrics:
                 metrics["val"] = dict[str, float]()
            metrics["val"][self.metric_name] = compute_score(val_pred_np, val_y, self.metric_name)

        # Test - Automatically skipped if test_Z is None (from executor.py)
        if test_Z is not None and readout is not None:
             # Use batched prediction for open-loop test prediction if needed for metrics
             test_p_np = batched_compute(
                predict_model_batch,
                test_Z,
                batch_size=2048,
                desc="Test Pred",
                file="strategies.py",
                step="8"
             )             
             if test_p_np is not None and test_y is not None:
                 metrics["test"] = {
                     self.metric_name: compute_score(test_p_np, test_y, self.metric_name)
                 }

        return cast("FitResultDict", {
            "train_pred": None, # Not returned by this strategy
            "val_pred": None,   # Not returned
            "test_pred": None,  # Not returned
            "metrics": metrics,
            "best_lambda": best_lambda,
            "best_score": best_score,
            "search_history": search_history,
            "weight_norms": weight_norms,
            "residuals_history": residuals_history if 'residuals_history' in locals() else None,
            "closed_loop_pred": closed_loop_pred,
            "closed_loop_history": closed_loop_hist,
            "closed_loop_truth": closed_loop_truth,
            "chaos_results": {**(chaos_results or {}), **(stats_dict or {})}, # Merge stats
        })

class ReadoutStrategyFactory:
    """Factory to create appropriate ReadoutStrategy based on config."""
    
    @staticmethod
    def create_strategy(
        readout: ReadoutModule | None,
        dataset_meta: DatasetMetadata,
        evaluator: Evaluator,
        metric_name: str
    ) -> ReadoutStrategy:
        if readout is None:
            return EndToEndStrategy(evaluator, metric_name)
        elif dataset_meta.classification:
            return ClassificationStrategy(evaluator, metric_name)
        else:
            return ClosedLoopRegressionStrategy(evaluator, metric_name)
