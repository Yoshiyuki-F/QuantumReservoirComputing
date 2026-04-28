"""
src/reservoir/components/readout/ridge.py
Refactored Ridge regression designed with SOLID principles.
"""
import jax
import jax.numpy as jnp
from typing import TypedDict, cast, TYPE_CHECKING
import jax.scipy.linalg

from reservoir.readout.base import ReadoutModule

if TYPE_CHECKING:
    from reservoir.core.types import JaxF64, ConfigDict

# Robustness: Ridge Regression often requires x64 for matrix inversion stability.
def _ensure_x64() -> None:
    if not jax.config.read("jax_enable_x64"):
        jax.config.update("jax_enable_x64", True)


class _RidgeData(TypedDict, total=False):
    """Typed structure for Ridge serialization — avoids ConfigDict union explosion."""
    ridge_lambda: float
    use_intercept: bool
    coef: tuple[float, ...] | None
    intercept: tuple[float, ...] | float | None
    lambda_candidates: tuple[float, ...]
    norm_threshold: float | None


def _parse_float_tuple(value: ConfigDict) -> tuple[float, ...]:
    return tuple(float(v) for v in cast("tuple[float, ...]", value))


def _parse_ridge_data(data: ConfigDict) -> _RidgeData:
    parsed: _RidgeData = {}

    ridge_lambda = data.get("ridge_lambda")
    if ridge_lambda is not None:
        parsed["ridge_lambda"] = float(str(ridge_lambda))

    use_intercept = data.get("use_intercept")
    if use_intercept is not None:
        parsed["use_intercept"] = bool(use_intercept)

    coef = data.get("coef")
    if coef is not None:
        parsed["coef"] = _parse_float_tuple(cast("ConfigDict", coef))

    intercept = data.get("intercept")
    if intercept is not None:
        try:
            parsed["intercept"] = _parse_float_tuple(cast("ConfigDict", intercept))
        except TypeError:
            parsed["intercept"] = float(str(intercept))

    lambda_candidates = data.get("lambda_candidates")
    if lambda_candidates is not None:
        parsed["lambda_candidates"] = _parse_float_tuple(cast("ConfigDict", lambda_candidates))

    norm_threshold = data.get("norm_threshold")
    if norm_threshold is not None:
        parsed["norm_threshold"] = float(str(norm_threshold))

    return parsed

class RidgeRegression(ReadoutModule):
    """
    Single Ridge Regression model (Pure JAX).
    Responsible ONLY for fitting parameters given a fixed lambda (SRP).
    """

    def __init__(self, ridge_lambda: float, use_intercept: bool = True) -> None:
        self.ridge_lambda = float(ridge_lambda)
        self.use_intercept = bool(use_intercept)
        self.coef_: JaxF64 | None = None
        self.intercept_: JaxF64 | None = None
        self._input_dim: int | None = None

    def _add_intercept(self, X: JaxF64) -> JaxF64:
        if not self.use_intercept:
            return X
        n_samples = X.shape[0]
        ones = jnp.ones((n_samples, 1))
        return jnp.concatenate([ones, X], axis=1)

    def fit(self, states: JaxF64, targets: JaxF64) -> RidgeRegression:
        _ensure_x64()
        # Inputs are already JaxF64 per type hint
        X = states
        y = targets
        
        if y.ndim == 1:
            y = y[:, None]
        
        self._input_dim = int(X.shape[1])
        
        # Prepare Data
        X_design = self._add_intercept(X)
        n_features = X_design.shape[1]

        # Solve Normal Equations: (X^T X + lambda I) w = X^T y
        XtX = X_design.T @ X_design
        Xty = X_design.T @ y
        
        # Regularization matrix
        eye = jnp.eye(n_features)
        reg_matrix = eye * self.ridge_lambda
        
        # JAX Solver (assuming x64 enabled externally if needed)
        solve_mat = XtX + reg_matrix
        
        # Robust solver
        if self.ridge_lambda < 1e-7:
             w = jax.scipy.linalg.solve(solve_mat, Xty)
        else:
             w = jax.scipy.linalg.solve(solve_mat, Xty, assume_a="pos")

        # Handle NaNs
        w = jnp.where(jnp.isfinite(w), w, jnp.zeros_like(w))

        # Extract weights
        if self.use_intercept:
            self.intercept_ = w[0].ravel()
            self.coef_ = w[1:]
        else:
            self.intercept_ = jnp.zeros(w.shape[1])
            self.coef_ = w

        # Flatten if original target was 1D
        if targets.ndim == 1:
            self.coef_ = self.coef_.ravel()
        
        return self

    def predict(self, states: JaxF64) -> JaxF64:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted.")
        
        X = states
        pred = X @ self.coef_
        if self.intercept_ is not None:
             pred = pred + self.intercept_
        return pred

    def to_dict(self) -> ConfigDict:
        # Use .tolist() which is a safe JAX method to get primitives for JSON/serialization
        return {
            "ridge_lambda": self.ridge_lambda,
            "use_intercept": self.use_intercept,
            "coef": tuple(self.coef_.tolist()) if self.coef_ is not None else None,
            "intercept": tuple(self.intercept_.tolist()) if self.intercept_ is not None else None
        }

    @classmethod
    def from_dict(cls, data: ConfigDict) -> RidgeRegression:
        d = _parse_ridge_data(data)
        r_lam = float(d.get("ridge_lambda", 0.0))
        model = cls(
            ridge_lambda=r_lam,
            use_intercept=bool(d.get("use_intercept", True))
        )
        if d.get("coef") is not None:
            model.coef_ = jnp.array(d["coef"], dtype=jnp.float64)
            model.intercept_ = jnp.array(d.get("intercept", 0.0), dtype=jnp.float64)
        return model


class RidgeCV(ReadoutModule):
    """
    Orchestrates validation search (Pure JAX).
    Acts as the primary ReadoutModule that wraps RidgeRegression.
    """
    def __init__(
        self, 
        lambda_candidates: tuple[float, ...],
        use_intercept: bool = True,
        norm_threshold: float | None = 100.0
    ):
        if not lambda_candidates:
            raise ValueError("lambda_candidates must not be empty.")

        self.lambda_candidates = lambda_candidates
        self.use_intercept = use_intercept
        self.norm_threshold = norm_threshold
        self.best_model: RidgeRegression | None = None

        # Initialize default model with first candidate (state consistency)
        if self.lambda_candidates:
            self.best_model = RidgeRegression(self.lambda_candidates[0], use_intercept)

    @property
    def ridge_lambda(self) -> float:
        return self.best_model.ridge_lambda if self.best_model else float(self.lambda_candidates[0])

    @property
    def coef_(self) -> JaxF64 | None:
        return self.best_model.coef_ if self.best_model else None

    @property
    def intercept_(self) -> JaxF64 | None:
        return self.best_model.intercept_ if self.best_model else None

    def fit(self, states: JaxF64, targets: JaxF64) -> RidgeCV:
        """Fit using current best/default lambda."""
        if self.best_model is None:
             self.best_model = RidgeRegression(self.lambda_candidates[0], self.use_intercept)
        self.best_model.fit(states, targets)
        return self

    def predict(self, states: JaxF64) -> JaxF64:
        if self.best_model is None:
            raise RuntimeError("RidgeCV model is not fitted.")
        return self.best_model.predict(states)

    def to_dict(self) -> ConfigDict:
        data = self.best_model.to_dict() if self.best_model else {}
        res: ConfigDict = dict(data)
        res["lambda_candidates"] = tuple(self.lambda_candidates)
        res["norm_threshold"] = self.norm_threshold
        return res

    @classmethod
    def from_dict(cls, data: ConfigDict) -> RidgeCV:
        d = _parse_ridge_data(data)
        candidates_list = d.get("lambda_candidates")
        if candidates_list is None:
             lam_val = d.get("ridge_lambda", 0.0)
             candidates = (float(lam_val),)
        else:
             candidates = tuple(float(x) for x in candidates_list)
             
        instance = cls(
            lambda_candidates=candidates, 
            use_intercept=bool(d.get("use_intercept", True)),
            norm_threshold=(float(d["norm_threshold"]) if d.get("norm_threshold") is not None else None)
        )
        instance.best_model = RidgeRegression.from_dict(data)
        return instance
