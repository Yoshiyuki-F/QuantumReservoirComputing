"""
src/reservoir/readout/poly_ridge.py
Polynomial feature expansion readout – inherits RidgeCV without modifying it.

Both modes are implemented with **pure JAX operations** so they work
inside jax.lax.scan (closed-loop generation).

Two modes:
  - "square_only": appends x_i^2 (and optionally x_i^3, …) to the original vector.
    Keeps dimensionality manageable (N → N * degree).
  - "full": all cross-terms x_i * x_j (i <= j) via jnp upper-triangle indexing.
    Produces N + N*(N+1)/2 features for degree=2.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal
import jax.numpy as jnp

from reservoir.readout.ridge import RidgeCV, RidgeRegression

if TYPE_CHECKING:
    from collections.abc import Callable

    from reservoir.core.types import JaxF64, ConfigDict


type PolyRidgeValidationResult = tuple[
    float,
    float,
    dict[float, float],
    dict[float, float],
    dict[float, JaxF64],
]


class PolyRidgeReadout(RidgeCV):
    """Ridge readout with polynomial feature expansion.

    Overrides fit / predict / fit_with_validation to expand features
    *before* delegating to the parent RidgeCV logic.
    All expansion is pure JAX – safe inside jax.lax.scan.
    """

    def __init__(
        self,
        lambda_candidates: tuple[float, ...],
        degree: int,
        mode: Literal["full", "square_only", "interaction_only"],
        use_intercept: bool = True,
        norm_threshold: float | None = 100.0
    ) -> None:
        super().__init__(lambda_candidates=lambda_candidates, use_intercept=use_intercept, norm_threshold=norm_threshold)
        self.degree = degree
        self.mode = mode

    # ------------------------------------------------------------------
    # Feature expansion (pure JAX)
    # ------------------------------------------------------------------
    def _expand_features(self, X: JaxF64) -> JaxF64:
        """Expand input features according to the configured mode."""
        if self.mode == "square_only":
            return self._expand_square_only(X)
        elif self.mode == "full":
            return self._expand_full(X)
        elif self.mode == "interaction_only":
            return self._expand_interaction_only(X)
        else:
            raise ValueError(f"Unknown PolyRidgeReadout mode: {self.mode!r}")

    def _expand_square_only(self, X: JaxF64) -> JaxF64:
        """Append x_i^k for k=2..degree to the original feature vector.

        For degree=2:  [x1, ..., xN, x1^2, ..., xN^2]
        """
        parts = [X]
        for k in range(2, self.degree + 1):
            parts.append(X ** k)
        return jnp.concatenate(parts, axis=-1)

    @staticmethod
    def _expand_full(X: JaxF64) -> JaxF64:
        """Pure-JAX full polynomial expansion (degree=2).

        Produces: [original features] + [x_i * x_j for i <= j]
        For n features → n + n*(n+1)/2 output features.
        """
        n_features = X.shape[-1]

        # Upper-triangle indices (including diagonal) → x_i * x_j for i <= j
        idx_i, idx_j = jnp.triu_indices(n_features)
        cross_terms = X[..., idx_i] * X[..., idx_j]  # works for any batch dims

        return jnp.concatenate([X, cross_terms], axis=-1)

    @staticmethod
    def _expand_interaction_only(X: JaxF64) -> JaxF64:
        """Pure-JAX interaction-only polynomial expansion (degree=2).

        Produces: [original features] + [x_i * x_j for i < j] (no self-squared terms).
        Perfectly matches QRC Z + ZZ feature dimensionality.
        For n features → n + n*(n-1)/2 output features.
        Sigma(1<=k<=degree, n choose k) for degree=2 → n + n*(n-1)/2. See sklearn PolynomialFeatures interaction_only=True.
        """
        n_features = X.shape[-1]

        # k=1 specifies strict upper-triangle indices (i < j), excluding the diagonal
        idx_i, idx_j = jnp.triu_indices(n_features, k=1)
        cross_terms = X[..., idx_i] * X[..., idx_j]  # works for any batch dims

        return jnp.concatenate([X, cross_terms], axis=-1)

    def map_features(self, X: JaxF64) -> JaxF64:
        """Public interface for feature expansion (used by strategies)."""
        return self._expand_features(X)

    # ------------------------------------------------------------------
    # Overridden ReadoutModule interface
    # ------------------------------------------------------------------
    def fit(self, states: JaxF64, targets: JaxF64) -> PolyRidgeReadout:
        """Expand features, then delegate to RidgeCV.fit."""
        X_expanded = self._expand_features(states)
        super().fit(X_expanded, targets)
        return self

    def predict(self, states: JaxF64) -> JaxF64:
        """Expand features, then delegate to RidgeCV.predict."""
        X_expanded = self._expand_features(states)
        return super().predict(X_expanded)

    def fit_with_validation(
        self,
        train_Z: JaxF64,
        train_y: JaxF64,
        val_Z: JaxF64,
        val_y: JaxF64,
        scoring_fn: Callable[[JaxF64, JaxF64], float],
        maximize_score: bool,
    ) -> PolyRidgeValidationResult:
        """Search lambda candidates on expanded features and keep the best model."""
        train_expanded = self._expand_features(train_Z)
        val_expanded = self._expand_features(val_Z)
        search_history: dict[float, float] = {}
        weight_norms: dict[float, float] = {}
        residuals_history: dict[float, JaxF64] = {}

        best_lambda = self.lambda_candidates[0]
        best_score = float("-inf") if maximize_score else float("inf")
        best_model: RidgeRegression | None = None

        for lam in self.lambda_candidates:
            lam_value = float(lam)
            candidate = RidgeRegression(lam_value, self.use_intercept).fit(train_expanded, train_y)
            pred = candidate.predict(val_expanded)
            score = float(scoring_fn(pred, val_y))
            search_history[lam_value] = score
            residuals_history[lam_value] = (pred - val_y) ** 2

            coef = candidate.coef_
            weight_norms[lam_value] = float(jnp.linalg.norm(coef)) if coef is not None else 0.0
            is_better = score > best_score if maximize_score else score < best_score
            if is_better:
                best_lambda = lam_value
                best_score = score
                best_model = candidate

        if best_model is None:
            best_model = RidgeRegression(best_lambda, self.use_intercept).fit(train_expanded, train_y)
        self.best_model = best_model
        return best_lambda, best_score, search_history, weight_norms, residuals_history

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> ConfigDict:
        data = super().to_dict()
        res: ConfigDict = dict(data)
        res["degree"] = int(self.degree)
        res["mode"] = str(self.mode)
        return res
