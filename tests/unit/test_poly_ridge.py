"""Unit tests for PolyRidgeReadout."""
from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import jax.numpy as jnp
import pytest

from reservoir.readout.poly_ridge import PolyRidgeReadout
from reservoir.readout.ridge import RidgeCV
from reservoir.models.config import PolyRidgeReadoutConfig
from reservoir.readout.factory import ReadoutFactory

if TYPE_CHECKING:
    from typing import Literal

    from jax import Array


# -------------------------------------------------------------------
# 1. Feature expansion shape tests
# -------------------------------------------------------------------
class TestSquareOnlyExpansion:
    """square_only mode should concatenate x^2 .. x^degree."""

    def test_degree2_doubles_features(self):
        model = PolyRidgeReadout(lambda_candidates=(1.0,), degree=2, mode="square_only")
        X = jnp.ones((100, 10))
        out = model._expand_features(X)
        assert out.shape == (100, 20), f"Expected (100,20) got {out.shape}"

    def test_degree3_triples_features(self):
        model = PolyRidgeReadout(lambda_candidates=(1.0,), degree=3, mode="square_only")
        X = jnp.ones((50, 8))
        out = model._expand_features(X)
        # original(8) + x^2(8) + x^3(8) = 24
        assert out.shape == (50, 24), f"Expected (50,24) got {out.shape}"

    def test_values_correct(self):
        model = PolyRidgeReadout(lambda_candidates=(1.0,), degree=2, mode="square_only")
        X = jnp.array([[2.0, 3.0]])
        out = model._expand_features(X)
        expected = jnp.array([[2.0, 3.0, 4.0, 9.0]])
        np.testing.assert_allclose(np.asarray(out), np.asarray(expected))


class TestFullExpansion:
    """full mode should produce original + upper-triangle cross terms (pure JAX)."""

    def test_degree2_cross_terms(self):
        model = PolyRidgeReadout(lambda_candidates=(1.0,), degree=2, mode="full")
        X = jnp.ones((100, 5))
        out = model._expand_features(X)
        # original(5) + upper-triangle(5*(5+1)/2 = 15) = 20
        assert out.shape == (100, 20), f"Expected (100,20) got {out.shape}"

    def test_transform_consistent(self):
        """Multiple calls should give same shape (no fit/transform state)."""
        model = PolyRidgeReadout(lambda_candidates=(1.0,), degree=2, mode="full")
        X1 = jnp.ones((100, 5))
        X2 = jnp.ones((30, 5))
        out1 = model._expand_features(X1)
        out2 = model._expand_features(X2)
        assert out1.shape == (100, 20)
        assert out2.shape == (30, 20)

    def test_values_correct(self):
        model = PolyRidgeReadout(lambda_candidates=(1.0,), degree=2, mode="full")
        X = jnp.array([[2.0, 3.0]])
        out = model._expand_features(X)
        # original: [2, 3]
        # triu cross: (0,0)=4, (0,1)=6, (1,1)=9
        expected = jnp.array([[2.0, 3.0, 4.0, 6.0, 9.0]])
        np.testing.assert_allclose(np.asarray(out), np.asarray(expected))

    def test_jax_traceable(self):
        """full mode must work inside jax.jit (proxy for jax.lax.scan compat)."""
        import jax

        model = PolyRidgeReadout(lambda_candidates=(1.0,), degree=2, mode="full")

        @jax.jit
        def expand(x: Array) -> Array:
            return model._expand_features(x)

        X = jnp.ones((1, 5))
        out = expand(X)
        assert out.shape == (1, 20)


# -------------------------------------------------------------------
# 2. End-to-end fit / predict
# -------------------------------------------------------------------
class TestFitPredict:
    """PolyRidgeReadout should fit and predict successfully."""

    def test_square_only_fit_predict(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 10)).astype(np.float64)
        y = X @ rng.standard_normal(10) + 0.1 * rng.standard_normal(200)

        model = PolyRidgeReadout(lambda_candidates=(1e-3, 1e-1, 1.0), degree=2, mode="square_only")
        model.fit(jnp.array(X), jnp.array(y))
        pred = model.predict(jnp.array(X))
        assert pred.shape == (200,), f"Prediction shape mismatch: {pred.shape}"

    def test_full_fit_predict(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((200, 5)).astype(np.float64)
        y = X @ rng.standard_normal(5) + 0.1 * rng.standard_normal(200)

        model = PolyRidgeReadout(lambda_candidates=(1e-3, 1.0), degree=2, mode="full")
        model.fit(jnp.array(X), jnp.array(y))
        pred = model.predict(jnp.array(X))
        assert pred.shape == (200,)


# -------------------------------------------------------------------
# 3. fit_with_validation
# -------------------------------------------------------------------
class TestFitWithValidation:
    def test_fit_with_validation_runs(self):
        rng = np.random.default_rng(42)
        X_tr = jnp.array(rng.standard_normal((160, 10)))
        y_tr = jnp.array(rng.standard_normal(160))
        X_va = jnp.array(rng.standard_normal((40, 10)))
        y_va = jnp.array(rng.standard_normal(40))

        def mse_fn(pred: Array, truth: Array) -> float:
            return float(np.mean((pred - truth) ** 2))

        model = PolyRidgeReadout(lambda_candidates=(1e-2, 1.0), degree=2, mode="square_only")
        result = model.fit_with_validation(
            train_Z=X_tr, train_y=y_tr, val_Z=X_va, val_y=y_va,
            scoring_fn=mse_fn, maximize_score=False,
        )
        best_lambda, best_score, search_history, weight_norms, residuals = result
        assert best_lambda in (1e-2, 1.0)
        assert len(search_history) == 2


# -------------------------------------------------------------------
# 4. Factory integration
# -------------------------------------------------------------------
class TestFactory:
    def test_factory_creates_poly_ridge(self):
        config = PolyRidgeReadoutConfig(
            use_intercept=True,
            lambda_candidates=(1e-2, 1.0),
            degree=2,
            mode="square_only",
        )
        readout = ReadoutFactory.create_readout(config, classification=False)
        assert isinstance(readout, PolyRidgeReadout)
        assert readout.degree == 2
        assert readout.mode == "square_only"


# -------------------------------------------------------------------
# 5. RidgeCV is not affected (smoke test)
# -------------------------------------------------------------------
class TestRidgeCVUnaffected:
    def test_ridge_cv_still_works(self):
        rng = np.random.default_rng(42)
        X = jnp.array(rng.standard_normal((100, 10)))
        y = jnp.array(rng.standard_normal(100))
        model = RidgeCV(lambda_candidates=(1.0,), use_intercept=True)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == (100,)


# -------------------------------------------------------------------
# 6. Config validation and serialisation
# -------------------------------------------------------------------
class TestConfig:
    def test_valid_config(self):
        cfg = PolyRidgeReadoutConfig(
            use_intercept=True, lambda_candidates=(1.0,), degree=2, mode="square_only"
        )
        cfg.validate()

    def test_invalid_degree(self):
        cfg = PolyRidgeReadoutConfig(
            use_intercept=True, lambda_candidates=(1.0,), degree=1, mode="square_only"
        )
        with pytest.raises(ValueError, match="degree must be >= 2"):
            cfg.validate()

    def test_invalid_mode(self):
        invalid_mode = cast("Literal['full', 'square_only', 'interaction_only']", "invalid")
        cfg = PolyRidgeReadoutConfig(
            use_intercept=True, lambda_candidates=(1.0,), degree=2, mode=invalid_mode
        )
        with pytest.raises(ValueError, match="mode must be"):
            cfg.validate()

    def test_to_dict_includes_poly_fields(self):
        cfg = PolyRidgeReadoutConfig(
            use_intercept=True, lambda_candidates=(0.1, 1.0), degree=2, mode="full"
        )
        d = cfg.to_dict()
        assert d["degree"] == 2
        assert d["mode"] == "full"
        assert d["use_intercept"] is True
        assert d["lambda_candidates"] == [0.1, 1.0]
