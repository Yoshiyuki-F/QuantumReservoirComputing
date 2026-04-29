"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/layers/projection.py
STEP3 Projection which changes the dimension of feature vectors.
"""
from __future__ import annotations
import abc
from functools import singledispatch

from beartype import beartype
import jax
import jax.numpy as jnp
from typing import TYPE_CHECKING

from reservoir.core.types import JaxF64  # noqa: TC001

if TYPE_CHECKING:
    from reservoir.core.types import ConfigDict
    from reservoir.models.config import (
        BoundedAffinePCAConfig,
        CenterCropProjectionConfig,
        PCAProjectionConfig,
        PolynomialProjectionConfig,
        ProjectionConfig,
        RandomProjectionConfig,
        ResizeProjectionConfig,
    )


# --- 1. Interface Definition ---

@beartype
class Projection(abc.ABC):
    """
    Abstract Base Class for Input Projections.
    Defines the contract that all projection layers must follow.
    Also handles common state (input_dim, output_dim).
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        self.input_dim = int(input_dim)
        self._output_dim = int(output_dim)
    
    @property
    def output_dim(self) -> int:
        """Return the output dimension size."""
        return self._output_dim

    def __call__(self, inputs: JaxF64) -> JaxF64:
        """
        Apply the projection to the inputs.
        template method: validates input shape then calls _forward.
        """
        arr = inputs
        if arr.ndim not in (2, 3):
             raise ValueError(f"{self.__class__.__name__} expects 2D or 3D input, got shape {arr.shape}")
        return self._project(arr)

    @abc.abstractmethod
    def _project(self, inputs: JaxF64) -> JaxF64:
        """Actual projection logic to be implemented by subclasses."""

    @abc.abstractmethod
    def to_dict(self) -> ConfigDict:
        """Serialize configuration parameters."""

    @property
    def label(self) -> str:
        """Short label for filenames."""
        return f"{self.__class__.__name__}{self._output_dim}"


# --- 2. Concrete Implementations ---

@beartype
class RandomProjection(Projection):
    """
    Fixed random input projection with sparsity and bias.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        input_scale: float,
        input_connectivity: float,
        seed: int,
        bias_scale: float,
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.input_scale = float(input_scale)
        self.connectivity = float(input_connectivity)
        self.bias_scale = float(bias_scale)
        self.seed = int(seed)
        
        # Determine if bias is used based on scale
        self.use_bias = self.bias_scale > 0.0

        k_w, k_b, k_mask = jax.random.split(jax.random.PRNGKey(self.seed), 3)

        boundary = self.input_scale
        W = jax.random.uniform(
            k_w,
            (self.input_dim, self._output_dim),
            minval=-boundary,
            maxval=boundary,
        )
        if 0.0 < self.connectivity < 1.0:
            mask = jax.random.bernoulli(k_mask, p=self.connectivity, shape=W.shape)
            W = jnp.where(mask, W, 0.0)
        
        bias = jnp.zeros((self._output_dim,))
        if self.use_bias:
            bias = jax.random.uniform(
                k_b,
                (self._output_dim,),
                minval=-self.bias_scale,
                maxval=self.bias_scale,
            )
        self.W = W
        self.bias = bias

    def _project(self, inputs: JaxF64) -> JaxF64:
        # Template ensures inputs is 2D or 3D ndarray
        if inputs.ndim == 3:
            return jnp.einsum("bti,io->bto", inputs, self.W) + self.bias
        # inputs.ndim == 2
        return jnp.dot(inputs, self.W) + self.bias

    def to_dict(self) -> ConfigDict:
        return {
            "type": "random_projection",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_scale": self.input_scale,
            "connectivity": self.connectivity,
            "bias_scale": self.bias_scale,
            "seed": self.seed,
        }

    @property
    def label(self) -> str:
        return f"RP{self._output_dim}_is{self.input_scale:.2f}_c{self.connectivity:.2f}_bs{self.bias_scale:.2f}"


@beartype
class CoherentDriveProjection(Projection):
    """
    Coherent Drive Projection for Quantum Reservoir Computing.
    
    Applies arcsin transformation to convert linear input values to
    quantum rotation angles that achieve amplitude encoding.
    
    The transformation: θ = 2 * arcsin(clip(x, -1, 1))
    When applied with Ry(θ), the state becomes: ψ = sqrt(1-a²)0 + a1
    
    This is non-periodic (unlike angle embedding) and suitable for
    trending time series like Mackey-Glass.
    """

    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            input_scale: float,
            input_connectivity: float,
            seed: int,
            bias_scale: float,
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.input_scale = float(input_scale)
        self.connectivity = float(input_connectivity)
        self.bias_scale = float(bias_scale)
        self.seed = int(seed)
        
        # Use RandomProjection-style linear transformation first
        self.use_bias = self.bias_scale > 0.0

        k_w, k_b, k_mask = jax.random.split(jax.random.PRNGKey(self.seed), 3)

        boundary = self.input_scale
        W = jax.random.uniform(
            k_w,
            (self.input_dim, self._output_dim),
            minval=-boundary,
            maxval=boundary,
        )
        if 0.0 < self.connectivity < 1.0:
            mask = jax.random.bernoulli(k_mask, p=self.connectivity, shape=W.shape)
            W = jnp.where(mask, W, 0.0)
        
        bias = jnp.zeros((self._output_dim,))
        if self.use_bias:
            bias = jax.random.uniform(
                k_b,
                (self._output_dim,),
                minval=-self.bias_scale,
                maxval=self.bias_scale,
            )
        self.W = W
        self.bias = bias

    def _project(self, inputs: JaxF64) -> JaxF64:
        # Step 1: Linear projection (like RandomProjection)
        if inputs.ndim == 3:
            linear_out = jnp.einsum("bti,io->bto", inputs, self.W) + self.bias
        else:
            linear_out = jnp.dot(inputs, self.W) + self.bias
        
        # Step 2: Coherent Drive transformation (arcsin)
        # Use tanh for soft saturation to [-1, 1] - preserves information at peaks
        # (Unlike clip which causes information loss for |x| > 1)
        saturated = jnp.tanh(linear_out)
        # θ = 2 * arcsin(a) maps amplitude a to rotation angle θ
        theta = 2.0 * jnp.arcsin(saturated)
        
        return theta

    def to_dict(self) -> ConfigDict:
        return {
            "type": "coherent_drive",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "input_scale": self.input_scale,
            "connectivity": self.connectivity,
            "bias_scale": self.bias_scale,
            "seed": self.seed,
        }

    @property
    def label(self) -> str:
        return f"CDP{self._output_dim}"


@beartype
class CenterCropProjection(Projection):
    """
    Fixed cropping projection that selects the central part of the input feature vector.
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim, output_dim)

        if self._output_dim > self.input_dim:
            raise ValueError(
                f"CenterCropProjection: output_dim ({self._output_dim}) cannot be larger "
                f"than input_dim ({self.input_dim})."
            )

        self.start_idx = (self.input_dim - self._output_dim) // 2
        self.end_idx = self.start_idx + self._output_dim

    def _project(self, inputs: JaxF64) -> JaxF64:
        # inputs shape: (Batch, Time, Features)
        if inputs.ndim != 3:
             raise ValueError(f"CenterCropProjection requires 3D input (Batch, Time, Features), got ndim={inputs.ndim}")
        return inputs[..., self.start_idx : self.end_idx]

    def to_dict(self) -> ConfigDict:
        return {
            "type": "center_crop",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "crop_range": (self.start_idx, self.end_idx),
        }

    @property
    def label(self) -> str:
        return f"CCP{self._output_dim}"


@beartype
class ResizeProjection(Projection):
    """
    Projection that resizes (interpolates) the feature dimension to the target size.
    Uses JAX's image resizing capabilities (bilinear interpolation by default).
    """

    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__(input_dim, output_dim)
        if self._output_dim > self.input_dim:
            raise ValueError(
                f"ResizeProjection: output_dim ({self._output_dim}) cannot be larger "
                f"than input_dim ({self.input_dim})."
            )

    def _project(self, inputs: JaxF64) -> JaxF64:
        # inputs shape: (Batch, Time, Features) or (Batch, Features)
        # We assume the last dimension is Features.
        # jax.image.resize expects shape to resize to.
        
        input_shape = inputs.shape
        # Target shape: preserve all but last dim, replace last with output_dim
        target_shape = input_shape[:-1] + (self._output_dim,)
        
        # 'linear' (bilinear) is standard for resizing. 
        # For 1D signal (Features), linear interpolation is equivalent to resizing along 1 axis.
        return jax.image.resize(inputs, target_shape, method='linear', antialias=True)

    def to_dict(self) -> ConfigDict:
        return {
            "type": "resize_projection",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }

    @property
    def label(self) -> str:
        return f"Res{self._output_dim}"


@beartype
class PolynomialProjection(Projection):
    """
    Polynomial feature expansion projection.
    Expands input features by adding polynomial terms up to specified degree.
    Output dimension = input_dim * degree (+ 1 if include_bias).
    """

    def __init__(self, input_dim: int, degree: int, include_bias: bool) -> None:
        # Output dim = input_dim * degree (linear, squared, cubed, etc.) + optional bias
        output_dim = input_dim * degree + (1 if include_bias else 0)
        super().__init__(input_dim, output_dim)
        self.degree = degree
        self.include_bias = include_bias

    def _project(self, inputs: JaxF64) -> JaxF64:
        # inputs shape: (Batch, Time, Features) or (Batch, Features)
        features = [inputs]
        if self.degree > 1:
            for d in range(2, self.degree + 1):
                features.append(jnp.power(inputs, d))

        out = jnp.concatenate(features, axis=-1)

        if self.include_bias:
            shape = out.shape[:-1] + (1,)
            bias = jnp.ones(shape)
            out = jnp.concatenate([out, bias], axis=-1)

        return out

    def to_dict(self) -> ConfigDict:
        return {
            "type": "polynomial_projection",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "degree": self.degree,
            "include_bias": self.include_bias,
        }

    @property
    def label(self) -> str:
        return f"Poly{self._output_dim}"


@beartype
class PCAProjection(Projection):
    """
    PCA (Principal Component Analysis) projection.
    Reduces dimensionality by projecting onto top n_units principal components.
    Requires fit() to be called with training data before use.
    
    Pre-condition: Input data should be preprocessed (e.g. StandardScaler or MinMax).
    
    Args:
        input_dim: Input feature dimension.
        n_units: Number of principal components to keep.
    """

    def __init__(self, input_dim: int, n_units: int) -> None:
        if n_units > input_dim:
            raise ValueError(
                f"PCAProjection: n_units ({n_units}) cannot exceed input_dim ({input_dim})"
            )
        super().__init__(input_dim, n_units)
        self.n_units = n_units
        self._components: JaxF64 | None = None  # (n_units, input_dim)
        self._fitted = False

    def fit(self, X: JaxF64) -> PCAProjection:
        """
        Fit PCA on training data.
        X shape: (Time, Features) or (Batch, Time, Features)
        """
        # Flatten to 2D if 3D
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])  # (Batch*Time, Features)
        
        # Compute covariance directly: Sigma = X.T @ X / (N-1)
        n_samples = X.shape[0]
        cov = jnp.dot(X.T, X) / (n_samples - 1)
        eigenvalues, eigenvectors = jnp.linalg.eigh(cov)
        
        # Sort by eigenvalue (descending) and take top n_units
        idx = jnp.argsort(eigenvalues)[::-1]
        self._components = eigenvectors[:, idx[:self.n_units]].T  # (n_units, input_dim)
        self._fitted = True
        return self

    def _project(self, inputs: JaxF64) -> JaxF64:
        components = self._components
        if not self._fitted or components is None:
            raise RuntimeError("PCAProjection: fit() must be called before projection")
        
        # Direct projection (no scaling)
        return jnp.dot(inputs, components.T)

    def to_dict(self) -> ConfigDict:
        return {
            "type": "pca",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
        }

    @property
    def label(self) -> str:
        return f"PCA{self.n_units}"


@beartype
class BoundedAffinePCA(PCAProjection):
    """PCA projection followed by MinMax[-1, 1] normalization and bounded affine transform.

    Guarantees output ∈ [-bound, bound] for any (scale, relative_shift) combination.

    Pipeline:
        1. PCA projection (inherited from PCAProjection)
        2. MinMax normalization of PCA output to [-1, 1] (using training min/max)
        3. Bounded affine: y = (x_norm * bound * scale) + shift
           where shift = relative_shift * bound * (1 - scale)

    Parameters
    ----------
    input_dim : int
        Input feature dimension.
    n_units : int
        Number of principal components.
    scale : float
        Contraction factor in (0, 1].
    relative_shift : float
        Shift proportion in [-1, 1].
    bound : float
        Maximum absolute boundary limit.
    """

    def __init__(self, input_dim: int, n_units: int, scale: float, relative_shift: float, bound: float = 1.0) -> None:
        super().__init__(input_dim, n_units)
        self.scale = float(scale)
        self.relative_shift = float(relative_shift)
        self.bound = float(bound)
        self._shift = self.relative_shift * self.bound * (1.0 - self.scale)
        # Min/max from training PCA output (per component)
        self._pca_min: JaxF64 | None = None
        self._pca_max: JaxF64 | None = None

    def fit(self, X: JaxF64) -> BoundedAffinePCA:
        """Fit PCA, then compute per-component min/max of PCA output on training data."""
        super().fit(X)

        # Project training data through PCA (no affine yet)
        components = self._components
        if components is None:
            raise RuntimeError("BoundedAffinePCA components are not initialized after fit().")
        pca_out = jnp.dot(X.reshape(-1, X.shape[-1]), components.T)

        # Per-component min/max for normalization
        pca_min = jnp.min(pca_out, axis=0)  # (n_units,)
        pca_max = jnp.max(pca_out, axis=0)  # (n_units,)

        # Avoid division by zero for constant components
        range_ = pca_max - pca_min
        self._pca_min = pca_min
        self._pca_max = jnp.where(range_ < 1e-12, pca_min + 1.0, pca_max)

        return self

    def _project(self, inputs: JaxF64) -> JaxF64:
        components = self._components
        pca_min = self._pca_min
        pca_max = self._pca_max
        if not self._fitted or components is None:
            raise RuntimeError("BoundedAffinePCA: fit() must be called before projection")
        if pca_min is None or pca_max is None:
            raise RuntimeError("BoundedAffinePCA: fit() must be called before projection")

        # 1. PCA projection (no scaling)
        pca_out = jnp.dot(inputs, components.T)

        # 2. MinMax normalize to [-1, 1] per component
        range_ = pca_max - pca_min
        x_norm = 2.0 * (pca_out - pca_min) / range_ - 1.0

        # 3. Bounded affine: y = (x_norm * bound * scale) + shift
        return (x_norm * self.bound * self.scale) + self._shift

    def to_dict(self) -> ConfigDict:
        return {
            "type": "bounded_affine_pca",
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "scale": self.scale,
            "relative_shift": self.relative_shift,
            "bound": self.bound,
        }

    @property
    def label(self) -> str:
        f_min = - (self.bound * self.scale) + self._shift
        f_max = (self.bound * self.scale) + self._shift
        return f"BAPCA{self.n_units}_Min{f_min:.2f}Max{f_max:.2f}"

# --- 3. The Factory Logic (Dependency Injection Helper) ---

@singledispatch
def create_projection(config: ProjectionConfig, input_dim: int) -> Projection:
    """
    Factory function to create a Projection instance based on the config type.
    Raises TypeError if the config type is not registered.
    """
    _ = input_dim
    raise TypeError(f"Unknown projection config type: {config.__class__.__name__}")

def register_projections(
    CenterCropConfigClass: type[CenterCropProjectionConfig],
    RandomProjectionConfigClass: type[RandomProjectionConfig],
    ResizeProjectionConfigClass: type[ResizeProjectionConfig],
    PolynomialProjectionConfigClass: type[PolynomialProjectionConfig] | None = None,
    PCAProjectionConfigClass: type[PCAProjectionConfig] | None = None,
    CoherentDriveConfigClass: type[RandomProjectionConfig] | None = None,
    BoundedAffinePCAConfigClass: type[BoundedAffinePCAConfig] | None = None,
) -> None:
    """
    Call this function once to register the handlers.
    """

    @create_projection.register(CenterCropConfigClass)
    def _(config: CenterCropProjectionConfig, input_dim: int) -> CenterCropProjection:
        return CenterCropProjection(
            input_dim=int(input_dim),
            output_dim=int(config.n_units),
        )

    @create_projection.register(RandomProjectionConfigClass)
    def _(config: RandomProjectionConfig, input_dim: int) -> RandomProjection:
        return RandomProjection(
            input_dim=int(input_dim),
            output_dim=int(config.n_units),
            input_scale=float(config.input_scale),
            input_connectivity=float(config.input_connectivity),
            seed=int(config.seed),
            bias_scale=float(config.bias_scale),
        )

    @create_projection.register(ResizeProjectionConfigClass)
    def _(config: ResizeProjectionConfig, input_dim: int) -> ResizeProjection:
        return ResizeProjection(
            input_dim=int(input_dim),
            output_dim=int(config.n_units),
        )

    if PolynomialProjectionConfigClass is not None:
        @create_projection.register(PolynomialProjectionConfigClass)
        def _(config: PolynomialProjectionConfig, input_dim: int) -> PolynomialProjection:
            return PolynomialProjection(
                input_dim=int(input_dim),
                degree=int(config.degree),
                include_bias=bool(config.include_bias),
            )

    if PCAProjectionConfigClass is not None:
        @create_projection.register(PCAProjectionConfigClass)
        def _(config: PCAProjectionConfig, input_dim: int) -> PCAProjection:
            return PCAProjection(
                input_dim=int(input_dim),
                n_units=int(config.n_units),
            )


    if CoherentDriveConfigClass is not None:
        @create_projection.register(CoherentDriveConfigClass)
        def _(config: RandomProjectionConfig, input_dim: int) -> CoherentDriveProjection:
            return CoherentDriveProjection(
                input_dim=int(input_dim),
                output_dim=int(config.n_units),
                input_scale=float(config.input_scale),
                input_connectivity=float(config.input_connectivity),
                seed=int(config.seed),
                bias_scale=float(config.bias_scale),
            )

    if BoundedAffinePCAConfigClass is not None:
        @create_projection.register(BoundedAffinePCAConfigClass)
        def _(config: BoundedAffinePCAConfig, input_dim: int) -> BoundedAffinePCA:
            return BoundedAffinePCA(
                input_dim=int(input_dim),
                n_units=int(config.n_units),
                scale=float(config.scale),
                relative_shift=float(config.relative_shift),
                bound=float(config.bound),
            )


__all__ = [
    "Projection",
    "RandomProjection",
    "CenterCropProjection",
    "ResizeProjection",
    "PolynomialProjection",
    "PCAProjection",
    "BoundedAffinePCA",
    "CoherentDriveProjection",
    "create_projection",
    "register_projections"
]
