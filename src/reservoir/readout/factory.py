"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/readout/factory.py
Factory for creating readout instances from configuration."""
from __future__ import annotations


from reservoir.models.config import RidgeReadoutConfig, PolyRidgeReadoutConfig, FNNReadoutConfig, ReadoutConfig
from reservoir.readout.ridge import RidgeCV
from reservoir.readout.poly_ridge import PolyRidgeReadout
from reservoir.readout.fnn_readout import FNNReadout
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reservoir.training.config import TrainingConfig
    from reservoir.readout.base import ReadoutModule


def _lambda_candidates_or_default(candidates: tuple[float, ...] | None) -> tuple[float, ...]:
    if candidates is None:
        return (1e-6,)
    return candidates


class ReadoutFactory:
    """Builds readout modules from ReadoutConfig."""

    @staticmethod
    def create_readout(
        config: ReadoutConfig | None,
        classification: bool,
        training_config: TrainingConfig | None = None,
    ) -> ReadoutModule | None:
        # None (End-to-End) の場合
        if config is None:
            return None

        # PolyRidgeの場合 (must check before RidgeReadoutConfig)
        if isinstance(config, PolyRidgeReadoutConfig):
            candidates = _lambda_candidates_or_default(config.lambda_candidates)
            return PolyRidgeReadout(
                use_intercept=config.use_intercept,
                lambda_candidates=candidates,
                degree=config.degree,
                mode=config.mode,
                norm_threshold=getattr(config, "norm_threshold", 100.0)
            )

        # Ridgeの場合
        if isinstance(config, RidgeReadoutConfig):
            candidates = _lambda_candidates_or_default(config.lambda_candidates)
            
            return RidgeCV(
                use_intercept=config.use_intercept,
                lambda_candidates=candidates,
                norm_threshold=getattr(config, "norm_threshold", 100.0)
            )

        # FNNの場合
        elif isinstance(config, FNNReadoutConfig):
            return FNNReadout(
                hidden_layers=config.hidden_layers,
                training_config=training_config,
                classification=classification
            )

        raise TypeError(f"ReadoutFactory received unknown config type: {config.__class__.__name__}")

__all__ = ["ReadoutFactory"]
