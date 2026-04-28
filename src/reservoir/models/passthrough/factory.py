"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/models/passthrough/factory.py
Factory for Passthrough model (projection + aggregation only).
"""
from __future__ import annotations


from reservoir.layers.aggregation import AggregationMode
from reservoir.models.config import PassthroughConfig
from reservoir.models.passthrough.passthrough import PassthroughModel
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reservoir.models.presets import PipelineConfig


class PassthroughFactory:
    """Factory for creating Passthrough Model."""

    @staticmethod
    def create_model(
        pipeline_config: PipelineConfig,
        projected_input_dim: int,
        output_dim: int,
        input_shape: tuple[int, ...] | None,
    ) -> PassthroughModel:
        """Create passthrough model - only needs aggregation mode from config."""
        model_config = pipeline_config.model
        if not isinstance(model_config, PassthroughConfig):
            raise TypeError(f"PassthroughFactory requires PassthroughConfig, got {type(model_config)}.")

        # Create model with just aggregation mode
        model = PassthroughModel(aggregation_mode=model_config.aggregation)

        # Build topology metadata for pipeline compatibility
        if input_shape and len(input_shape) >= 2:
            batch_dim = int(input_shape[0]) if len(input_shape) == 3 else None
            t_steps = int(input_shape[1] if len(input_shape) == 3 else input_shape[0])
            feature_units = model.get_feature_dim(projected_input_dim, t_steps)
            
            def _with_batch(s: tuple[int, ...]) -> tuple[int, ...]:
                return (batch_dim,) + s if batch_dim else s
            
            if model_config.aggregation == AggregationMode.SEQUENCE:
                feat_shape = _with_batch((t_steps, feature_units))
            else:
                feat_shape = _with_batch((feature_units,))
        else:
            feat_shape = None
            
        model.topology_meta = {
            "type": "PASSTHROUGH",
            "shapes": {
                "adapter": None,  # No adapter for passthrough
                "internal": None,  # No internal layers
                "feature": feat_shape,
                "output": (output_dim,),
            },
            "details": {"agg_mode": model_config.aggregation.value, "structure": "Projection -> Aggregation"},
        }
        return model
