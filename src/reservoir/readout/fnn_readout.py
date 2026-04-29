"""FNN-based readout module implementing ReadoutModule protocol."""
from __future__ import annotations


from beartype import beartype

from reservoir.readout.base import ReadoutModule
from reservoir.models.config import FNNConfig
from reservoir.models.nn.fnn import FNNModel
from reservoir.training.config import TrainingConfig  # noqa: TC001
from reservoir.core.types import JaxF64, ConfigDict, ConfigValue, TrainLogs  # noqa: TC001


def _parse_hidden_layers(value: ConfigValue | None) -> tuple[int, ...]:
    def parse_item(item: ConfigValue) -> int:
        match item:
            case int() | float() | str():
                return int(item)
            case _:
                raise ValueError(f"hidden_layers entries must be numeric, got {item.__class__.__name__}.")

    match value:
        case None:
            return ()
        case tuple() | list():
            return tuple(parse_item(v) for v in value)
        case _:
            raise ValueError(f"hidden_layers must be a sequence of integers, got {value.__class__.__name__}.")


@beartype
class FNNReadout(ReadoutModule):
    """FNN-based readout using FNNModel as backend."""

    def __init__(
        self,
        hidden_layers: tuple[int, ...] | None = None,
        training_config: TrainingConfig | None = None,
        classification: bool = False
    ) -> None:
        self.hidden_layers = hidden_layers or ()
        self.training_config = training_config
        self.classification = classification
        self._model: FNNModel | None = None
        self._input_dim: int | None = None
        self._output_dim: int | None = None
        self.training_logs: TrainLogs | None = None

    def fit(self, states: JaxF64, targets: JaxF64) -> FNNReadout:
        """Fit the FNN readout on states and targets."""
        X = states
        y = targets

        if X.ndim != 2:
            raise ValueError(f"States must be 2D, got {X.shape}")
        if y.ndim == 1:
            y = y[:, None]

        input_dim: int = X.shape[1]
        output_dim: int = y.shape[1]
        self._input_dim = input_dim
        self._output_dim = output_dim

        # Create FNNModel with appropriate dimensions
        fnn_config = FNNConfig(hidden_layers=self.hidden_layers)
        if self.training_config is None:
            raise ValueError("FNNReadout requires training_config. Use Factory to create properly configured instance.")
        model = FNNModel(
            model_config=fnn_config,
            training_config=self.training_config,
            input_dim=input_dim,
            output_dim=output_dim,
            classification=self.classification
        )
        self._model = model

        # Train the model and store logs
        self.training_logs = model.train(X, y)
        return self

    def predict(self, states: JaxF64) -> JaxF64:
        """Predict using the trained FNN."""
        if self._model is None:
            raise RuntimeError("FNNReadout is not fitted yet.")
        X = states
        return self._model.predict(X)

    def to_dict(self) -> ConfigDict:
        """Serialize the FNN readout."""
        return {
            "hidden_layers": self.hidden_layers,
            "input_dim": self._input_dim,
            "output_dim": self._output_dim
        }

    @classmethod
    def from_dict(cls, data: ConfigDict) -> FNNReadout:
        return cls(hidden_layers=_parse_hidden_layers(data.get("hidden_layers")))
