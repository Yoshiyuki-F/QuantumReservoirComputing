# /home/yoshi/PycharmProjects/Reservoir/src/reservoir/pipelines/config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from reservoir.core.types import TopologyMeta
    from reservoir.layers.projection import Projection
    from reservoir.layers.preprocessing import Preprocessor
    from reservoir.training import TrainingConfig
    from reservoir.data.config import DatasetPreset
    from reservoir.data import SplitDataset
    from reservoir.readout.base import ReadoutModule
    from reservoir.data.identifiers import Dataset
    from reservoir.models.generative import ClosedLoopModel


@dataclass(frozen=True)
class FrontendContext:
    processed_split: SplitDataset
    preprocessor: Preprocessor | None  # Single Preprocessor instance
    preprocessed_shape: tuple[int, ...]
    projected_shape: tuple[int, ...] | None
    input_shape_for_meta: tuple[int, ...]
    input_dim_for_factory: int
    projection_layer: Projection | None = None


@dataclass(frozen=True)
class DatasetMetadata:
    dataset: Dataset
    dataset_name: str
    preset: DatasetPreset
    training: TrainingConfig
    classification: bool
    input_shape: tuple[int, ...]


@dataclass(frozen=True)
class ModelStack:
    model: ClosedLoopModel
    readout: ReadoutModule | None
    topo_meta: TopologyMeta
    metric: str
    model_label: str
