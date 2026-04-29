
"""
src/reservoir/pipelines/components/data_coordinator.py

Implements DataCoordinator to handle data fetching, padding, and alignment,
freeing PipelineExecutor from low-level data manipulation.
"""
from beartype import beartype
from reservoir.core.types import to_jax_f64, BatchIterator, DataLoaderProtocol
from reservoir.core.types import NpF64


from reservoir.utils.data_prep import apply_halo_padding
from reservoir.utils.reporting import print_feature_stats
from reservoir.layers.projection import Projection  # noqa: TC001
from reservoir.pipelines.config import DatasetMetadata, FrontendContext  # noqa: TC001

@beartype
class DataLoader:
    def __init__(self, X: NpF64 | None, y: NpF64 | None, batch_size: int, projection: Projection | None = None):
        self.X = X
        self.y = y
        self.projection = projection
        self.num_samples = X.shape[0] if X is not None else 0
        
        # If 2D (Time, Feat), we shouldn't batch across Time for recurrent models usually, 
        # but to keep it simple and match batched_compute's old behavior:
        # we treat the whole sequence as one batch.
        if X is not None and X.ndim == 2:
            self.batch_size = self.num_samples
        else:
            self.batch_size = batch_size

    def __iter__(self) -> BatchIterator:
        x = self.X
        y = self.y
        if self.num_samples == 0 or x is None:
            return
        for i in range(0, self.num_samples, self.batch_size):
            end = min(i + self.batch_size, self.num_samples)
            bx = to_jax_f64(x[i:end])
            if self.projection is not None:
                bx = self.projection(bx)
            by = to_jax_f64(y[i:end]) if y is not None else None
            yield bx, by


@beartype
class DataCoordinator:
    """
    Coordinates data access and preparation for the pipeline.
    Handles Halo Padding and Target Alignment.
    """

    def __init__(self, frontend_ctx: FrontendContext, dataset_meta: DatasetMetadata):
        self.ctx = frontend_ctx
        self.meta = dataset_meta
        self.processed = frontend_ctx.processed_split

    def get_train_dataloader(self, batch_size: int, projection: Projection | None = None) -> DataLoaderProtocol:
        return DataLoader(
            self.processed.train_X,
            self.processed.train_y,
            batch_size,
            projection
        )

    def get_eval_dataloader(self, split: str, batch_size: int, window_size: int = 0, projection: Projection | None = None) -> DataLoaderProtocol | None:
        if split == "val":
            inputs = self.get_val_inputs(window_size)
            # targets are aligned later, but we can pass raw y if needed.
            # actually, during extraction we just need X.
            targets = None
        elif split == "test":
            inputs = self.get_test_inputs(window_size)
            targets = None
        elif split == "train":
            inputs = self.get_train_inputs()
            targets = None
        elif split == "warmup":
            # For warmup, we use the first half of train data
            inputs = self.get_train_inputs()
            if inputs is not None:
                inputs = inputs[: len(inputs) // 2]
            targets = None
        else:
            raise ValueError(f"Unknown split: {split}")
            
        if inputs is None:
            return None
            
        return DataLoader(inputs, targets, batch_size, projection)

    def get_train_inputs(self) -> NpF64:
        """Get training inputs (no padding usually)."""
        return self.processed.train_X

    def get_val_inputs(self, window_size: int = 0) -> NpF64 | None:
        """Get validation inputs with Halo Padding applied."""
        if self.processed.val_X is None:
            return None
        
        # Apply Halo Padding using Train as context
        return apply_halo_padding(
            self.processed.val_X, 
            self.processed.train_X, 
            window_size
        )

    def get_test_inputs(self, window_size: int = 0) -> NpF64 | None:
        """Get test inputs with Halo Padding applied."""
        if self.processed.test_X is None:
            return None
            
        # Context is Val if exists, else Train
        context_source = self.processed.val_X if self.processed.val_X is not None else self.processed.train_X
        
        return apply_halo_padding(
            self.processed.test_X,
            context_source,
            window_size
        )

    def align_targets(
        self, 
        features: NpF64 | None, 
        split: str
    ) -> NpF64 | None:
        """
        Align targets for a specific split to match the given features.
        Trims targets from the start if they are longer than features (causal alignment).
        """
        if features is None:
            return None
            
        # Retrieve original targets based on split name
        if split == "train":
            targets = self.processed.train_y
        elif split == "val":
            targets = self.processed.val_y
        elif split == "test":
            targets = self.processed.test_y
        else:
            targets = None
        if targets is None:
            return None

        len_f = features.shape[0]
        len_t = targets.shape[0]
        
        if len_t > len_f:
            diff = len_t - len_f
            aligned = targets[diff:]
            print_feature_stats(aligned, "data_coordinator.py","6.5:Aligned:y:{split} (Trimmed {diff})")
            return aligned
        elif len_t < len_f:
            # This is weird but we warn
            print(f"    [DataCoordinator] WARNING: Features ({len_f}) > Targets ({len_t}) for {split}. No alignment.")
            print_feature_stats(targets, "data_coordinator.py",f"6.5:Aligned:y:{split} (Mismatch)")
            return targets
        else:
            print_feature_stats(targets, "data_coordinator.py",f"6.5:Aligned:y:{split}")
            return targets
