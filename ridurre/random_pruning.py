from typing import Callable, Optional, List

import numpy as np
from keras import models, layers

from ridurre import base_filter_pruning


class RandomFilterPruning(base_filter_pruning.BasePruning):
    def __init__(self,
                 removal_factor: float,
                 model_compile_fn: Callable[[models.Model], None],
                 model_finetune_fn: Optional[Callable[[models.Model, int, int], None]],
                 nb_finetune_epochs: int,
                 maximum_prune_iterations: int = None,
                 maximum_pruning_percent: float = 0.9,
                 nb_trained_for_epochs: int = 0):
        super().__init__(pruning_factor=removal_factor,
                         model_compile_fn=model_compile_fn,
                         model_finetune_fn=model_finetune_fn,
                         nb_finetune_epochs=nb_finetune_epochs,
                         nb_trained_for_epochs=nb_trained_for_epochs,
                         maximum_prune_iterations=maximum_prune_iterations,
                         maximum_pruning_percent=maximum_pruning_percent)

    def run_pruning_for_conv2d_layer(self, pruning_factor: float, layer: layers.Conv2D, layer_weight_mtx) -> List[int]:
        _, _, _, nb_channels = layer_weight_mtx.shape

        # If there is only a single filter left, then do not prune it
        if nb_channels == 1:
            print("Layer {0} has only a single filter left. No pruning is applied.".format(layer.name))
            return []

        # Calculate how much filters should be removed
        _, nb_of_filters_to_remove = self._calculate_number_of_channels_to_keep(1.0 - pruning_factor, nb_channels)

        # Select prunable filters randomly
        filter_indices = np.arange(nb_channels)
        np.random.shuffle(filter_indices)
        filter_indices = list(filter_indices[:nb_of_filters_to_remove])

        return filter_indices
