from typing import Callable

import kerassurgeon
import numpy as np
from keras import models, layers

from filter_pruning import base_filter_pruning


class KMeansFilterPruning(base_filter_pruning.BasePruning):
    def __init__(self, removal_factor: float,
                 model_compile_fn: Callable[[models.Model], None],
                 model_finetune_fn: Callable[[models.Model, int, int], None],
                 nb_finetune_epochs: int,
                 maximum_prune_iterations: int=None,
                 maximum_pruning_percent: float=0.9,
                 nb_trained_for_epochs: int = 0):
        super().__init__(model_compile_fn=model_compile_fn,
                         model_finetune_fn=model_finetune_fn,
                         nb_finetune_epochs=nb_finetune_epochs,
                         nb_trained_for_epochs=nb_trained_for_epochs,
                         maximum_prune_iterations=maximum_prune_iterations,
                         maximum_pruning_percent=maximum_pruning_percent)

        self._removal_factor = removal_factor

    def run_pruning_for_conv2d_layer(self, layer: layers.Layer, surgeon: kerassurgeon.Surgeon) -> int:
        # Extract the Conv2D layer kernel weight matrix
        layer_weight_mtx = layer.get_weights()[0]
        height, width, input_channels, nb_channels = layer_weight_mtx.shape

        # If there is only a single filter left, then do not prune it
        if nb_channels == 1:
            print("Layer {0} has only a single filter left. No pruning is applied.".format(layer.name))
            return 0

        # Calculate how much filters should be removed
        nb_of_filters_to_remove = int(np.ceil(nb_channels * self._removal_factor))

        if nb_of_filters_to_remove >= nb_channels:
            nb_of_filters_to_remove = nb_channels - 1

        # Select prunable filters randomly
        filter_indices = np.arange(len(nb_channels))
        np.random.shuffle(filter_indices)
        filter_indices = list(filter_indices[:nb_of_filters_to_remove])

        # Remove selected filters from layer
        surgeon.add_job("delete_channels", layer, channels=filter_indices)

        return nb_of_filters_to_remove
