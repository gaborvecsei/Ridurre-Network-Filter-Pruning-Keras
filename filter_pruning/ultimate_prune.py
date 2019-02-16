import warnings
from typing import Callable

import kerassurgeon
import numpy as np
from keras import models, layers

from filter_pruning import base_filter_pruning


class UltimatePruning(base_filter_pruning.BasePruning):
    """
    You do not dare to do it ;)

    - Best. Pruning. Ever.
    """

    def __init__(self, model_compile_fn: Callable[[models.Model], None]):
        super().__init__(1.0, model_compile_fn, None, 0, 0, 1, 1.0)
        warnings.warn("Really? You are going to use the ultimate pruning method? Who do you think you are?")

    def run_pruning_for_conv2d_layer(self,
                                     pruning_factor: float,
                                     layer: layers.Conv2D,
                                     surgeon: kerassurgeon.Surgeon,
                                     layer_weight_mtx) -> int:
        nb_channels = layer_weight_mtx.shape[-1]
        nb_filters_to_remove = nb_channels - 1

        filter_indices_to_remove = np.arange(nb_channels)
        np.random.shuffle(filter_indices_to_remove)
        filter_indices_to_remove = list(filter_indices_to_remove[:nb_filters_to_remove])

        surgeon.add_job("delete_channels", layer, channels=filter_indices_to_remove)

        return len(filter_indices_to_remove)
