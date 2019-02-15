from typing import Callable

import kerassurgeon
import numpy as np
from keras import models, layers
from sklearn import cluster, metrics

from filter_pruning import base_filter_pruning


class KMeansFilterPruning(base_filter_pruning.BasePruning):
    def __init__(self,
                 clustering_factor: float,
                 model_compile_fn: Callable[[models.Model], None],
                 model_finetune_fn: Callable[[models.Model, int, int], None],
                 nb_finetune_epochs: int,
                 maximum_prune_iterations: int = None,
                 maximum_pruning_percent: float = 0.9,
                 nb_trained_for_epochs: int = 0):
        super().__init__(pruning_factor=clustering_factor,
                         model_compile_fn=model_compile_fn,
                         model_finetune_fn=model_finetune_fn,
                         nb_finetune_epochs=nb_finetune_epochs,
                         nb_trained_for_epochs=nb_trained_for_epochs,
                         maximum_prune_iterations=maximum_prune_iterations,
                         maximum_pruning_percent=maximum_pruning_percent)

    def run_pruning_for_conv2d_layer(self, pruning_factor: float, layer: layers.Conv2D, surgeon: kerassurgeon.Surgeon,
                                     layer_weight_mtx) -> int:
        _, _, _, nb_channels = layer_weight_mtx.shape

        # Initialize KMeans
        nb_of_clusters, _ = self._calculate_number_of_channels_to_keep(pruning_factor, nb_channels)
        kmeans = cluster.KMeans(nb_of_clusters, "k-means++")

        # Fit with the flattened weight matrix
        # (height, width, input_channels, output_channels) -> (output_channels, flattened features)
        layer_weight_mtx_reshaped = layer_weight_mtx.transpose(3, 0, 1, 2).reshape(nb_channels, -1)
        # Apply some fuzz to the weights, to avoid duplicates
        self._apply_fuzz(layer_weight_mtx_reshaped)
        kmeans.fit(layer_weight_mtx_reshaped)

        # If a cluster has only a single member, then that should not be pruned
        # so that point will always be the closest to the cluster center
        closest_point_to_cluster_center_indices = metrics.pairwise_distances_argmin(kmeans.cluster_centers_,
                                                                                    layer_weight_mtx_reshaped)
        # Compute filter indices which can be pruned
        channel_indices = set(np.arange(len(layer_weight_mtx_reshaped)))
        channel_indices_to_keep = set(closest_point_to_cluster_center_indices)
        channel_indices_to_prune = list(channel_indices.difference(channel_indices_to_keep))
        channel_indices_to_keep = list(channel_indices_to_keep)

        # TODO: These things can happen because of the KMeans clustering, this needs more investigation
        if len(channel_indices_to_keep) > nb_of_clusters:
            print("Number of selected channels for pruning is less than expected")
            diff = len(channel_indices_to_keep) - nb_of_clusters
            print("Randomly adding {0} channels for pruning".format(diff))
            np.random.shuffle(channel_indices_to_keep)
            for i in range(diff):
                channel_indices_to_prune.append(channel_indices_to_keep.pop(i))
        elif len(channel_indices_to_keep) < nb_of_clusters:
            print("Number of selected channels for pruning is greater than expected. Leaving too few channels.")
            diff = nb_of_clusters - len(channel_indices_to_keep)
            print("Discarding {0} pruneable channels".format(diff))
            for i in range(diff):
                channel_indices_to_keep.append(channel_indices_to_prune.pop(i))

        if len(channel_indices_to_keep) != nb_of_clusters:
            raise ValueError(
                "Number of clusters {0} is not equal with the selected "
                "pruneable channels {1}".format(nb_of_clusters, len(channel_indices_to_prune)))

        # Remove "unnecessary" filters from layer
        surgeon.add_job("delete_channels", layer, channels=channel_indices_to_prune)

        return len(channel_indices_to_prune)
