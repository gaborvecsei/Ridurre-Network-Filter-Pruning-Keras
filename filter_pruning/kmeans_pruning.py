import numpy as np
from keras import layers
from sklearn import cluster

from filter_pruning import base_pruning


class KMeansFilterPruning(base_pruning.BaseFilterPruning):
    def __init__(self, start_at_epoch: int, finetune_for_epochs: int, clustering_factor: float,
                 prunable_layers_regex: str = ".*"):
        super().__init__(start_at_epoch, finetune_for_epochs, prunable_layers_regex)

        self.clustering_factor = clustering_factor

    def _calculate_nb_of_clusters(self, nb_of_channels) -> int:
        nb_of_clusters = int(np.ceil(nb_of_channels * self.clustering_factor))

        if nb_of_clusters >= nb_of_channels:
            nb_of_clusters = nb_of_channels - 1
        elif nb_of_clusters < 2:
            nb_of_clusters = 2

        return nb_of_clusters

    def run_pruning_for_conv_layer(self, layer: layers.Conv2D) -> int:
        layer_weight_mtx = layer.get_weights()[0]
        height, width, input_channels, channels = layer_weight_mtx.shape

        # Initialize KMeans
        nb_of_clusters = self._calculate_nb_of_clusters(channels)
        kmeans = cluster.KMeans(nb_of_clusters, "k-means++")

        # Fit with the flattened weight matrix
        layer_weight_mtx_reshaped = layer_weight_mtx.transpose(3, 0, 1, 2).reshape(channels, -1)
        kmeans.fit(layer_weight_mtx_reshaped)

        # TODO: calculate which filters to prune

        # TODO: return number of pruned filters
        return -1
