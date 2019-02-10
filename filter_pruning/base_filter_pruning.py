import abc
import re
import tempfile
from typing import Tuple, Callable

import kerassurgeon
import tensorflow as tf
from keras import backend as K
from keras import models


class BasePruning:
    def __init__(self, model_compile_fn: Callable[[models.Model], None],
                 model_finetune_fn: Callable[[models.Model, int, int], None],
                 nb_finetune_epochs: int,
                 nb_trained_for_epochs: int,
                 maximum_prune_iterations: int,
                 maximum_pruning_percent: float):

        self._tmp_model_file_name = tempfile.NamedTemporaryFile().name

        self._model_compile_fn = model_compile_fn
        self._model_finetune_fn = model_finetune_fn

        self._nb_finetune_epochs = nb_finetune_epochs
        self._current_nb_of_epochs = nb_trained_for_epochs
        self._maximum_prune_iterations = maximum_prune_iterations
        self._maximum_pruning_percent = maximum_pruning_percent

        self._original_number_of_filters = -1

        # TODO: select a subset of layers to prune
        self._prunable_layers_regex = ".*"

    def run_pruning(self, model: models.Model) -> Tuple[models.Model, int]:
        self._original_number_of_filters = self._count_number_of_filters(model)

        pruning_iteration = 0

        while True:
            # Pruning step
            print("Running filter pruning {0}".format(pruning_iteration))
            model, pruning_dict = self._prune(model)

            # Computing statistics
            nb_of_pruned_filters = sum(pruning_dict.values())
            print("Number of pruned filters at this step: {0}".format(nb_of_pruned_filters))
            pruning_percent = self._compute_pruning_percent(model)
            print("Network is pruned from the original state, by {0} %".format(pruning_percent * 100))

            # Finetune step
            self._save_after_pruning(model)
            self._clean_up_after_pruning(model)
            model = self._load_back_saved_model()
            self._model_compile_fn(model)
            self._model_finetune_fn(model, self._current_nb_of_epochs,
                                    self._current_nb_of_epochs + self._nb_finetune_epochs)
            self._current_nb_of_epochs += self._nb_finetune_epochs

            # Stopping conditions
            if nb_of_pruned_filters < 1:
                print("No filters were pruned. Pruning is stopped.")
                break
            if self._maximum_pruning_percent is not None:
                if pruning_percent > self._maximum_pruning_percent:
                    print(
                        "Network pruning (currently {0} %) reached the maximum based on your definition ({1} %)".format(
                            pruning_percent * 100, self._maximum_pruning_percent * 100))
                    break
            pruning_iteration += 1

            if self._maximum_prune_iterations is not None:
                if pruning_iteration > self._maximum_prune_iterations:
                    break

        print("Pruning stopped.")
        return model, self._current_nb_of_epochs

    def _prune(self, model: models.Model) -> Tuple[models.Model, dict]:
        surgeon = kerassurgeon.Surgeon(model, copy=True)
        pruning_dict = dict()
        for layer in model.layers:
            if layer.__class__.__name__ == "Conv2D":
                if re.match(self._prunable_layers_regex, layer.name):
                    nb_pruned_filters = self.run_pruning_for_conv2d_layer(layer, surgeon)
                    pruning_dict[layer.name] = nb_pruned_filters
        new_model = surgeon.operate()
        return new_model, pruning_dict

    @staticmethod
    def _count_number_of_filters(model: models.Model) -> int:
        nb_of_filters = 0
        for layer in model.layers:
            if layer.__class__.__name__ == "Conv2D":
                layer_weight_mtx = layer.get_weights()[0]
                _, _, _, channels = layer_weight_mtx.shape
                nb_of_filters += channels
        return nb_of_filters

    def _compute_pruning_percent(self, model: models.Model) -> float:
        nb_filters = self._count_number_of_filters(model)
        left_filters_percent = 1.0 - (nb_filters / self._original_number_of_filters)
        return left_filters_percent

    def _save_after_pruning(self, model: models.Model):
        model.save(self._tmp_model_file_name)

    @staticmethod
    def _clean_up_after_pruning(model: models.Model):
        del model
        K.clear_session()
        tf.reset_default_graph()

    def _load_back_saved_model(self) -> models.Model:
        model = models.load_model(self._tmp_model_file_name)
        return model

    @abc.abstractmethod
    def run_pruning_for_conv2d_layer(self, layer, surgeon: kerassurgeon.Surgeon) -> int:
        raise NotImplementedError
