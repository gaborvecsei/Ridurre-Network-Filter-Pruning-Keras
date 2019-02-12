import abc
import re
import tempfile
from typing import Tuple, Callable
import traceback
import kerassurgeon
import tensorflow as tf
from keras import backend as K
from keras import models
import numpy as np


class BasePruning:
    _FUZZ_EPSILON = 1e-5

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

    def run_pruning(self, model: models.Model, custom_objects_inside_model: dict = None) -> Tuple[models.Model, int]:
        self._original_number_of_filters = self._count_number_of_filters(model)

        pruning_iteration = 0

        while True:
            # Pruning step
            print("Running filter pruning {0}".format(pruning_iteration))
            model, pruning_dict = self._prune(model)

            # Computing statistics
            nb_of_pruned_filters = sum(pruning_dict.values())
            if nb_of_pruned_filters == 0:
                print("Number of pruned filters == 0, so pruning is stopped")
                break
            print("Number of pruned filters at this step: {0}".format(nb_of_pruned_filters))
            pruning_percent = self._compute_pruning_percent(model)
            print("Network is pruned from the original state, by {0} %".format(pruning_percent * 100))

            # Finetune step
            self._save_after_pruning(model)
            self._clean_up_after_pruning(model)
            model = self._load_back_saved_model(custom_objects_inside_model)
            self._model_compile_fn(model)
            if self._model_finetune_fn is not None:
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
        try:
            new_model = surgeon.operate()
        except Exception as e:
            print("Could not complete pruning step because got Exception: {0}".format(e))
            print(traceback.format_exc())
            return model, {k: 0 for k, _ in pruning_dict.items()}
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
        model.save(self._tmp_model_file_name, overwrite=True, include_optimizer=True)

    @staticmethod
    def _clean_up_after_pruning(model: models.Model):
        del model
        K.clear_session()
        tf.reset_default_graph()

    def _load_back_saved_model(self, custom_objects: dict) -> models.Model:
        model = models.load_model(self._tmp_model_file_name, custom_objects=custom_objects)
        return model

    @staticmethod
    def _apply_fuzz_to_vector(x: np.ndarray):
        # Prepare the vector element indices
        indices = np.arange(0, len(x), dtype=int)
        np.random.shuffle(indices)
        # Select the indices to be modified (always modify only N-1 values)
        nb_of_values_to_modify = np.random.randint(0, len(x) - 1)
        modify_indices = indices[:nb_of_values_to_modify]
        # Modify the selected elements of the vector
        x[modify_indices] += BasePruning._epsilon()

    @staticmethod
    def _apply_fuzz(x: np.ndarray):
        for i in range(len(x)):
            BasePruning._apply_fuzz_to_vector(x[i])

    @staticmethod
    def _epsilon():
        return BasePruning._FUZZ_EPSILON

    @staticmethod
    def set_epsilon(e: float):
        BasePruning._FUZZ_EPSILON = e

    @staticmethod
    def _calculate_number_of_channels_to_keep(keep_factor: float, nb_of_channels: int) -> Tuple[int, int]:
        # This is the number of channels we would like to keep
        new_nb_of_channels = int(np.ceil(nb_of_channels * keep_factor))

        if new_nb_of_channels > nb_of_channels:
            # This happens when (factor > 1)
            new_nb_of_channels = nb_of_channels
        elif new_nb_of_channels < 1:
            # This happens when (factor <= 0)
            new_nb_of_channels = 1

        # Number of channels which will be removed
        nb_channels_to_remove = nb_of_channels - new_nb_of_channels

        return new_nb_of_channels, nb_channels_to_remove

    @abc.abstractmethod
    def run_pruning_for_conv2d_layer(self, layer, surgeon: kerassurgeon.Surgeon) -> int:
        raise NotImplementedError
