import re
from typing import Tuple, Callable

import tensorflow as tf
from keras import models
import tempfile
import abc
from keras import backend as K

import kerassurgeon


class BasePruning:
    def __init__(self, prunable_layers_regex: str,
                 model_compile_fn: Callable[[models.Model], None],
                 model_finetune_fn: Callable[[models.Model, int, int], None], nb_finetune_epochs: int,
                 nb_trained_for_epochs: int):
        self._prunable_layers_regex = prunable_layers_regex

        self._tmp_model_file_name = tempfile.NamedTemporaryFile().name

        self._model_compile_fn = model_compile_fn
        self._model_finetune_fn = model_finetune_fn

        self._nb_finetune_epochs = nb_finetune_epochs
        self._current_nb_of_epochs = nb_trained_for_epochs

    def run_pruning(self, model: models.Model):
        while True:
            # TODO: define stopping criteria
            print("Running filter pruning...")
            model, pruned_filters_dict = self._prune(model)
            self._save_after_pruning(model)
            self._clean_up_after_pruning(model)
            model = self._load_back_saved_model()
            self._model_compile_fn(model)
            self._model_finetune_fn(model, self._current_nb_of_epochs,
                                    self._current_nb_of_epochs + self._nb_finetune_epochs)
            self._current_nb_of_epochs += self._nb_finetune_epochs

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
