import abc
import re
import tensorflow as tf
from typing import Callable
from keras import callbacks, models
from keras import backend as K
import kerassurgeon


class BaseFilterPruning(callbacks.Callback):
    def __init__(self, start_at_epoch: int, fine_tune_for_epochs: int, prunable_layers_regex: str,
                 model_compile_fn: Callable[[models.Model], None]):
        super().__init__()

        self.start_at_epoch = start_at_epoch
        self.fine_tune_for_epochs = fine_tune_for_epochs
        self.prunable_layers_regex = prunable_layers_regex

        self._current_finetuning_step = fine_tune_for_epochs
        self._model_compile_function = model_compile_fn

    def on_epoch_begin(self, epoch, logs=None):
        super().on_epoch_begin(epoch, logs)

        # TODO: pruning stopping criteria (for example: accuracy difference from the initial state dropped by 2%)

        if epoch >= self.start_at_epoch:
            if self._current_finetuning_step >= self.fine_tune_for_epochs:
                pruning_dict = self._run_pruning()
                # TODO: Log pruned filters to Tensorboard
                self._current_finetuning_step = 0
            else:
                self._current_finetuning_step += 1

    def _run_pruning(self) -> dict:
        print("Running filter pruning...")
        surgeon = kerassurgeon.Surgeon(self.model, copy=True)
        pruning_dict = dict()
        for layer in self.model.layers:
            if layer.__class__.__name__ == "Conv2D":
                if re.match(self.prunable_layers_regex, layer.name):
                    nb_pruned_filters = self.run_pruning_for_conv_layer(layer, surgeon)
                    pruning_dict[layer.name] = nb_pruned_filters
        new_model = surgeon.operate()

        new_model.save("asdasd.h5")
        del self.model
        # K.clear_session()
        tf.reset_default_graph()

        self.model = models.load_model("asdasd.h5")
        self._model_compile_function(self.model)
        return pruning_dict

    @abc.abstractmethod
    def run_pruning_for_conv_layer(self, layer, surgeon: kerassurgeon.Surgeon) -> int:
        raise NotImplementedError
