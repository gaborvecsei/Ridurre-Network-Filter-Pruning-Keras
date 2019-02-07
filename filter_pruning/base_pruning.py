import abc
import re

from keras import callbacks


class BaseFilterPruning(callbacks.Callback):
    def __init__(self, start_at_epoch: int, fine_tune_for_epochs: int, prunable_layers_regex: str):
        super().__init__()

        self.start_at_epoch = start_at_epoch
        self.fine_tune_for_epochs = fine_tune_for_epochs
        self.prunable_layers_regex = prunable_layers_regex

        self._current_finetuning_step = fine_tune_for_epochs

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
        pruning_dict = dict()
        for layer in self.model.layers:
            if layer.__class__.__name__ == "Conv2D":
                if re.match(self.prunable_layers_regex, layer.name):
                    nb_pruned_filters = self.run_pruning_for_conv_layer(layer)
                    pruning_dict[layer.name] = nb_pruned_filters
        return pruning_dict

    @abc.abstractmethod
    def run_pruning_for_conv_layer(self, layer) -> int:
        raise NotImplementedError
