from pathlib import Path
from typing import Union, Tuple

import tensorflow as tf
from keras import callbacks
from keras.utils import layer_utils
from swiss_army_tensorboard import tfboard_loggers


def calculate_flops_and_parameters(model_session: tf.Session, verbose: int = 0) -> Tuple[int, int]:
    profiler_output = "stdout" if verbose > 0 else "none"

    run_meta = tf.RunMetadata()

    opts_dict = tf.profiler.ProfileOptionBuilder.float_operation()
    opts_dict["output"] = profiler_output
    flops = tf.profiler.profile(model_session.graph, run_meta=run_meta, cmd='op', options=opts_dict)

    opts_dict = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
    opts_dict["output"] = profiler_output
    params = tf.profiler.profile(model_session.graph, run_meta=run_meta, cmd='op', options=opts_dict)

    return flops.total_float_ops, params.total_parameters


class ModelComplexityCallback(callbacks.Callback):
    def __init__(self, log_dir: Union[str, Path], model_session: tf.Session, verbose: int = 1):
        super().__init__()

        log_dir = str(log_dir)
        self._flops_logger = tfboard_loggers.TFBoardScalarLogger(log_dir + "/flops")
        self._params_logger = tfboard_loggers.TFBoardScalarLogger(log_dir + "/params")
        self._model_session = model_session
        self._verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        flops, params = calculate_flops_and_parameters(self._model_session, verbose=0)
        self._flops_logger.log_scalar("model_flops", flops, epoch)
        self._params_logger.log_scalar("model_params", params, epoch)

        if self._verbose > 0:
            print("FLOPS at epoch {0}: {1:,}".format(epoch, flops))
            print("Number of PARAMS at epoch {0}: {1:,}".format(epoch, params))


class ModelParametersCallback(callbacks.Callback):
    def __init__(self, log_dir: Union[str, Path], sub_folder: str = "parameters", verbose: int = 1):
        super().__init__()

        log_dir = str(log_dir) + "/" + sub_folder
        self._params_logger = tfboard_loggers.TFBoardScalarLogger(log_dir)
        self._verbose = verbose

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        trainable_count = layer_utils.count_params(self.model.trainable_weights)
        non_trainable_count = layer_utils.count_params(self.model.non_trainable_weights)

        self._params_logger.log_scalar("trainable_parameters", trainable_count, epoch)
        self._params_logger.log_scalar("non_trainable_parameters", non_trainable_count, epoch)

        if self._verbose > 0:
            print("Trainable PARAMS at epoch {0}: {1:,}".format(epoch, trainable_count))
            print("Non trainable PARAMS at epoch {0}: {1:,}".format(epoch, non_trainable_count))
