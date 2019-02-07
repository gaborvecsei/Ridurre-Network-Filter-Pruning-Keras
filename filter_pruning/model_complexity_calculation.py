import tensorflow as tf
from keras import callbacks
from swiss_army_tensorboard import tfboard_loggers


def calculate_flops_and_parameters(model_session):
    run_meta = tf.RunMetadata()
    with model_session as sess:
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        return flops.total_float_ops, params.total_parameters


class ModelComplexityCallback(callbacks.Callback):
    def __init__(self, log_dir: str, model_session: tf.Session):
        super().__init__()

        self.flops_logger = tfboard_loggers.TFBoardScalarLogger(log_dir + "/flops")
        self.params_logger = tfboard_loggers.TFBoardScalarLogger(log_dir + "/params")
        self.model_session = model_session

    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)

        flops, params = calculate_flops_and_parameters(self.model_session)
        self.flops_logger.log_scalar("model_flops", flops, epoch)
        self.params_logger.log_scalar("model_params", params, epoch)
