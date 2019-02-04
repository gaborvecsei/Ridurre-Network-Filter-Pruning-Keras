import tensorflow as tf


def calculate_flops_and_parameters(model_session):
    run_meta = tf.RunMetadata()
    with model_session as sess:
        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        opts = tf.profiler.ProfileOptionBuilder.trainable_variables_parameter()
        params = tf.profiler.profile(sess.graph, run_meta=run_meta, cmd='op', options=opts)

        return flops.total_float_ops, params.total_parameters
