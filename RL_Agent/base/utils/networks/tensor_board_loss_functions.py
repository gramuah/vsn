import tensorflow as tf

def loss_sumaries(loss, names, step):
    if isinstance(loss, list):
        with tf.name_scope('Losses'):
            for l, n in zip(loss, names):
                tf.summary.scalar(n, l, step=step)

def rl_loss_sumaries(data, names, step):
    with tf.name_scope('RL_Values'):
        for d, n in zip(data, names):
            with tf.name_scope(n):
                tf.summary.histogram('histogram', d, step=step)
                tf.summary.scalar('mean', tf.reduce_mean(d), step=step)
                tf.summary.scalar('std', tf.math.reduce_std(d), step=step)
                tf.summary.scalar('max', tf.reduce_max(d), step=step)
                tf.summary.scalar('min', tf.reduce_min(d), step=step)

def rl_sumaries(data, names, step):
    with tf.name_scope('RL'):
        for d, n in zip(data, names):
            with tf.name_scope(n):
                tf.summary.scalar(n, d, step=step)

def variable_summaries(name, var, e):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(str(name)):
        with tf.name_scope('summaries'):
            histog_summary = tf.summary.histogram('histogram', var, step=e)