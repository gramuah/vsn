import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

mse = tf.keras.losses.MeanSquaredError()

@tf.function
def dqn_loss(y, y_):
    return mse(y, y_), []
    # return tf.math.reduce_mean(tf.math.square(y-y_))

@tf.function(experimental_relax_shapes=True)
def dpg_loss(pred, actions, returns):
    log_prob = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)(actions, pred)
    loss = tf.math.reduce_mean(log_prob * returns)
    return loss, [log_prob, returns]

@tf.function(experimental_relax_shapes=True)
def dpg_loss_continuous(pred, actions, returns):
    prob = tfp.distributions.Normal(pred, tf.math.reduce_std(pred))
    log_prob = - prob.log_prob(actions)
    returns = tf.expand_dims(returns, axis=-1)

    loss = tf.math.reduce_mean(log_prob * returns)
    return loss, [log_prob, returns]

@tf.function(experimental_relax_shapes=True)
def dpg_loss_continuous_beta(pred, actions, returns):
    alpha_pred, beta_pred = tf.split(pred, num_or_size_splits=2, axis=1)
    dist = tfp.distributions.Beta(alpha_pred, beta_pred)
    log_prob = - dist.log_prob(actions)
    returns = tf.expand_dims(returns, axis=-1)

    loss = tf.math.reduce_mean(log_prob * returns)
    return loss, [log_prob, returns]

@tf.function(experimental_relax_shapes=True)
def ddpg_actor_loss(values):
    loss = -tf.math.reduce_mean(values)
    return loss, [values]


@tf.function(experimental_relax_shapes=True)
def ddpg_critic_loss(q, q_target):
    loss = mse(q, q_target)
    return loss, []

@tf.function(experimental_relax_shapes=True)
def log10(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
  return numerator / denominator

@tf.function(experimental_relax_shapes=True)
def log2(x):
  numerator = tf.math.log(x)
  denominator = tf.math.log(tf.constant(2, dtype=numerator.dtype))
  return numerator / denominator

# @tf.function(experimental_relax_shapes=True)
# def ppo_loss_continuous(y_true, y_pred, advantage, old_prediction, returns, values, stddev=1.0, loss_clipping=0.3,
#                         critic_discount=0.5, entropy_beta=0.001):
#     """
#     f(x) = (1/σ√2π)exp(-(1/2σ^2)(x−μ)^2)
#     X∼N(μ, σ)
#     """
#     dist_new = tfp.distributions.Normal(tf.math.reduce_mean(y_pred), tf.math.maximum(tf.math.reduce_std(y_pred), 1.0))
#     # dist_new = tfp.distributions.Normal(tf.math.reduce_mean(y_true), tf.math.maximum(tf.math.reduce_std(y_true), 0.4))
#     dist_old = tfp.distributions.Normal(tf.math.reduce_mean(old_prediction), tf.math.maximum(tf.math.reduce_std(old_prediction), 0.4))
#
#     new_prob = dist_new.prob(y_true)
#     old_prob = dist_old.prob(y_true)
#
#     log_new_prob = dist_new.log_prob(y_true)
#     log_old_prob = dist_old.log_prob(y_true)
#     # new_prob = dist_new.prob(y_pred)
#     # old_prob = dist_old.prob(y_pred)
#
#     # log_new_prob = dist_new.log_prob(y_pred)
#     # log_old_prob = dist_old.log_prob(y_pred)
#
#     ratio = tf.exp(log_new_prob - log_old_prob)  # pnew / pold
#     # ratio = (new_prob) / (old_prob + 1e-20)
#
#     p1 = tf.math.multiply(ratio, advantage)
#     p2 = tf.math.multiply(tf.clip_by_value(ratio, clip_value_min=1 - loss_clipping,
#                                            clip_value_max=1 + loss_clipping), advantage)
#
#     actor_loss = tf.reduce_mean(tf.math.minimum(p1, p2))
#     critic_loss = tf.reduce_mean(tf.math.square(returns - values))
#     shannon_entropy = dist_new.entropy()
#
#     _y_t = y_true.numpy()
#     _y_p = y_pred.numpy()
#     _a_ = advantage.numpy()
#     _o_p = old_prediction.numpy()
#     _r_ = returns.numpy()
#     _v_ = values.numpy()
#     _n_p = new_prob.numpy()
#     _o_p = old_prob.numpy()
#     _l_n_p = log_new_prob.numpy()
#     _l_o_p = log_old_prob.numpy()
#     _rat = ratio.numpy()
#     _p_min = tf.math.minimum(p1, p2).numpy()
#     _p1_ = p1.numpy()
#     _p2_ = p2.numpy()
#     _a_l = actor_loss.numpy()
#     _c_l = critic_loss.numpy()
#     _s_e_ = shannon_entropy.numpy()
#     _std_n = tf.math.reduce_std(y_true).numpy()
#     _std_o = tf.math.reduce_std(old_prediction).numpy()
#
#     if np.any(np.isnan(_a_l)) or np.any(np.isinf(_a_l)):
#         print('actor_loss = nan or inf')
#     if np.any(np.isnan(_s_e_)) or np.any(np.isinf(_s_e_)):
#         print('entropy_loss = nan or inf')
#
#     return -actor_loss + critic_discount * critic_loss - entropy_beta * shannon_entropy, [-actor_loss, critic_loss, -shannon_entropy]

@tf.function(experimental_relax_shapes=True)
def ppo_loss_continuous(y_true, y_pred, advantage, old_prediction, returns, values, stddev=1.0, loss_clipping=0.3,
                        critic_discount=0.5, entropy_beta=0.001):
    """
    f(x) = (1/σ√2π)exp(-(1/2σ^2)(x−μ)^2)
    X∼N(μ, σ)
    """
    # If stddev < 1.0 can appear probabilities greater than 1.0 and negative entropy values.
    # stddev = tf.math.maximum(stddev, 1.0)
    # var = tf.math.square(stddev)
    var = tf.math.maximum(tf.math.square(tf.math.reduce_std(y_pred)), 0.4)
    var_old = tf.math.maximum(tf.math.square(tf.math.reduce_std(old_prediction)), 0.4)
    # var = tf.math.square(tf.math.reduce_std(y_pred))
    # var_old = tf.math.square(tf.math.reduce_std(old_prediction))
    pi = np.pi

    # σ√2π
    denom = tf.math.sqrt(2 * pi * var)
    denom_old = tf.math.sqrt(2 * pi * var_old)

    # exp(-((x−μ)^2/2σ^2))
    prob_num = tf.math.exp(- tf.math.square(y_true - y_pred) / (2 * var))
    old_prob_num = tf.math.exp(- tf.math.square(y_true - old_prediction) / (2 * var_old))

    # exp(-((x−μ)^2/2σ^2))/(σ√2π)
    new_prob = prob_num / denom
    old_prob = old_prob_num / denom_old

    # ratio = (new_prob) / (old_prob + 1e-20)
    ratio = tf.exp(tf.math.log(new_prob) - tf.math.log(old_prob))  # pnew / pold

    p1 = ratio * advantage
    p2 = tf.math.multiply(tf.clip_by_value(ratio, clip_value_min=1 - loss_clipping,
                                           clip_value_max=1 + loss_clipping), advantage)

    actor_loss = tf.reduce_mean(tf.math.minimum(p1, p2))
    critic_loss = tf.reduce_mean(tf.math.square(returns - values))
    entropy = new_prob * log2(new_prob + 1e-20)
    shannon_entropy = -tf.reduce_mean(entropy)

    # _y_t = y_true.numpy()
    # _y_p = y_pred.numpy()
    # _a_ = advantage.numpy()
    # _o_p = old_prediction.numpy()
    # _n_p = new_prob.numpy()
    # _r_ = returns.numpy()
    # _v_ = values.numpy()
    # _o_p = old_prob.numpy()
    # _rat = ratio.numpy()
    # _p_min = tf.math.minimum(p1, p2).numpy()
    # _p1_ = p1.numpy()
    # _p2_ = p2.numpy()
    # _a_l = actor_loss.numpy()
    # _c_l = critic_loss.numpy()
    # _s_e_ = shannon_entropy.numpy()
    # _std_n = tf.math.reduce_std(y_true).numpy()
    # _std_o = tf.math.reduce_std(old_prediction).numpy()
    #
    # if np.any(np.isnan(_a_l)) or np.any(np.isinf(_a_l)):
    #     print('actor_loss = nan or inf')
    # if np.any(np.isnan(_s_e_)) or np.any(np.isinf(_s_e_)):
    #     print('entropy_loss = nan or inf')

    return -actor_loss + critic_discount * critic_loss - entropy_beta * shannon_entropy, [-actor_loss, critic_loss, -shannon_entropy]

@tf.function(experimental_relax_shapes=True)
def ppo_loss_continuous_beta(y_true, y_pred, advantage, old_prediction, returns, values, stddev=1.0, loss_clipping=0.3,
                        critic_discount=0.5, entropy_beta=0.001):
    """
    f(x) = (1/σ√2π)exp(-(1/2σ^2)(x−μ)^2)
    X∼N(μ, σ)
    """
    alpha_pred, beta_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)
    alpha_old, beta_old = tf.split(old_prediction, num_or_size_splits=2, axis=1)

    new_dist = tfp.distributions.Beta(alpha_pred, beta_pred)
    old_dist = tfp.distributions.Beta(alpha_old, beta_old)

    new_prob = new_dist.prob(y_true)
    old_prob = old_dist.prob(y_true)

    # ratio = (new_prob) / (old_prob + 1e-20)
    ratio = tf.exp(tf.math.log(new_prob) - tf.math.log(old_prob))  # pnew / pold

    p1 = ratio * advantage
    p2 = tf.math.multiply(tf.clip_by_value(ratio, clip_value_min=1 - loss_clipping,
                                           clip_value_max=1 + loss_clipping), advantage)

    actor_loss = tf.reduce_mean(tf.math.minimum(p1, p2))
    critic_loss = tf.reduce_mean(tf.math.square(returns - values))
    entropy = new_prob * log2(new_prob + 1e-20)
    shannon_entropy = -tf.reduce_mean(entropy)

    # _y_t = y_true.numpy()
    # _y_p = y_pred.numpy()
    # _a_ = advantage.numpy()
    # _o_p = old_prediction.numpy()
    # _n_p = new_prob.numpy()
    # _r_ = returns.numpy()
    # _v_ = values.numpy()
    # _o_p = old_prob.numpy()
    # _rat = ratio.numpy()
    # _p_min = tf.math.minimum(p1, p2).numpy()
    # _p1_ = p1.numpy()
    # _p2_ = p2.numpy()
    # _a_l = actor_loss.numpy()
    # _c_l = critic_loss.numpy()
    # _s_e_ = shannon_entropy.numpy()
    # _std_n = tf.math.reduce_std(y_true).numpy()
    # _std_o = tf.math.reduce_std(old_prediction).numpy()
    #
    # if np.any(np.isnan(_a_l)) or np.any(np.isinf(_a_l)):
    #     print('actor_loss = nan or inf')
    # if np.any(np.isnan(_s_e_)) or np.any(np.isinf(_s_e_)):
    #     print('entropy_loss = nan or inf')

    return -actor_loss + critic_discount * critic_loss - entropy_beta * shannon_entropy, [-actor_loss, critic_loss, -shannon_entropy]

# TODO: Al hacer esta pérdida correctamente se desconecta el grafo por que hay que usar fsolve para sacar alpha y beta
# @tf.function(experimental_relax_shapes=True)
def ppo_loss_continuous_beta_explore(y_true, y_pred, advantage, old_prediction, returns, values, stddev=1.0, loss_clipping=0.3,
                        critic_discount=0.5, entropy_beta=0.001):
    """
    f(x) = (1/σ√2π)exp(-(1/2σ^2)(x−μ)^2)
    X∼N(μ, σ)
    """
    alpha_pred, beta_pred = tf.split(y_pred, num_or_size_splits=2, axis=1)
    alpha_old, beta_old = tf.split(old_prediction, num_or_size_splits=2, axis=1)

    new_dist = tfp.distributions.Beta(alpha_pred, beta_pred)
    old_dist = tfp.distributions.Beta(alpha_old, beta_old)

    new_prob = new_dist.prob(y_true)
    old_prob = old_dist.prob(y_true)

    # ratio = (new_prob) / (old_prob + 1e-20)
    ratio = tf.exp(tf.math.log(new_prob) - tf.math.log(old_prob))  # pnew / pold

    p1 = ratio * advantage
    p2 = tf.math.multiply(tf.clip_by_value(ratio, clip_value_min=1 - loss_clipping,
                                           clip_value_max=1 + loss_clipping), advantage)

    actor_loss = tf.reduce_mean(tf.math.minimum(p1, p2))
    critic_loss = tf.reduce_mean(tf.math.square(returns - values))
    entropy = new_prob * log2(new_prob + 1e-20)
    shannon_entropy = -tf.reduce_mean(entropy)

    # _y_t = y_true.numpy()
    # _y_p = y_pred.numpy()
    # _a_ = advantage.numpy()
    # _o_p = old_prediction.numpy()
    # _n_p = new_prob.numpy()
    # _r_ = returns.numpy()
    # _v_ = values.numpy()
    # _o_p = old_prob.numpy()
    # _rat = ratio.numpy()
    # _p_min = tf.math.minimum(p1, p2).numpy()
    # _p1_ = p1.numpy()
    # _p2_ = p2.numpy()
    # _a_l = actor_loss.numpy()
    # _c_l = critic_loss.numpy()
    # _s_e_ = shannon_entropy.numpy()
    # _std_n = tf.math.reduce_std(y_true).numpy()
    # _std_o = tf.math.reduce_std(old_prediction).numpy()
    #
    # if np.any(np.isnan(_a_l)) or np.any(np.isinf(_a_l)):
    #     print('actor_loss = nan or inf')
    # if np.any(np.isnan(_s_e_)) or np.any(np.isinf(_s_e_)):
    #     print('entropy_loss = nan or inf')

    return -actor_loss + critic_discount * critic_loss - entropy_beta * shannon_entropy, [-actor_loss, critic_loss, -shannon_entropy]
# @tf.function(experimental_relax_shapes=True)
# def ppo_loss_continuous(y_true, y_pred, advantage, old_prediction, returns, values, stddev=1.0, loss_clipping=0.3,
#                         critic_discount=0.5, entropy_beta=0.001):
#     """
#     f(x) = (1/σ√2π)exp(-(1/2σ^2)(x−μ)^2)
#     X∼N(μ, σ)
#     """
#     dist_new = tfp.distributions.Normal(tf.math.reduce_mean(y_pred), tf.math.reduce_std(y_pred))
#     dist_old = tfp.distributions.Normal(tf.math.reduce_mean(old_prediction), tf.math.reduce_std(old_prediction))
#
#     new_prob = dist_new.prob(y_true)
#     old_prob = dist_old.prob(y_true)
#
#     ratio = (new_prob) / (old_prob + 1e-20)
#
#
#
#     p1 = tf.math.multiply(ratio, advantage)
#     p2 = tf.math.multiply(tf.clip_by_value(ratio, clip_value_min=1 - loss_clipping,
#                                            clip_value_max=1 + loss_clipping), advantage)
#
#     actor_loss = tf.reduce_mean(tf.math.minimum(p1, p2))
#     critic_loss = tf.reduce_mean(tf.math.square(returns - values))
#     shannon_entropy = dist_new.entropy()
#
#     _y_t = y_true.numpy()
#     _y_p = y_pred.numpy()
#     _a_ = advantage.numpy()
#     _o_p = old_prediction.numpy()
#     _r_ = returns.numpy()
#     _v_ = values.numpy()
#     _n_p = new_prob.numpy()
#     _o_p = old_prob.numpy()
#     _rat = ratio.numpy()
#     _p1_ = p1.numpy()
#     _p2_ = p2.numpy()
#     _a_l = actor_loss.numpy()
#     _c_l = critic_loss.numpy()
#     _s_e_ = shannon_entropy.numpy()
#     return -actor_loss + critic_discount * critic_loss - entropy_beta * shannon_entropy, [-actor_loss, critic_loss, -shannon_entropy]

@tf.function(experimental_relax_shapes=True)
def ppo_loss_discrete(y_true, y_pred, advantage, old_prediction, returns, values,
                                               stddev=None, loss_clipping=0.3, critic_discount=0.5, entropy_beta=0.001):
    # new_prob = tf.math.multiply(y_true, y_pred)
    # new_prob = tf.reduce_mean(new_prob, axis=-1)
    new_prob = tf.reduce_sum(y_true * y_pred, axis=-1)
    # old_prob = tf.math.multiply(y_true, old_prediction)
    # old_prob = tf.reduce_mean(old_prob, axis=-1)
    old_prob = tf.reduce_sum(y_true * old_prediction, axis=-1)

    # ratio = tf.math.divide(new_prob + 1e-10, old_prob + 1e-10)
    # ratio = K.exp(K.log(new_prob + 1e-10) - K.log(old_prob + 1e-10))
    ratio = (new_prob) / (old_prob + 1e-10)

    # p1 = tf.math.multiply(ratio, advantage)
    # p2 = tf.math.multiply(tf.clip_by_value(ratio, clip_value_min=1 - self.loss_clipping,
    #                       clip_value_max=1 + self.loss_clipping), advantage)
    p1 = ratio * advantage
    p2 = tf.clip_by_value(ratio, clip_value_min=1 - loss_clipping, clip_value_max=1 + loss_clipping) * advantage

    # actor_loss = tf.reduce_mean(tf.math.minimum(p1, p2))
    # critic_loss = tf.reduce_mean(tf.math.square(returns - values))
    # entropy = tf.reduce_mean(-(new_prob * tf.math.log(new_prob + 1e-10)))
    actor_loss = tf.math.reduce_mean(tf.math.minimum(p1, p2))
    critic_loss = tf.math.reduce_mean(tf.math.square(returns - values))
    entropy = tf.math.reduce_mean(-(new_prob * tf.math.log(new_prob + 1e-10)))

    # y_p = y_pred.numpy()
    # y_t = y_true.numpy()
    # n_p = new_prob.numpy()
    # o_p = old_prob.numpy()
    # r = ratio.numpy()
    # p1_ = p1.numpy()
    # p2_ = p2.numpy()
    # a_l = actor_loss.numpy()
    # c_l = critic_loss.numpy()
    # e = entropy.numpy()
    # o = old_prediction.numpy()

    return - actor_loss + critic_discount * critic_loss - entropy_beta * entropy, [actor_loss, critic_loss, entropy]


@tf.function(experimental_relax_shapes=True)
def a2c_actor_loss(log_prob, adv, entropy_beta, entropy):
    loss = - tf.math.reduce_mean(log_prob * adv)
    entropy = - tf.math.reduce_mean(entropy)
    return tf.math.reduce_mean(loss + (entropy*entropy_beta)), [loss, entropy]

@tf.function(experimental_relax_shapes=True)
def a2c_actor_loss_beta(y_true, y_pred , values, returns, entropy_beta=0.0):
    alpha_, beta_ = tf.split(y_pred, num_or_size_splits=2, axis=1)
    dist = tfp.distributions.Beta(alpha_, beta_)

    log_prob = dist.log_prob(y_true)
    adv = returns - values
    entropy = dist.entropy()
    loss = - tf.math.reduce_mean(log_prob * adv)
    entropy = - tf.math.reduce_mean(entropy)
    return tf.math.reduce_mean(loss + (entropy*entropy_beta)), [loss, entropy]

@tf.function(experimental_relax_shapes=True)
def a2c_critic_loss(y, y_):
    return mse(y, y_), []
