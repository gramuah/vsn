import tensorflow as tf
import numpy as np

def discount_and_norm_rewards(rewards, mask, gamma, norm=True, n_step_return=None):
    # TODO: Revisar si la funcion se puede implementar más eficiente para el caso de n_step_return=n
    """
    Calculate the return as cumulative discounted rewards of an episode.
    :param episode_rewards: ([float]) List of rewards of an episode.
    """
    discounted_episode_rewards = np.zeros_like(rewards)

    # Calculate cumulative returns for n-steps of each trajectory
    if n_step_return is not None:
        cumulative_return = 0
        for i in reversed(range(rewards.size-n_step_return, rewards.size)):
            cumulative_return = rewards[i] + cumulative_return * gamma * mask[i]
            discounted_episode_rewards[i] = cumulative_return

        for i in reversed(range(rewards.size-n_step_return)):
            cumulative_return = 0
            for j in reversed(range(i, i + n_step_return)):
                cumulative_return = rewards[j] + cumulative_return * gamma * mask[j]
            discounted_episode_rewards[i] = cumulative_return
    else:
        cumulative_return = 0
        # Calculate cumulative returns for entire trajectories
        for i in reversed(range(rewards.size)):
            cumulative_return = rewards[i] + cumulative_return * gamma * mask[i]
            discounted_episode_rewards[i] = cumulative_return

    if norm:
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards) + 1e-10  # para evitar valores cero
    return discounted_episode_rewards

def discount_and_norm_rewards_legacy(rewards, mask, gamma, norm=True, n_step_return=None):
    # TODO: Revisar si la funcion discount_and_norm_rewards desde arriba se puede implementar más eficiente
    """
    Calculate the return as cumulative discounted rewards of an episode.
    :param episode_rewards: ([float]) List of rewards of an episode.
    """
    discounted_episode_rewards = np.zeros_like(rewards)
    cumulative_return = 0
    # Calculate cumulative returns for entire trajectories
    for i in reversed(range(rewards.size)):
        cumulative_return = rewards[i] + cumulative_return * gamma * mask[i]
        discounted_episode_rewards[i] = cumulative_return

    discounted_episode_rewards = np.zeros_like(rewards)
    # Calculate cumulative returns for n-steps of each trajectory
    if n_step_return is not None:
        for i in reversed(range(rewards.size)):
            cumulative_return = 0
            for j in range(i, i - n_step_return):
                cumulative_return = rewards[j] + cumulative_return * gamma * mask[j]
                discounted_episode_rewards[i] = cumulative_return

    # Calculate cumulative returns for n-steps of each trajectory
    if n_step_return is not None:
        # Get the initial episodes indexes
        init_epi_index = np.where(mask == False)[0]+1

        # Add the initial index
        if init_epi_index.size > 0:
            if init_epi_index[0] != 0:
                init_epi_index = np.concatenate([[0], init_epi_index])

            if init_epi_index[-1] != discounted_episode_rewards.size:
                init_epi_index = np.concatenate([init_epi_index, [discounted_episode_rewards.size]])
        else:
            init_epi_index = np.concatenate([[0], [discounted_episode_rewards.size]])

        for j in range(init_epi_index.size-1):
            for i in range(init_epi_index[j], init_epi_index[j+1]-n_step_return):
                if i > init_epi_index[j+1]-n_step_return - 10:
                    print()
                discounted_episode_rewards[i] = discounted_episode_rewards[i] - \
                                                discounted_episode_rewards[i + n_step_return] * \
                                                np.power(gamma, n_step_return)

    if norm:
        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
        discounted_episode_rewards /= np.std(discounted_episode_rewards) + 1e-10  # para evitar valores cero
    return discounted_episode_rewards

def naive_n_step_return(rewards, mask, gamma, norm=True, n_step_return=None):
    """
    Calculate the return as cumulative discounted rewards of an episode.
    :param episode_rewards: ([float]) List of rewards of an episode.
    """
    n_step_return = 3
    n_step_discount_returns = np.zeros_like(rewards)
    for i in range(rewards.size):
        cumulative_return = 0
        max_step = i+n_step_return if rewards.size > i+n_step_return else rewards.size
        for j in range(i, max_step):
            cumulative_return = cumulative_return + rewards[j]*np.power(gamma, j-i)
        n_step_discount_returns[i] = cumulative_return

    if norm:
        n_step_discount_returns -= np.mean(n_step_discount_returns)
        n_step_discount_returns /= np.std(n_step_discount_returns) + 1e-10  # para evitar valores cero
    return n_step_discount_returns

def gae(values, masks, rewards, gamma, lmbda):
    returns, advantages = _gae(values, masks, rewards, gamma, lmbda)
    return returns.numpy(), advantages.numpy()

@tf.function
def _gae(values, masks, rewards, gamma, lmbda):
    """
    Generalized Advantage Estimation imlementation
    """
    values = tf.concat([values, [values[-1]]], axis=0)
    # returns = tf.Variable(tf.zeros(rewards.shape, tf.float32))
    returns = []
    gae = 0.
    for i in reversed(range(rewards.shape[0])):
        delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
        gae = delta + gamma * lmbda * masks[i] * gae
        returns.insert(0, gae + values[i])
    returns = tf.convert_to_tensor(returns, dtype=tf.float32)
    adv = returns - values[:-1]
    advantages = (adv - tf.math.reduce_mean(adv)) / (tf.math.reduce_std(adv) + 1e-10)
    return returns, advantages