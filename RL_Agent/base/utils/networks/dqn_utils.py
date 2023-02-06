import numpy as np


def dqn_calc_target(done, reward, next_obs, gamma, target_pred):
    """
    Calculate the target values forcing the DQN training process to affect only to the actions selected.
    :param done: (bool). Flag for episode finished. True if next_obs is a final state.
    :param reward: (float). Reward for the action taken in the current state.
    :param next_obs:  (numpy nd array). Next Observation (Next State), numpy arrays with state shape.
    """
    target_aux = (reward + gamma *
                  np.amax(target_pred, axis=1))
    target = reward

    not_done = [not i for i in done]
    target__aux = target_aux * not_done
    target = done * target
    return target__aux + target

def ddqn_calc_target(done, reward, next_obs, gamma, pred, target_pred):
    """
    Calculate the target values forcing the DQN training process to affect only to the actions selected.
    :param done: (bool). Flag for episode finished. True if next_obs is a final state.
    :param reward: (float). Reward for the action taken in the current state.
    :param next_obs:  (numpy nd array). Next Observation (Next State), numpy arrays with state shape.
    """
    armax = np.argmax(pred, axis=1)
    target_value = target_pred
    values = []

    for i in range(target_value.shape[0]):
        values.append(target_value[i][armax[i]])

    target_aux = (reward + gamma * np.array(values))
    target = reward

    not_done = [not i for i in done]
    target__aux = target_aux * not_done
    target = done * target

    return target__aux + target
