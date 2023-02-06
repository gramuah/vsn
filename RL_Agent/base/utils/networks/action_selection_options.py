import random
import numpy as np
"""
These functions uses different strategies for choice the actions to perform given the outputs of the network. 
Each function have the next structure and  must be respected in order to implement a custom function:

def function(act_pred, n_actions, epsilon=0., n_env=1, exploration_noise=1.0):
    :param act_pred: (nd array of floats) network predictions.
    :param n_actions: (int) number of actions. In a discrete action configuration represent the number of possibles 
                    actions. In a continuous action configuration represent the number of actions 
                    to take simultaneously.
    :param epsilon: (float in range [0., 1.]) Exploration rate. Probability of selecting an exploitative action.            
    :param n_env: (int) Number of simultaneous environment in multithread agents. Also may be seen as the number of 
                    input states; if there is one state only an action is selected, if there is three (or multiple) 
                    states three (or multiple) actions must be selected.
    :param exploration_noise: (float in range [0., 1.]) Multiplier of exploration rate of scale of exploration. E.g.: 
                                Used for setting the stddev when sampling from a normal distribution.
"""
#########################################################################
#       Discrete action spaces
#########################################################################

def argmax(act_pred, n_actions, epsilon=0., n_env=1, exploration_noise=1.0):
    action = np.argmax(act_pred, axis=-1)
    return action

def random_choice(act_pred, n_actions, epsilon=0., n_env=1, exploration_noise=1.0):
    action = [np.random.choice(n_actions, p=act_pred[i]) for i in range(n_env)]
    return action

def greedy_action(act_pred, n_actions, epsilon=0., n_env=1, exploration_noise=1.0):
    if np.random.rand() <= epsilon:
        # TODO: al utilizar algoritmos normales puede fallar
        action = np.random.rand(*act_pred.shape)
        action = np.argmax(action, axis=-1)
    else:
        action = np.argmax(act_pred, axis=-1)
    return action

def greedy_random_choice(act_pred, n_actions, epsilon=0., n_env=1, exploration_noise=1.0):
    if np.random.rand() <= epsilon:
        action = [np.random.choice(n_actions) for i in range(n_env)]
    else:
        action = [np.random.choice(n_actions, p=act_pred[i]) for i in range(n_env)]
    return action

#########################################################################
#       Continuous action spaces
#########################################################################

def identity(act_pred, n_actions, epsilon=0., n_env=1, exploration_noise=1.0):
    return act_pred

def gaussian_noise(act_pred, n_actions, epsilon=0., n_env=1, exploration_noise=1.0):
    action = act_pred + np.random.normal(loc=0, scale=exploration_noise*epsilon, size=act_pred.shape)
    return action

def random_normal(act_pred, n_actions, epsilon=0., n_env=1, exploration_noise=0.5):
    action = np.random.normal(act_pred, np.clip(exploration_noise, 0., exploration_noise))
    return action

def beta_estimation(beta_estimator):

    def act_selection(act_pred, n_actions, epsilon=0., n_env=1, exploration_noise=0.5):
        beta_estimator.calculate_new_params_in_change(epsilon*exploration_noise)
        alpha, beta = beta_estimator.get_alpha_beta(act_pred)
        action = np.concatenate([alpha, beta], axis=-1)

        return action

    return act_selection
