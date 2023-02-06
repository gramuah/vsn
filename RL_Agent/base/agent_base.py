import numpy as np
from abc import ABCMeta, abstractmethod
import tensorflow as tf


class AgentInterface(object, metaclass=ABCMeta):
    """
    This class is an interface for building reinforcement learning agents. Here are the definitions of the methods that
    are required for an agent to work in the library.
    """
    def __init__(self):
        self.state_size = None  # (tuple) size and shape of states.
        self.n_actions = None  # (int) Number of actions.
        self.stack = None  # (bool) True means that a sequence of input in contiguous time steps are stacked to form the state.
        self.img_input = None  # (bool) Set to True when states are images (3D array), False otherwise.
        self.agent_name = None   # (str) id/name of the agent.

    @abstractmethod
    def build_agent(self):
        """
        Define the agent params, structure, architecture, neural nets and agent_builded flag ...
        """
        pass

    def compile(self):
        pass

    @abstractmethod
    def act_train(self, obs):
        """
        Select an action given an observation :param obs: (numpy nd array) observation or state.
        :return: (int or [floats]) int if actions are discrete or numpy array of float of action shape if actions are
            continuous)
        """
        pass

    @abstractmethod
    def act(self, obs):
        """
        Select an action given an observation in only exploitation mode.
        :return: (int or [floats]) int if actions are discrete or numpy array of float of action shape if actions are
            continuous)
        """
        pass

    @abstractmethod
    def remember(self, obs, action, reward, next_obs, done):
        """
        Store an experience in memory for training the agent.
        :param obs: (numpy nd array). Current Observation (State), numpy array with state shape.
        :param action: (int, [int] or [floats]) Action selected, numpy array of actions. If actions are discrete an
            unique int can be used or a hot encoded array of ints. If actions are continuous an array of float should
            be used.
        :param reward: (float). Reward for the action taken in the current state.
        :param next_obs:  (numpy nd array). Next Observation (Next State), numpy arrays with state shape.
        :param done: (bool). Flag for episode finished. True if next_obs is a final state.
        """
        pass

    @abstractmethod
    def replay(self):
        """
        Run the train step for the agent.
        """
        pass

    @abstractmethod
    def _load(self, path):
        """
        Load a tensorflow or keras model.
        :param path: (str) file name
        """
        pass

    @abstractmethod
    def _save_network(self, path):
        """
        Save a tensorflow or keras model.
        :param path: (str) file name
        """
        pass

    def bc_fit(self, expert_traj_s, expert_traj_a, epochs, batch_size, shuffle, optimizer, loss, metrics,
               validation_split, verbose):
        """
        Behavioral cloning training procedure for the neural network.
        :param expert_traj_s: (nd array) observations from expert demonstrations.
        :param expert_traj_a: (nd array) actions from expert demonstrations.
        :param epochs: (int) Training epochs.
        :param batch_size: (int) Training batch size.
        :param shuffle: (bool) Shuffle or not the examples on expert_traj.
        :param optimizer: (keras optimizer o keras optimizer id) Optimizer to be used in training procedure.
        :param loss: (keras loss id, keras loss or custom loss based on keras losses interface) Loss metrics for the
                     training procedure.
        :param loss: (keras metric or custom metric based on keras metrics interface) Metrics for the
                     training procedure.
        :param validation_split: (float in [0., 1.]) Fraction of data to be used for validation.
        :param verbose: (int [0, 2]) Set verbosity of the function. 0 -> no verbosity.
                                                                    1 -> batch level verbosity.
                                                                    2 -> epoch level verbosity.
        :param one_hot_encode_actions: (bool) If True, expert_traj_a will be transformed into one hot encoding.
                                              If False, expert_traj_a will be no altered. Useful for discrete actions.
        """
        pass

    def copy_model_to_target(self):
        """
        Copy the main neural network model to a target model for stabilizing the training process.
        This is not an abstract method because may be not needed.
        """
        pass


class AgentSuper(AgentInterface):
    """
    All agents in this library inherit from this class. Here can be found basic and useful utilities for agents
    implementation.
    """
    def __init__(self, learning_rate=None, actor_lr=None, critic_lr=None, batch_size=None, epsilon=None,
                 epsilon_decay=None, epsilon_min=None, gamma=None, tau=None, n_step_return=None, memory_size=None,
                 loss_clipping=None, loss_critic_discount=None, loss_entropy_beta=None, lmbda=None, train_epochs=None,
                 exploration_noise=None, n_stack=None, img_input=None, is_habitat=None, state_size=None, n_threads=None,
                 tensorboard_dir=None, net_architecture=None, train_action_selection_options=None,
                 action_selection_options=None):
        """
        Abstract agent class for defining the principal attributes of an rl agent.
        :param learning_rate: (float) learning rate for training the agent NN. Not used if actor_lr or critic_lr are 
            defined.
        :param actor_lr: (float) learning rate for training the actor NN of an Actor-Critic agent.
        :param critic_lr: (float) learning rate for training the critic NN of an Actor-Critic agent.
        :param batch_size: (int) Size of training batches.
        :param epsilon: (float in [0., 1.]) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
       :param epsilon_decay: (float or func) Exploration-exploitation rate reduction factor. If float, it reduce epsilon
            by multiplication (new epsilon = epsilon * epsilon_decay). If func it receives (epsilon, epsilon_min) as
            arguments and it is applied to return the new epsilon value (float).
        :param epsilon_min: (float, [0., 1.])  Minimum exploration-exploitation rate allowed ing training.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param tau: (float) Transference factor between main and target discriminator.
        :param n_step_return: (int) Number of steps used for calculating the return.
        :param memory_size: (int) Size of experiences memory.
        :param loss_clipping: (float) Loss clipping factor for PPO.
        :param loss_critic_discount: (float) Discount factor for critic loss of PPO.
        :param loss_entropy_beta: (float) Discount factor for entropy loss of PPO.
        :param lmbda: (float) PPO lambda factor.
        :param train_steps: (int) Train steps for each training iteration.
        :param exploration_noise: (float) Standard deviation of a normal distribution for selecting actions during PPO
            training.
        :param n_stack: (int) Number of time steps stacked on the state (observation stacked).
        :param img_input: (bool) Flag for using a images as states. True state are images (3D array).
        : param is habitat: (bool) Flag to specify if the observations come from an habitat environment.
        :param state_size: (tuple of ints) State size. Needed if the original environment state size is modified by any
            preprocessing.
        :param n_threads: (int) Number of parallel environments when using A3C or PPO. By default number of cpu
            kernels are selected.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            RL_Agent.base.utils.networks
        # :param loads_saved_params: (bool) If True when loading from a checkpoint all the agent parameters will be loaded
        #                             in the state they was saved. If False the new specified parameters are maintained
        #                             when a saved agent is loaded.
        """
        super().__init__()

        self.learning_rate = learning_rate
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr

        self.batch_size = batch_size

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.gamma = gamma
        self.tau = tau

        self.memory_size = memory_size
        self.loss_clipping = loss_clipping
        self.critic_discount = loss_critic_discount
        self.entropy_beta = loss_entropy_beta
        self.lmbda = lmbda
        self.train_epochs = train_epochs
        self.exploration_noise = exploration_noise

        self.n_step_return = n_step_return

        self.n_stack = n_stack
        self.img_input = img_input
        self.is_habitat = is_habitat
        self.state_size = state_size
        self.env_state_size = state_size

        self.n_threads = n_threads

        self.net_architecture = net_architecture

        self.save_if_better = True

        self.optimizer = None

        self.tensorboard_dir = tensorboard_dir
        self.agent_builded = False
        # self.loads_saved_params = loads_saved_params

        self.train_action_selection_options = train_action_selection_options
        self.action_selection_options = action_selection_options

        self.model = None   # Neural network model. The model should inherits from
                            # RL_Agent.base.utils.networks.networks_interface.RLNetModel

    def build_agent(self, state_size, n_actions, stack):
        """
        Define the agent params, structure, architecture, neural nets ...
        :param state_size: (tuple of ints) State size.
        :param n_actions: (int) Number of actions.
        :param stack: (bool) True means that a sequence of input in contiguous time steps are stacked in the state.
        """
        super().build_agent()

        self.state_size = state_size
        self.n_actions = n_actions
        self.stack = stack
        self.agent_builded = True

    def compile(self):
        super().compile()

    def _format_obs_act(self, obs):
        """
        Reshape the observation (state) to fits the neural network inputs.
        :param obs: Observation (state) array of state shape.
        :return: (nd array)
        """
        if self.is_habitat:  # and isinstance(obs, dict):
            # TODO: [Sergio] Formatear y estandarizar correctamente los tipos de estradas para habitat
            if self.n_stack > 1:
                rgb = [[o['rgb'].astype(np.float32) for o in obs]]
                goal = [obs[0]['objectgoal'].astype(np.float32)]
                obs = [rgb, goal]
            else:
                # TODO: CARLOS -> format habitat inputs to the neural networks
                obs = [[obs['rgb'].astype(np.float32)], [obs['objectgoal'].astype(np.float32)]]
            return obs

        elif self.img_input:
            if self.stack:
                obs = np.dstack(obs)
            obs = np.array([obs])

        elif self.stack:
            obs = np.array([obs])
        else:
            obs = np.array(obs).reshape(-1, self.state_size)

        return obs.astype(np.float32)

    def set_batch_size(self, batch_size):
        """
        Method for change the batch size for the neural net training.
        :param batch_size: (int).
        """
        self.batch_size = batch_size

    def set_gamma(self, gamma):
        """
        Method for change the discount or confidence factor for target state value.
        :param gamma: (float).
        """
        self.gamma = gamma

    def set_train_steps(self, train_epochs):
        """
        Method for change the number of train steps for the neural net in each training execution of the neural net.
        :param train_epochs: (int).
        """
        self.train_epochs = train_epochs

    def set_optimizer(self, opt):
        """
        Method for change optimizer used for the neural net training.
        :param opt: (keras optimizer or keras optimizer id).
        """
        self.optimizer = opt

    def bc_fit(self, expert_traj_s, expert_traj_a, epochs, batch_size, shuffle=False,
               optimizer=tf.keras.optimizers.Adam(name="bc_adam"), loss=tf.keras.losses.MeanSquaredError(name="bc_mse"),
               metrics=tf.metrics.Accuracy(name="bc_acc"), validation_split=0., verbose=1,
               one_hot_encode_actions=False):
        """
        Behavioral cloning training procedure for the neural network.
        :param expert_traj_s: (nd array) observations from expert demonstrations.
        :param expert_traj_a: (nd array) actions from expert demonstrations.
        :param epochs: (int) Training epochs.
        :param batch_size: (int) Training batch size.
        :param shuffle: (bool) Shuffle or not the examples on expert_traj.
        :param optimizer: (keras optimizer o keras optimizer id) Optimizer to be used in training procedure.
        :param loss: (keras loss id, keras loss or custom loss based on keras losses interface) Loss metrics for the
                     training procedure.
        :param loss: (keras metric or custom metric based on keras metrics interface) Metrics for the
                     training procedure.
        :param validation_split: (float in [0., 1.]) Fraction of data to be used for validation.
        :param verbose: (int [0, 2]) Set verbosity of the function. 0 -> no verbosity.
                                                                    1 -> batch level verbosity.
                                                                    2 -> epoch level verbosity.
        :param one_hot_encode_actions: (bool) If True, expert_traj_a will be transformed into one hot encoding.
                                              If False, expert_traj_a will be no altered. Useful for discrete actions.
        """
        if one_hot_encode_actions:
            expert_traj_a = self._actions_to_onehot(expert_traj_a)
        self.model.bc_compile(optimizer=[optimizer], loss=[loss], metrics=metrics)
        self.model.bc_fit(expert_traj_s, expert_traj_a, batch_size=batch_size, shuffle=shuffle, epochs=epochs,
                       validation_split=validation_split, verbose=verbose)

    def _actions_to_onehot(self, actions):
        action_matrix = []
        for action in actions:
            action_aux = np.zeros(self.n_actions)
            action_aux[action] = 1
            action_matrix.append(action_aux)
        return np.array(action_matrix)

    def save_tensorboar_rl_histogram(self, histograms):
        if self.model.train_summary_writer is not None:
            with self.model.train_summary_writer.as_default():
                for [total_episodes, episodic_reward, steps, success, epsilon, global_steps] in histograms:
                    self.model.rl_sumaries([episodic_reward, success, epsilon, steps],
                                           ['Reward', 'Success', 'Epsilon', 'Steps per episode'],
                                           global_steps)

    def _reduce_epsilon(self):
        """
        Reduce the exploration rate.
        """
        if isinstance(self.epsilon_decay, float):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_decay(self.epsilon, self.epsilon_min)

