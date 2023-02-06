import datetime
import os

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from RL_Agent.base.utils import net_building
from RL_Agent.base.utils.networks.default_networks import ppo_net
from RL_Agent.base.agent_base import AgentSuper
from RL_Agent.base.utils.networks.networks_interface import RLNetModel as RLNetModel
import tensorflow as tf
from RL_Agent.base.utils.networks.agent_networks import PPONet


# worker class that inits own environment, trains on it and updloads weights to global net
class PPOSuper(AgentSuper):
    """
    Super class for implementing Proximal Policy Optimization (PPO) agents.
    Abstract class as a base for implementing PPO algorithms.
    """

    def __init__(self, actor_lr, critic_lr, batch_size, epsilon=0., epsilon_decay=0., epsilon_min=0.,
                 gamma=0.95, n_step_return=10, memory_size=512, loss_clipping=0.2, loss_critic_discount=0.5,
                 loss_entropy_beta=0.001, lmbda=0.95, train_epochs=1, exploration_noise=1.0, n_stack=1, img_input=False,
                 is_habitat=False, state_size=None, net_architecture=None, n_threads=None, tensorboard_dir=None,
                 train_action_selection_options=None, action_selection_options=None):
        """
        Super class for implementing Proximal Policy Optimization (PPO) agents.

        :param actor_lr: (float) learning rate for training the actor NN.
        :param critic_lr: (float) learning rate for training the critic NN.
        :param batch_size: (int) Size of training batches.
        :param epsilon: (float in [0., 1.]) exploration-exploitation rate during training. epsilon=1.0 -> Exploration,
            epsilon=0.0 -> Exploitation.
        :param epsilon_decay: (float or func) Exploration-exploitation rate
            reduction factor. If float, it reduce epsilon by multiplication (new epsilon = epsilon * epsilon_decay). If
            func it receives (epsilon, epsilon_min) as arguments and it is applied to return the new epsilon value
            (float)
        :param epsilon_min: (float, [0., 1.])  Minimum exploration-exploitation rate allowed ing training.
        :param gamma: (float) Discount or confidence factor for target value estimation.
        :param n_step_return: (int > 0) Number of steps used for calculating the return.
        :param memory_size: (int) Size of experiences memory.
        :param loss_clipping: (float > 0) Clipping factor of PPO loss function. Controls how much the updated policy
            differs from the previous policy for each training iteration.
        :param loss_critic_discount: (float > 0) Factor of importance of the critic loss for the actor network. The
            actor loss is defined as: actor_loss + loss_critic_discount * critic_loss + loss_entropy_beta * entropy_loss.
        :param loss_entropy_beta: (float > 0) Factor of importance of the entropy term for the actor network loss
            function. The actor loss is defined as: actor_loss + loss_critic_discount * critic_loss + loss_entropy_beta
            * entropy_loss. Entropy term is used to improve the exploration, higher values will result in a more
            explorative training process.
        :param lmbda: (float) PPO lambda factor.
        :param train_epochs: (int > 0) Number of epochs for training the agent network in each iteration of the algorithm.
        :param exploration_noise: (float [0, 1]) Maximum value of noise for action selection in exploration mode. By
            default is used as maximum stddev for selecting actions from a normal distribution during exploration and it
            is multiplied by epsilon to reduce the stddev. This result on exploration factor reduction through the time
            steps.
        :param n_stack: (int) Number of time steps stacked on the state.
        :param img_input: (bool) Flag for using a images as states. If True, the states are supposed to be images (3D
            array).
        : param is habitat: (bool) Flag to specify if the observations come from an habitat environment.
        :param state_size: (tuple of ints) State size. Only needed if the original state size is modified by any
            preprocessing. Shape of the state that must match network's inputs. This shape must include the number of
            stacked states.
        :param tensorboard_dir: (str) path to store tensorboard summaries.
        :param net_architecture: (dict) Define the net architecture. Is recommended use dicts from
            IL_Problem.base.utils.networks.networks_dictionaries.py.
        :param n_threads: (int) Number of threads to run where a multithread agent is used.
        :param train_action_selection_options: (func) How to select the actions in exploration mode. This allows to
            change the exploration method used acting directly over the actions selected by the neural network or
            adapting the action selection procedure to an especial neural network. Some usable functions and
            documentation on how to implement your own function on RL_Agent.base.utils.networks.action_selection_options.
        :param action_selection_options:(func) How to select the actions in exploitation mode. This allows to change or
            modify the actions selection procedure acting directly over the actions selected by the neural network or
            adapting the action selection procedure to an especial neural network. Some usable functions and
            documentation on how to implement your own function on RL_Agent.base.utils.networks.action_selection_options.
        """
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma, n_step_return=n_step_return,
                         memory_size=memory_size, loss_clipping=loss_clipping,
                         loss_critic_discount=loss_critic_discount, loss_entropy_beta=loss_entropy_beta, lmbda=lmbda,
                         train_epochs=train_epochs, exploration_noise=exploration_noise, n_stack=n_stack,
                         img_input=img_input, is_habitat=is_habitat, state_size=state_size, n_threads=n_threads,
                         tensorboard_dir=tensorboard_dir, net_architecture=net_architecture,
                         train_action_selection_options=train_action_selection_options,
                         action_selection_options=action_selection_options)

    def build_agent(self, state_size, n_actions, stack=False):
        """
        Define the agent params, structure, architecture, neural nets ...
        :param state_size: (tuple of ints) State size. Only needed if the original state size is modified by any
            preprocessing. Shape of the state that must match network's inputs. This shape must include the number of
            stacked states.
        :param n_actions: (int) Number of action of the agent.
        :param stack: (bool) If True, the input states are supposed to be stacked (various time steps).
        """
        super().build_agent(state_size=state_size, n_actions=n_actions, stack=stack)

        self.memory = []
        self.loss_selected = None  # Select the discrete or continuous version

    def variable_summaries(self, var):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            mean_summary = tf.summary.scalar('mean', mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            stddev_summary = tf.summary.scalar('stddev', stddev)
            max_summary = tf.summary.scalar('max', tf.reduce_max(var))
            min_summary = tf.summary.scalar('min', tf.reduce_min(var))
            histog_summary = tf.summary.histogram('histogram', var)

        return tf.summary.merge([mean_summary, stddev_summary, max_summary, min_summary, histog_summary])

    def remember(self, obs, action, pred_act, rewards, mask):
        """
        Store a memory in a list of memories
        :param obs: Current Observation (State)
        :param action: Action selected with noise
        :param pred_act: Action predicted
        :param reward: Reward
        :param next_obs: Next Observation (Next State)
        :param done: If the episode is finished
        :return:
        """
        if self.is_habitat:
            if self.n_stack > 1:
                rgb = []
                target_encoding = []
                for stack_obs in obs:
                    rgb.append([o['rgb'].astype(np.float32) for o in stack_obs])
                    target_encoding.append(stack_obs[0]['objectgoal'].astype(np.float32))
            else:
                rgb = [ob['rgb'] for ob in obs]
                target_encoding = [ob['objectgoal'] for ob in obs]
            # TODO: CARLOS -> add rgb and target encoding as an observation
            obs = [rgb, target_encoding]
        else:
            obs = np.array(obs)
        action = np.array(action)
        pred_act = np.array([a[0] for a in pred_act])
        rewards = np.array(rewards)
        mask = np.array(mask)

        index = range(len(obs))
        self.memory = [obs, action, pred_act, rewards, mask]

    def remember_multithread(self, obs, action, pred_act, rewards, mask):
        """
        Store a memory in a list of memories
        :param obs: Current Observation (State)
        :param action: Action selected with noise
        :param pred_act: Action predicted
        :param reward: Reward
        :param next_obs: Next Observation (Next State)
        :param done: If the episode is finished
        :return:
        """

        if self.img_input:
            # TODO: Probar img en color en pc despacho, en personal excede la memoria
            if len(np.array(obs).shape) > 4:
                obs = np.transpose(obs, axes=(1, 0, 2, 3, 4))
            else:
                obs = np.transpose(obs, axes=(1, 0, 2, 3))
        elif self.is_habitat:
            rgb = obs['rgb']
            if len(np.array(rgb).shape) > 4:
                rgb = np.transpose(rgb, axes=(1, 0, 2, 3, 4))
            else:
                rgb = np.transpose(rgb, axes=(1, 0, 2, 3))
            # TODO: CARLOS -> add rgb and target encoding as an observation
            target_encoding = obs['objectgoal']
            obs = rgb
        elif self.stack:
            obs = np.transpose(obs, axes=(1, 0, 2, 3))
        else:
            obs = np.transpose(obs, axes=(1, 0, 2))

        if len(np.array(pred_act).shape) > 3:
            action = np.transpose(action, axes=(1, 0, 2, 3))
        else:
            action = np.transpose(action, axes=(1, 0, 2))
        if len(np.array(pred_act).shape) > 3:
            pred_act = np.transpose(pred_act, axes=(1, 0, 2, 3))
        else:
            pred_act = np.transpose(pred_act, axes=(1, 0, 2))
        rewards = np.transpose(rewards, axes=(1, 0))
        mask = np.transpose(mask, axes=(1, 0))

        o = obs[0]
        a = action[0]
        p_a = pred_act[0]
        r = rewards[0]
        m = mask[0]

        # TODO: Optimizar, es muy lento
        for i in range(1, self.n_threads):
            o = np.concatenate((o, obs[i]), axis=0)
            a = np.concatenate((a, action[i]), axis=0)
            p_a = np.concatenate((p_a, pred_act[i]), axis=0)
            r = np.concatenate((r, rewards[i]), axis=0)
            m = np.concatenate((m, mask[i]), axis=0)

        # TODO: Decidir la solución a utilizar
        index = range(len(o))
        # index = np.random.choice(range(len(obs)), self.buffer_size, replace=False)
        self.memory = [o, a, p_a, r, m]

    def load_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        obs = self.memory[0]
        action = self.memory[1]
        pred_act = self.memory[2]
        rewards = self.memory[3]
        mask = self.memory[4]

        return obs, action, pred_act, rewards, mask

    def replay(self):
        """"
        Training process
        """
        obs, action, old_prediction, rewards, mask = self.load_memories()

        # TODO: [Sergio] Hay alguna manera de hacer esto mejor? Me parece un parche esta soluci
        if not self.is_habitat:
            obs = np.float32(obs)

        # TODO: Aqui tienen que entrar las variables correspondientes, de momento entran las que hay disponibles.
        actor_loss, critic_loss = self.model.fit(obs,
                                                 obs,  # TODO: [Sergio] aquí habria que pasar nex_obs pero no se están guardando en el collect_memories
                                                 np.float32(action),
                                                 np.float32(rewards),
                                                 np.float32(np.logical_not(mask)),
                                                 epochs=self.train_epochs,
                                                 batch_size=self.batch_size,
                                                 shuffle=True,
                                                 verbose=True,
                                                 kargs=[np.float32(old_prediction),
                                                        np.float32(mask),
                                                        np.float32(self.exploration_noise * self.epsilon),
                                                        np.float32(self.loss_clipping),
                                                        np.float32(self.critic_discount),
                                                        np.float32(self.entropy_beta),
                                                        np.float32(self.gamma),
                                                        np.float32(self.lmbda)])

        self._reduce_epsilon()
        return actor_loss, critic_loss

    def _load_protobuf(self, path):
        """
        Loads the neural networks of the agent.
        :param path: (str) path to folder to load the network
        :param checkpoint: (bool) If True the network is loaded as Tensorflow checkpoint, otherwise the network is
                                   loaded in protobuffer format.
        """
        self.model.restore_from_protobuf(path)
        print("Loaded model from disk")

    def _load(self, path, checkpoint=False):
        """
        Loads the neural networks of the agent.
        :param path: (str) path to folder to load the network
        :param checkpoint: (bool) If True the network is loaded as Tensorflow checkpoint, otherwise the network is
                                   loaded in protobuffer format.
        """
        self.model.restore(path)
        # if checkpoint:
        #     # Load a checkpoint
        #     actor_chkpoint = tf.train.Checkpoint(model=self.model.actor_net)
        #     actor_manager = tf.train.CheckpointManager(actor_chkpoint,
        #                                                os.path.join(path, 'actor', 'checkpoint'),
        #                                                checkpoint_name='actor',
        #                                                max_to_keep=3)
        #     actor_chkpoint.restore(actor_manager.latest_checkpoint)
        #
        #     critic_chkpoint = tf.train.Checkpoint(model=self.model.critic_net)
        #     critic_manager = tf.train.CheckpointManager(critic_chkpoint,
        #                                                os.path.join(path, 'critic', 'checkpoint'),
        #                                                checkpoint_name='critic',
        #                                                max_to_keep=3)
        #     critic_chkpoint.restore(critic_manager.latest_checkpoint)
        # else:
        #     # Load a protobuffer
        #     self.model.actor_net = tf.saved_model.load(os.path.join(path, 'actor'))
        #     self.model.critic_net = tf.saved_model.load(os.path.join(path, 'critic'))
        print("Loaded model from disk")

    def _save_protobuf(self, path):
        """
        Saves the neural networks of the agent.
        :param path: (str) path to folder to store the network
        :param checkpoint: (bool) If True the network is stored as Tensorflow checkpoint, otherwise the network is
                                    stored in protobuffer format.
        """
        # if checkpoint:
        #     # Save a checkpoint
        #     actor_chkpoint = tf.train.Checkpoint(model=self.model.actor_net)
        #     actor_manager = tf.train.CheckpointManager(actor_chkpoint,
        #                                                os.path.join(path, 'actor', 'checkpoint'),
        #                                                checkpoint_name='actor',
        #                                                max_to_keep=3)
        #     save_path = actor_manager.save()
        #
        #     critic_chkpoint = tf.train.Checkpoint(model=self.model.critic_net)
        #     critic_manager = tf.train.CheckpointManager(critic_chkpoint,
        #                                                os.path.join(path, 'critic', 'checkpoint'),
        #                                                checkpoint_name='critic',
        #                                                max_to_keep=3)
        #     critic_manager.save()
        # else:
        # Save a protobuffer

        self.model.export_to_protobuf(path)
        # tf.saved_model.save(self.model.actor_net, os.path.join(path, 'actor'))
        # tf.saved_model.save(self.model.critic_net, os.path.join(path, 'critic'))

        print("Model saved  to disk")
        print(datetime.datetime.now())

    def _save_network(self, path):
        """
        Saves the neural networks of the agent.
        :param path: (str) path to folder to store the network
        :param checkpoint: (bool) If True the network is stored as Tensorflow checkpoint, otherwise the network is
                                    stored in protobuffer format.
        """
        # if checkpoint:
        #     # Save a checkpoint
        #     actor_chkpoint = tf.train.Checkpoint(model=self.model.actor_net)
        #     actor_manager = tf.train.CheckpointManager(actor_chkpoint,
        #                                                os.path.join(path, 'actor', 'checkpoint'),
        #                                                checkpoint_name='actor',
        #                                                max_to_keep=3)
        #     save_path = actor_manager.save()
        #
        #     critic_chkpoint = tf.train.Checkpoint(model=self.model.critic_net)
        #     critic_manager = tf.train.CheckpointManager(critic_chkpoint,
        #                                                os.path.join(path, 'critic', 'checkpoint'),
        #                                                checkpoint_name='critic',
        #                                                max_to_keep=3)
        #     critic_manager.save()
        # else:
        # Save a protobuffer

        self.model.save(path)
        # tf.saved_model.save(self.model.actor_net, os.path.join(path, 'actor'))
        # tf.saved_model.save(self.model.critic_net, os.path.join(path, 'critic'))

        print("Saved model to disk")
        print(datetime.datetime.now())

    def _build_model(self, net_architecture, last_activation):
        # Neural Net for Actor-Critic Model
        if net_architecture is None:  # Standart architecture
            net_architecture = ppo_net
            define_output_layer = False
        else:
            define_output_layer = net_architecture['define_custom_output_layer']

        # Building actor
        if self.is_habitat:
            # TODO: [Sergio] Revisar state_size cuando usamos habitat. Incluir la dim. de imagen y del objectgoal
            actor_net = net_building.build_conv_net(net_architecture, self.state_size, actor=True)
        elif self.img_input:
            actor_net = net_building.build_conv_net(net_architecture, self.state_size, actor=True)
        elif self.stack:
            actor_net = net_building.build_stack_net(net_architecture, self.state_size, actor=True)
        else:
            actor_net = net_building.build_nn_net(net_architecture, self.state_size, actor=True)

        if isinstance(actor_net, RLNetModel):
            agent_model = actor_net
            actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
            critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)

            agent_model.compile(optimizer=[actor_optimizer, critic_optimizer],
                                loss=self.loss_selected)
        else:

            # Building actor
            if self.is_habitat:
                critic_model = net_building.build_conv_net(net_architecture, self.state_size, critic=True)
            elif self.img_input:
                critic_model = net_building.build_conv_net(net_architecture, self.state_size, critic=True)
            elif self.stack:
                critic_model = net_building.build_stack_net(net_architecture, self.state_size, critic=True)
            else:
                critic_model = net_building.build_nn_net(net_architecture, self.state_size, critic=True)

            if not define_output_layer:
                actor_net.add(Dense(self.n_actions, name='output', activation=last_activation))
            if not define_output_layer:
                critic_model.add(Dense(1))

            agent_model = PPONet(actor_net=actor_net, critic_net=critic_model, tensorboard_dir=self.tensorboard_dir)

            actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
            critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)

            agent_model.compile(optimizer=[actor_optimizer, critic_optimizer],
                                loss=self.loss_selected)
            agent_model.summary()

        return agent_model

    def _create_hist(self, loss):
        """
        Clase para suplantar tf.keras.callbacks.History() cuando no se está usando keras.
        """

        class historial:
            def __init__(self, loss):
                self.history = {"loss": [loss]}

        return historial(loss)

    def proximal_policy_optimization_loss_continuous(self, y_true, y_pred, advantage, old_prediction, returns, values,
                                                     stddev=None):

        """
        f(x) = (1/σ√2π)exp(-(1/2σ^2)(x−μ)^2)
        X∼N(μ, σ)
        """
        # If stddev < 1.0 can appear probabilities greater than 1.0 and negative entropy values.
        stddev = tf.maximum(stddev, 1.0)
        var = tf.math.square(stddev)
        # var = K.square(stddev)
        pi = 3.1415926

        # σ√2π
        # denom = K.sqrt(2 * pi * var)
        denom = tf.math.sqrt(2 * pi * var)

        # exp(-((x−μ)^2/2σ^2))
        # prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
        # old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))
        prob_num = tf.math.exp(- K.square(y_true - y_pred) / (2 * var))
        old_prob_num = tf.math.exp(- K.square(y_true - old_prediction) / (2 * var))

        # exp(-((x−μ)^2/2σ^2))/(σ√2π)
        new_prob = prob_num / denom
        old_prob = old_prob_num / denom

        # ratio = K.exp(K.log(new_prob + 1e-10) - K.log(old_prob + 1e-10))
        ratio = (new_prob) / (old_prob + 1e-20)

        p1 = ratio * advantage
        # p2 = K.clip(ratio, min_value=1 - self.loss_clipping, max_value=1 + self.loss_clipping) * advantage
        p2 = tf.math.multiply(tf.clip_by_value(ratio, clip_value_min=1 - self.loss_clipping,
                                               clip_value_max=1 + self.loss_clipping), advantage)

        # actor_loss = K.mean(K.minimum(p1, p2))
        # critic_loss = K.mean(K.square(returns - values))
        # entropy = K.mean(-(new_prob * K.log(new_prob + 1e-10)))

        actor_loss = tf.reduce_mean(tf.math.minimum(p1, p2))
        critic_loss = tf.reduce_mean(tf.math.square(returns - values))
        entropy = tf.reduce_mean(-(new_prob * tf.math.log(new_prob + 1e-10)))

        return -actor_loss + self.critic_discount * critic_loss - self.entropy_beta * entropy

    def proximal_policy_optimization_loss_discrete(self, y_true, y_pred, advantage, old_prediction, returns, values,
                                                   stddev=None):
        # new_prob = tf.math.multiply(y_true, y_pred)
        # new_prob = tf.reduce_mean(new_prob, axis=-1)
        new_prob = K.sum(y_true * y_pred, axis=-1)
        # old_prob = tf.math.multiply(y_true, old_prediction)
        # old_prob = tf.reduce_mean(old_prob, axis=-1)
        old_prob = K.sum(y_true * old_prediction, axis=-1)

        # ratio = tf.math.divide(new_prob + 1e-10, old_prob + 1e-10)
        # ratio = K.exp(K.log(new_prob + 1e-10) - K.log(old_prob + 1e-10))
        ratio = (new_prob) / (old_prob + 1e-20)

        # p1 = tf.math.multiply(ratio, advantage)
        # p2 = tf.math.multiply(tf.clip_by_value(ratio, clip_value_min=1 - self.loss_clipping,
        #                       clip_value_max=1 + self.loss_clipping), advantage)
        p1 = ratio * advantage
        p2 = K.clip(ratio, min_value=1 - self.loss_clipping, max_value=1 + self.loss_clipping) * advantage

        # actor_loss = tf.reduce_mean(tf.math.minimum(p1, p2))
        # critic_loss = tf.reduce_mean(tf.math.square(returns - values))
        # entropy = tf.reduce_mean(-(new_prob * tf.math.log(new_prob + 1e-10)))
        actor_loss = K.mean(K.minimum(p1, p2))
        critic_loss = K.mean(K.square(returns - values))
        entropy = K.mean(-(new_prob * K.log(new_prob + 1e-10)))

        return - actor_loss + self.critic_discount * critic_loss - self.entropy_beta * entropy

    def _tensorboard_aux_loss_continuous(self, advantage, old_prediction, rewards, values, stddev):

        def loss(y_true, y_pred):
            """
            f(x) = (1/σ√2π)exp(-(1/2σ^2)(x−μ)^2)
            X∼N(μ, σ)
            """
            # var = np.square(self.exploration_noise*self.epsilon)
            var = np.square(stddev)
            pi = 3.1415926

            # σ√2π
            denom = stddev * np.sqrt(2. * pi)

            # exp(-((x−μ)^2/2σ^2))
            prob_num = np.exp((-1. / 2.) * np.square((y_true - y_pred) / (stddev)))
            old_prob_num = np.exp((-1. / 2.) * np.square((y_true - old_prediction) / (stddev)))

            # exp(-((x−μ)^2/2σ^2))/(σ√2π)
            new_prob = prob_num / denom
            old_prob = old_prob_num / denom

            # ratio = np.exp(np.log(new_prob + 1e-10) - np.log(old_prob + 1e-10))
            ratio = (new_prob) / (old_prob + 1e-20)

            p1 = ratio * advantage
            p2 = np.clip(ratio, a_min=1 - self.loss_clipping, a_max=1 + self.loss_clipping) * advantage
            actor_loss = np.mean(np.minimum(p1, p2))
            critic_loss = np.mean(np.square(rewards - values))
            entropy = np.mean(-(new_prob * np.log(new_prob + 1e-10)))
            # entropy = np.mean(-(new_prob * np.log(new_prob + 1e-10)))

            return actor_loss, critic_loss, entropy, ratio, p1, p2, var, np.mean(new_prob)

        return loss

    def get_advantages(self, values, masks, rewards):
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

    def _format_obs_act_multithread(self, obs):
        if self.img_input:
            if self.stack:
                obs = np.array([np.dstack(o) for o in obs])
            else:
                obs = obs


        elif self.stack:
            # obs = obs.reshape(-1, *self.state_size)
            obs = obs
        else:
            # obs = obs.reshape(-1, self.state_size)
            obs = obs

        return obs

    # TODO: behavioral cloning fit
    def bc_fit_legacy(self, expert_traj, epochs, batch_size, learning_rate=1e-3, shuffle=False, optimizer=Adam(),
                      loss='mse',
                      validation_split=0.15):

        expert_traj_s = np.array([x[0] for x in expert_traj])
        expert_traj_a = np.array([x[1] for x in expert_traj])
        expert_traj_a = self._actions_to_onehot(expert_traj_a)
        dummy_advantage = np.zeros((expert_traj_a.shape[0], 1))
        dummy_old_prediction = np.zeros(expert_traj_a.shape)
        dummy_rewards = np.zeros((expert_traj_a.shape[0], 1))
        dummy_values = np.zeros((expert_traj_a.shape[0], 1))
        optimizer.lr = learning_rate
        self.model.compile(optimizer=optimizer, loss=loss)
        hist = self.model.fit([expert_traj_s, dummy_advantage, dummy_old_prediction, dummy_rewards, dummy_values],
                              [expert_traj_a], batch_size=batch_size, shuffle=shuffle, epochs=epochs, verbose=2,
                              validation_split=validation_split)

    def _actions_to_onehot(self, actions):
        return actions

    def _reduce_epsilon(self):
        if isinstance(self.epsilon_decay, float):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_decay(self.epsilon, self.epsilon_min)
