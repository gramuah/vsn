from RL_Agent.base.agent_base import AgentInterface
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from RL_Agent.base.utils import net_building
from RL_Agent.base.utils.networks.default_networks import ppo_net


# worker class that inits own environment, trains on it and updloads weights to global net
class Agent(AgentInterface):
    def __init__(self, state_size, n_actions, stack=False, img_input=False, lr_actor=0.0001, lr_critic=0.001,
                 action_bound=None, batch_size=32, buffer_size=2048, net_architecture=None):
        self.state_size = state_size
        self.n_actions = n_actions
        self.stack = stack
        self.img_input = img_input

        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = 0.99
        self.batch_size = batch_size
        self.action_bound = action_bound
        self.buffer_size = buffer_size
        self.loss_clipping = 0.2
        self.train_epochs = 10
        self.exploraion_noise = 1.0
        self.actor, self.critic = self._build_model(net_architecture)
        # self.critic = self.build_critic()
        # self.actor = self.build_actor_continuous()
        self.memory = []
        self.epsilon = 0.0  # Only for rendering

        self.dummy_action, self.dummy_value = np.zeros((1, self.n_actions)), np.zeros((1, 1))

    def act(self, obs):
        if self.img_input or self.stack:
            obs = obs.reshape(-1, *self.state_size)
        else:
            obs = obs.reshape(-1, self.state_size)

        p = self.actor.predict([obs, self.dummy_value, self.dummy_action])
        action = action_matrix = p[0] + np.random.normal(loc=0, scale=self.exploraion_noise, size=p[0].shape)
        return action, action_matrix, p

    def act(self, obs):
        if self.img_input or self.stack:
            obs = obs.reshape(-1, *self.state_size)
        else:
            obs = obs.reshape(-1, self.state_size)
        p = self.actor.predict([obs, self.dummy_value, self.dummy_action])
        action = p[0]
        return action

    def remember(self, obs, action, pred_act, reward):
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
        obs, action, pred_act, reward = np.array(obs), np.array(action), np.array(pred_act), np.reshape(
            np.array(reward), (len(reward), 1))
        pred_act = np.reshape(pred_act, (pred_act.shape[0], pred_act.shape[2]))

        # TODO: Decidir la solución a utilizar
        index = range(self.buffer_size)
        # index = np.random.choice(range(len(obs)), self.buffer_size, replace=False)
        self.memory = [obs[index], action[index], pred_act[index], reward[index]]

    def load_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        # memory = np.array(self.memory)
        # index = range(self.buffer_size)  #np.random.choice(range(len(self.memory)), size=self.buffer_size, replace=False)
        # obs, action, pred_act, reward = memory[index, 0], memory[index, 1], memory[index, 2], memory[index, 3]
        # obs = np.array([x.reshape(self.state_size) for x in obs])
        # action = np.array([x.reshape(self.n_actions) for x in action])
        # pred_act = np.array([x.reshape(self.n_actions) for x in pred_act])
        # reward = np.reshape(reward, (-1, 1))
        # obs, action, pred, reward = obs[:self.buffer_size], action[:self.buffer_size], pred_act[:self.buffer_size], \
        #                             reward[:self.buffer_size]
        # self.memory = []
        # return obs, action, pred_act, reward
        return self.memory[0], self.memory[1], self.memory[2], self.memory[3]

    def replay(self):
        """"
        Training process
        """
        obs, action, pred, reward = self.load_memories()
        old_prediction = pred
        pred_values = self.critic.predict(obs)

        advantage = reward - pred_values

        actor_loss = self.actor.fit([obs, advantage, old_prediction], [action], batch_size=self.batch_size, shuffle=True,
                                    epochs=self.train_epochs, verbose=False)
        critic_loss = self.critic.fit([obs], [reward], batch_size=self.batch_size, shuffle=True, epochs=self.train_epochs,
                                      verbose=False)

        return actor_loss, critic_loss

    def _load(self, path):
        self.worker._load(path)

    def _save_network(self, path):
        self.saver._save_network(self.sess)

    def _build_model(self, net_architecture):
        # Neural Net for Actor-Critic Model
        if net_architecture is None:  # Standart architecture
            net_architecture = ppo_net

        # Building actor
        if self.img_input:
            actor_net = net_building.build_conv_net(net_architecture, self.state_size, actor=True)
        elif self.stack:
            actor_net = net_building.build_stack_net(net_architecture, self.state_size, actor=True)
        else:
            actor_net = net_building.build_nn_net(net_architecture, self.state_size, actor=True)

        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(self.n_actions,))

        actor_net.add(Dense(self.n_actions, name='output', activation='tanh'))

        actor_model = Model(inputs=[actor_net.inputs, advantage, old_prediction], outputs=[actor_net.outputs])
        actor_model.compile(optimizer=Adam(lr=self.lr_actor),
                      loss=[self.proximal_policy_optimization_loss_continuous(
                          advantage=advantage,
                          old_prediction=old_prediction)])
        actor_model.summary()

        # Building actor
        if self.img_input:
            critic_model = net_building.build_conv_net(net_architecture, self.state_size, critic=True)
        elif self.stack:
            critic_model = net_building.build_stack_net(net_architecture, self.state_size, critic=True)
        else:
            critic_model = net_building.build_nn_net(net_architecture, self.state_size, critic=True)

        critic_model.add(Dense(1))
        critic_model.compile(optimizer=Adam(lr=self.lr_critic), loss='mse')

        return actor_model, critic_model

    def proximal_policy_optimization_loss_continuous(self, advantage, old_prediction):

        def loss(y_true, y_pred):
            var = K.square(self.exploraion_noise)
            pi = 3.1415926
            denom = K.sqrt(2 * pi * var)
            prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
            old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

            prob = prob_num / denom
            old_prob = old_prob_num / denom
            r = prob / (old_prob + 1e-10)

            return -K.mean(K.minimum(r * advantage,
                                     K.clip(r, min_value=1 - self.loss_clipping, max_value=1 + self.loss_clipping) * advantage))

        return loss

    def compute_reward_return(self, reward):
        for j in range(len(reward) - 2, -1, -1):
            reward[j] += reward[j + 1] * self.gamma
        return reward
