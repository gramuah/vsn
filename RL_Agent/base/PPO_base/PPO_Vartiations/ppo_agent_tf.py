from RL_Agent.base.agent_base import AgentInterface
import numpy as np
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.layers import Dense
from tensorflow.keras import backend as K
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
        self.critic_discount = 0.5
        self.entropy_beta = 0.001
        self.lmbda = 0.95
        self.train_epochs = 10
        self.exploration_noise = 1.0
        self._build_graph(net_architecture)
        # self.critic = self.build_critic()
        # self.actor = self.build_actor_continuous()
        self.memory = []
        self.epsilon = 0.0  # Only for rendering

        self.dummy_action, self.dummy_value = np.zeros((1, self.n_actions)), np.zeros((1, 1))

        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        self.sess.run(tf.global_variables_initializer())

    def act(self, obs):
        if self.img_input or self.stack:
            obs = obs.reshape(-1, *self.state_size)
        else:
            obs = obs.reshape(-1, self.state_size)

        # p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        dict = {self.state_t: obs}
        p = self.sess.run(self.actor_model, feed_dict=dict)
        action = action_matrix = p[0] + np.random.normal(loc=0, scale=self.exploration_noise, size=p[0].shape)
        value = self.sess.run(self.critic_model, feed_dict=dict)[0]
        # value = self.critic.predict([obs])[0]
        return action, action_matrix, p, value

    def act(self, obs):
        if self.img_input or self.stack:
            obs = obs.reshape(-1, *self.state_size)
        else:
            obs = obs.reshape(-1, self.state_size)
        # p = self.actor.predict([obs, self.dummy_value, self.dummy_action, self.dummy_value, self.dummy_value])
        dict = {self.state_t: obs}
        p = self.sess.run(self.actor_model, feed_dict=dict)
        action = p[0]
        return action

    def remember(self, obs, action, pred_act, rewards, values, mask):
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
        values.append(values[-1])
        returns, advantages = self.get_advantages(values, mask, rewards)
        obs = np.array(obs)
        action = np.array(action)
        pred_act = np.array(pred_act)
        pred_act = np.reshape(pred_act, (pred_act.shape[0], pred_act.shape[2]))
        returns = np.reshape(np.array(returns), (len(returns), 1))
        rewards = np.array(rewards).reshape((-1, 1))
        values = np.array(values).reshape((-1, 1))
        mask = np.array(mask).reshape((-1, 1))
        advantages = np.array(advantages).reshape((-1, 1))

        # TODO: Decidir la solución a utilizar
        index = range(len(obs))
        # index = np.random.choice(range(len(obs)), self.buffer_size, replace=False)
        self.memory = [obs[index], action[index], pred_act[index], returns[index], rewards[index], values[index],
                       mask[index], advantages[index]]

    def load_memories(self):
        """
        Load a batch of memories
        :return: Current Observation, Action, Reward, Next Observation, episode finished flag
        """
        obs = self.memory[0]
        action = self.memory[1]
        pred_act = self.memory[2]
        returns = self.memory[3]
        rewards = self.memory[4]
        values = self.memory[5]
        mask = self.memory[6]
        advantages = self.memory[7]

        return obs, action, pred_act, returns, rewards, values, mask, advantages
    def replay(self):
        """"
        Training process
        """
        obs, action, old_prediction, returns, rewards, values, mask, advantages = self.load_memories()

        # pred_values = self.critic.predict(obs)

        # advantage = returns - pred_values

        # actor_loss = self.actor.fit([obs, advantages, old_prediction, rewards, values], [action], batch_size=self.batch_size, shuffle=True,
        #                             epochs=self.train_epochs, verbose=False)
        # critic_loss = self.critic.fit([obs], [returns], batch_size=self.batch_size, shuffle=True, epochs=self.train_epochs,
        #                               verbose=False)
        # train the nework
        dict = {self.state_t: obs,
                self.advantage_t: advantages,
                self.old_prediction_t: old_prediction,
                self.values_t: values,
                self.actions_t: action,
                self.returns_t: returns
                }
        _, _, actor_loss, critic_loss = self.sess.run([self.fit_actor, self.fit_critic, self.actor_loss, self.critic_loss], feed_dict=dict)

        return actor_loss, critic_loss

    def _load(self, path):
        # Create a clean graph and import the MetaGraphDef nodes.
        # new_graph = tf.Graph()
        # with tf.keras.backend.get_session() as sess:
        # Import the previously export meta graph.
        name = path.join(path, name)
        loaded_model = tf.train.import_meta_graph(name + '.meta')
        # tf.keras.backend.clear_session()
        sess = tf.keras.backend.get_session()
        loaded_model.restore(sess, tf.train.latest_checkpoint(path + "./"))
        print("Loaded model from disk")

    def _save_network(self, path):
        # sess = tf.keras.backend.get_session()  # op_input_list=(self.actor.get_layers(), self.critic.get_layers())
        # self.saver = tf.train.Saver()
        # name = name + "-" + str(reward)
        # self.saver.save(sess, name)

        print("Saved model to disk")

    def _build_graph(self, net_architecture):
        # Building actor
        if self.img_input:
            self.state_t = tf.placeholder(tf.float32, shape=(None, *self.state_size), name='state')
        elif self.stack:
            self.state_t = tf.placeholder(tf.float32, shape=(None, *self.state_size), name='state')
        else:
            self.state_t = tf.placeholder(tf.float32, shape=(None, self.state_size), name='state')

        self.actions_t = tf.placeholder(tf.float32, shape=(None, self.n_actions), name='old_pred')
        self.old_prediction_t = tf.placeholder(tf.float32, shape=(None, self.n_actions), name='old_pred')
        self.advantage_t = tf.placeholder(tf.float32, shape=(None, 1), name='advantage')
        self.values_t = tf.placeholder(tf.float32, shape=(None, 1), name='value')
        self.returns_t = tf.placeholder(tf.float32, shape=(None, 1), name='return')

        self.actor_model, self.critic_model = self._build_model(self.state_t, net_architecture)

        # td_critic = tf.subtract(self.returns_t, self.critic_model, name='TD_error_critic')
        with tf.name_scope('c_loss'):
            self.critic_loss = tf.reduce_mean(tf.square(self.returns_t - self.critic_model))

        # critic_loss = self.critic_discount * K.mean(K.square(rewards - values))

        with tf.name_scope('a_loss'):
            y_pred = self.actor_model
            y_true = self.actions_t
            var = tf.square(self.exploration_noise)
            pi = 3.1415926

            # σ√2π
            denom = tf.sqrt(2 * pi * var)

            # exp(-((x−μ)^2/2σ^2))
            prob_num = tf.exp(- K.square(y_true - y_pred) / (2 * var))
            old_prob_num = tf.exp(- K.square(y_true - self.old_prediction_t) / (2 * var))

            # exp(-((x−μ)^2/2σ^2))/(σ√2π)
            new_prob = prob_num / denom
            old_prob = old_prob_num / denom

            ratio = tf.exp(tf.log(new_prob + 1e-10) - K.log(old_prob + 1e-10))

            p1 = ratio * self.advantage_t
            p2 = tf.clip_by_value(ratio, 1 - self.loss_clipping, 1 + self.loss_clipping) * self.advantage_t
            actor_loss = - tf.reduce_mean(K.minimum(p1, p2))

            # critic_loss__actor = tf.reduce_mean(tf.square(self.returns_t - self.values_t))
            entropy = - tf.reduce_mean(-(new_prob * K.log(new_prob + 1e-10)))

            self.actor_loss = actor_loss + self.critic_discount * self.critic_loss + self.entropy_beta * entropy

            # we use adam optimizer for minimizing the loss
            self.fit_actor = tf.train.AdamOptimizer(self.lr_actor).minimize(actor_loss)
            self.fit_critic = tf.train.AdamOptimizer(self.lr_critic).minimize(self.critic_loss)

    def _build_model(self, state, net_architecture):

        # Neural Net for Actor-Critic Model
        if net_architecture is None:  # Standart architecture
            net_architecture = ppo_net

        with tf.variable_scope('actor'):
            # Building actor
            if self.img_input:
                actor_net = net_building.build_conv_net(net_architecture, self.state_size, actor=True)
            elif self.stack:
                actor_net = net_building.build_stack_net(net_architecture, self.state_size, actor=True)
            else:
                actor_net = net_building.build_nn_net(net_architecture, self.state_size, actor=True)


        # advantage = Input(shape=(1,))
        # old_prediction = Input(shape=(self.n_actions,))
        # rewards = Input(shape=(1,))
        # values = Input(shape=(1,))

            actor_net.add(Dense(self.n_actions, name='output', activation='tanh'))

            actor_model = actor_net(state)
        # actor_model = Model(inputs=[actor_net.inputs, advantage, old_prediction, rewards, values], outputs=[actor_net.outputs])
        # actor_model.compile(optimizer=Adam(lr=self.lr_actor),
        #                     loss=[self.proximal_policy_optimization_loss_continuous(advantage=advantage,
        #                                                                             old_prediction=old_prediction,
        #                                                                             rewards=rewards,
        #                                                                             values=values)])
        # actor_model.summary()

        # Building actor
        with tf.variable_scope('critic'):
            if self.img_input:
                critic_net = net_building.build_conv_net(net_architecture, self.state_size, critic=True)
            elif self.stack:
                critic_net = net_building.build_stack_net(net_architecture, self.state_size, critic=True)
            else:
                critic_net = net_building.build_nn_net(net_architecture, self.state_size, critic=True)

            critic_net.add(Dense(1))
            critic_model = critic_net(state)
        # critic_model.compile(optimizer=Adam(lr=self.lr_critic), loss='mse')

        return actor_model, critic_model

    # def proximal_policy_optimization_loss_continuous(self, advantage, old_prediction, rewards, values):
    #
    #     def loss(y_true, y_pred):
    #         """
    #         f(x) = (1/σ√2π)exp(-(1/2σ^2)(x−μ)^2)
    #         X∼N(μ, σ)
    #         """
    #         var = K.square(self.exploration_noise)
    #         pi = 3.1415926
    #
    #         # σ√2π
    #         denom = K.sqrt(2 * pi * var)
    #
    #         # exp(-((x−μ)^2/2σ^2))
    #         prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
    #         old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))
    #
    #         # exp(-((x−μ)^2/2σ^2))/(σ√2π)
    #         new_prob = prob_num / denom
    #         old_prob = old_prob_num / denom
    #
    #         ratio = K.exp(K.log(new_prob + 1e-10) - K.log(old_prob + 1e-10))
    #
    #         p1 = ratio * advantage
    #         p2 = K.clip(ratio, min_value=1 - self.loss_clipping, max_value=1 + self.loss_clipping) * advantage
    #         actor_loss = - K.mean(K.minimum(p1, p2))
    #         critic_loss = self.critic_discount * K.mean(K.square(rewards - values))
    #         entropy = - self.entropy_beta * K.mean(-(new_prob * K.log(new_prob + 1e-10)))
    #
    #         return actor_loss + critic_loss + entropy
    #
    #     return loss

    def compute_reward_return(self, reward):
        for j in range(len(reward) - 2, -1, -1):
            reward[j] += reward[j + 1] * self.gamma
        return reward

    def get_advantages(self, values, masks, rewards):
        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + self.gamma * self.lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)