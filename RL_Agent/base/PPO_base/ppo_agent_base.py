import os
import datetime
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from RL_Agent.base.utils import net_building
from RL_Agent.base.utils.networks.default_networks import ppo_net
from RL_Agent.base.agent_base import AgentSuper
import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# worker class that inits own environment, trains on it and updloads weights to global net
class PPOSuper(AgentSuper):
    def __init__(self, actor_lr, critic_lr, batch_size, epsilon=0., epsilon_decay=0., epsilon_min=0.,
                 gamma=0.95, n_step_return=10, memory_size=512, loss_clipping=0.2, loss_critic_discount=0.5,
                 loss_entropy_beta=0.001, lmbda=0.95, train_steps=1, exploration_noise=1.0, n_stack=1, img_input=False,
                 state_size=None, net_architecture=None, n_threads=None, tensorboard_dir=None):
        super().__init__(actor_lr=actor_lr, critic_lr=critic_lr, batch_size=batch_size, epsilon=epsilon,
                         epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, gamma=gamma, n_step_return=n_step_return,
                         memory_size=memory_size, loss_clipping=loss_clipping,
                         loss_critic_discount=loss_critic_discount, loss_entropy_beta=loss_entropy_beta, lmbda=lmbda,
                         train_steps=train_steps, exploration_noise=exploration_noise, n_stack=n_stack,
                         img_input=img_input, state_size=state_size, n_threads=n_threads,
                         net_architecture=net_architecture)
        self.tensorboard_dir = tensorboard_dir

    def build_agent(self, state_size, n_actions, stack=False):
        self.sess = tf.keras.backend.get_session()
        super().build_agent(state_size=state_size, n_actions=n_actions, stack=stack)

        # self.lr_actor = lr_actor
        # self.lr_critic = lr_critic
        # self.gamma = 0.99
        # self.batch_size = batch_size
        # self.buffer_size = buffer_size
        # self.loss_clipping = 0.2
        # self.critic_discount = 0.5
        # self.entropy_beta = 0.001
        # self.lmbda = 0.95
        # self.train_epochs = 10
        # self.exploration_noise = 1.0
        self.stddev_loss_calculation = 3.
        self.actor, self.critic = None, None
        # self.actor, self.critic = self._build_model(net_architecture, activation)
        self.memory = []
        self.loss_selected = None  # Select the discrete or continuous version
        self.dummy_action, self.dummy_value = None, None

        if self.tensorboard_dir is not None:
            logdir = os.path.join("ppo", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
            self.logdir = os.path.join(self.tensorboard_dir, logdir)
            self._build_tensorboard_model()
            # actor_logdir = os.path.join(self.tensorboard_dir, 'actor')
            # actor_logdir = os.path.join(actor_logdir, self.logdir)
            # critic_logdir = os.path.join(self.tensorboard_dir, 'critic')
            # critic_logdir = os.path.join(critic_logdir, self.logdir)
            # tensorboard_actor_callback = tf.keras.callbacks.TensorBoard(actor_logdir, histogram_freq=1)
            # tensorboard_critic_callback = tf.keras.callbacks.TensorBoard(critic_logdir, histogram_freq=1)
            # self.actor_callback = [tensorboard_actor_callback]
            # self.critic_callback = [tensorboard_critic_callback]
            self.total_train_steps = 0
            self.total_bc_steps = 0

        # self.epsilon = 0

    def _build_tensorboard_model(self):
        with tf.name_scope("bc_loss"):
            self.tf_bc_loss = tf.placeholder(tf.float32, shape=(), name="bc_actor_loss")
            bc_l_summary = tf.summary.scalar('bc_actor_loss', self.tf_bc_loss)

            self.tf_bc_acc = tf.placeholder(tf.float32, shape=(), name="bc_actor_acc")
            bc_a_summary = tf.summary.scalar('bc_actor_acc', self.tf_bc_acc)

            self.tf_bc_val_loss = tf.placeholder(tf.float32, shape=(), name="bc_actor_val_loss")
            bc_vl_summary = tf.summary.scalar('bc_actor_val_loss', self.tf_bc_val_loss)
            self.tf_bc_val_acc = tf.placeholder(tf.float32, shape=(), name="bc_actor_val_acc")
            bc_va_summary = tf.summary.scalar('bc_actor_val_acc', self.tf_bc_val_acc)

        with tf.name_scope("loss"):
            self.tf_actor_loss = tf.placeholder(tf.float32, shape=(), name="actor_loss")
            a_l_summary = tf.summary.scalar('actor', self.tf_actor_loss)

            self.tf_critic_loss = tf.placeholder(tf.float32, shape=(), name="critic_loss")
            c_l_summary = tf.summary.scalar('critic', self.tf_critic_loss)

            self.total_loss = self.tf_actor_loss + self.critic_discount * self.tf_critic_loss
            t_l_summary = tf.summary.scalar('total', self.total_loss)

        with tf.name_scope("epoch_losses"):
            self.tf_epoch_actor_loss = tf.placeholder(tf.float32, shape=(), name="epoch_actor_loss")
            e_a_l_summary = tf.summary.scalar('actor', self.tf_epoch_actor_loss)

            self.tf_epoch_critic_loss = tf.placeholder(tf.float32, shape=(), name="epoch_critic_loss")
            e_c_l_summary = tf.summary.scalar('critic', self.tf_epoch_critic_loss)

            self.tf_epoch_entropy = tf.placeholder(tf.float32, shape=(), name="epoch_entropy")
            e_e_l_summary = tf.summary.scalar('entropy', self.tf_epoch_entropy)

            self.tf_epoch_total_loss = tf.placeholder(tf.float32, shape=(), name="epoch_total_loss")
            e_t_l_summary = tf.summary.scalar('total', self.tf_epoch_total_loss)

            self.tf_new_prob = tf.placeholder(tf.float32, shape=(), name="epoch_total_loss")
            e_p_l_summary = tf.summary.scalar('prob_values_mean', self.tf_new_prob)

        with tf.name_scope("reward"):
            self.tf_reward = tf.placeholder(tf.float32, shape=(None, 1), name='reward')
            rew_summary = self.variable_summaries(self.tf_reward)

        with tf.name_scope("return"):
            self.tf_return = tf.placeholder(tf.float32, shape=(None, 1), name='return')
            ret_summary = self.variable_summaries(self.tf_return)

        with tf.name_scope("advantage"):
            self.tf_advantage = tf.placeholder(tf.float32, shape=(None, 1), name='advantage')
            ad_summary = self.variable_summaries(self.tf_advantage)

        with tf.name_scope("value"):
            self.tf_values = tf.placeholder(tf.float32, shape=(None, 1), name='value')
            val_summary = self.variable_summaries(self.tf_values)

        with tf.name_scope("train_actions"):
            self.tf_actions = tf.placeholder(tf.float32, shape=(None, self.n_actions), name='actions')
            act_summary = self.variable_summaries(self.tf_actions)

        with tf.name_scope("policy_ratio"):
            self.tf_ratio = tf.placeholder(tf.float32, shape=(None, self.n_actions), name='policy_ratio')
            rat_summary = self.variable_summaries(self.tf_ratio)

        with tf.name_scope("ratio_advantage"):
            self.tf_ratadv = tf.placeholder(tf.float32, shape=(None, self.n_actions), name='ratio_advantage')
            ratadv_summary = self.variable_summaries(self.tf_ratadv)

        with tf.name_scope("clipped_ratio_advantage"):
            self.tf_clip_ratadv = tf.placeholder(tf.float32, shape=(None, self.n_actions), name='clipped_obj')
            clip_summary = self.variable_summaries(self.tf_clip_ratadv)

        with tf.name_scope("stddev_for_exploration"):
            self.tf_epoch_stddev = tf.placeholder(tf.float32, shape=(), name='stddev_for_exploration')
            var_summary = self.variable_summaries(self.tf_epoch_stddev)

        self.bc_summary = tf.summary.merge([bc_l_summary, bc_a_summary, bc_vl_summary, bc_va_summary])
        self.obj_sumary = tf.summary.merge([rat_summary, ratadv_summary, clip_summary, var_summary])
        self.summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.loss_sumary = tf.summary.merge([a_l_summary, c_l_summary, t_l_summary])
        self.rl_sumary = tf.summary.merge([e_a_l_summary, e_c_l_summary, e_e_l_summary, e_t_l_summary, e_p_l_summary,
                                           rew_summary, ret_summary, ad_summary, val_summary, act_summary])
        self.summary_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)

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

    def dummies_sequential(self):
        return np.zeros((1, self.n_actions)), np.zeros((1, 1))

    def dummies_multithread(self, n_threads):
        return np.zeros((n_threads, self.n_actions)), np.zeros((n_threads, 1))

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
        # pred_act = np.array(pred_act)
        # pred_act = np.reshape(pred_act, (pred_act.shape[0], pred_act.shape[2]))
        pred_act = np.array([a[0] for a in pred_act])
        # returns = np.reshape(np.array(returns), (len(returns), 1))
        # returns = np.array(returns)[:, np.newaxis]
        returns = np.array(returns)
        rewards = np.array(rewards)
        values = np.array(values)
        mask = np.array(mask)
        advantages = np.array(advantages)

        # TODO: Decidir la solución a utilizar
        index = range(len(obs))
        # index = np.random.choice(range(len(obs)), self.buffer_size, replace=False)
        self.memory = [obs[index], action[index], pred_act[index], returns[index], rewards[index], values[index],
                       mask[index], advantages[index]]

    def remember_multithread(self, obs, action, pred_act, rewards, values, mask):
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
                obs = np.transpose(obs, axes=(1, 0, 2, 3, 4))
        elif self.stack:
            obs = np.transpose(obs, axes=(1, 0, 2, 3))
        else:
            obs = np.transpose(obs, axes=(1, 0, 2))

        action = np.transpose(action, axes=(1, 0, 2))
        pred_act = np.transpose(pred_act, axes=(1, 0, 2))
        rewards = np.transpose(rewards, axes=(1, 0))
        values = np.transpose(values, axes=(1, 0, 2))
        mask = np.transpose(mask, axes=(1, 0))

        o = obs[0]
        a = action[0]
        p_a = pred_act[0]
        r = rewards[0]
        v = values[0]
        m = mask[0]

        # TODO: Optimizar, es muy lento
        for i in range(1, self.n_threads):
            o = np.concatenate((o, obs[i]), axis=0)
            a = np.concatenate((a, action[i]), axis=0)
            p_a = np.concatenate((p_a, pred_act[i]), axis=0)
            r = np.concatenate((r, rewards[i]), axis=0)
            v = np.concatenate((v, values[i]), axis=0)
            m = np.concatenate((m, mask[i]), axis=0)

        v = np.concatenate((v, [v[-1]]), axis=0)
        returns, advantages = self.get_advantages(v, m, r)
        advantages = np.array(advantages)
        returns = np.array(returns)

        # TODO: Decidir la solución a utilizar
        index = range(len(o))
        # index = np.random.choice(range(len(obs)), self.buffer_size, replace=False)
        self.memory = [o[index], a[index], p_a[index], returns[index], r[index], v[index],
                       m[index], advantages[index]]

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

        actor_loss = self.actor.fit([obs, advantages, old_prediction, returns, values], [action], batch_size=self.batch_size, shuffle=True,
                                    epochs=self.train_epochs, verbose=False)
        critic_loss = self.critic.fit([obs], [returns], batch_size=self.batch_size, shuffle=True, epochs=self.train_epochs,
                                      verbose=False)

        if self.tensorboard_dir is not None:
            # loss_func = self._tensorboard_aux_loss_continuous(advantages, old_prediction, returns, values,  self.exploration_noise*self.epsilon)
            loss_func = self._tensorboard_aux_loss_continuous(advantages, old_prediction, returns, values,
                                                              self.stddev_loss_calculation)
            outs = self.actor.predict([obs, advantages, old_prediction, returns, values])
            epoch_actor_loss, epoch_critic_loss, epoch_entropy, epoch_ratio, epoch_p1, epoch_p2, epoch_var, new_prob = loss_func(
                action, outs)
            epoch_total_loss = - epoch_actor_loss + self.critic_discount * epoch_critic_loss - self.entropy_beta * epoch_entropy

            a_loss = actor_loss.history['loss']
            c_loss = critic_loss.history['loss']

            dict = {self.tf_reward: np.expand_dims(rewards, axis=-1),
                    self.tf_return: returns,
                    self.tf_advantage: advantages,
                    self.tf_values: values,
                    self.tf_actions: action,
                    self.tf_epoch_actor_loss: epoch_actor_loss,
                    self.tf_epoch_critic_loss: epoch_critic_loss,
                    self.tf_epoch_entropy: epoch_entropy,
                    self.tf_epoch_total_loss: epoch_total_loss,
                    self.tf_new_prob: new_prob}

            summary = self.sess.run(self.rl_sumary, dict)
            self.summary_writer.add_summary(summary, int(self.total_train_steps/self.train_epochs))

            dict = {self.tf_ratio: epoch_ratio,
                    self.tf_ratadv: epoch_p1,
                    self.tf_clip_ratadv: epoch_p2,
                    self.tf_epoch_stddev: self.exploration_noise*self.epsilon}

            summary = self.sess.run(self.obj_sumary, dict)
            self.summary_writer.add_summary(summary, int(self.total_train_steps))


            for i in range(len(a_loss)):
                dict = {self.tf_actor_loss: a_loss[i],
                        self.tf_critic_loss: c_loss[i]}
                summary, total_loss = self.sess.run([self.loss_sumary, self.total_loss], dict)
                iter = i+self.total_train_steps
                self.summary_writer.add_summary(summary, iter)

                # self.loss_selected(advantage=advantage,
                #                    old_prediction=old_prediction,
                #                    rewards=rewards,
                #                    values=values

            self.total_train_steps += self.train_epochs

        self._reduce_epsilon()
        return actor_loss, critic_loss

    def _load(self, path):
        # Create a clean graph and import the MetaGraphDef nodes.
        # new_graph = tf.Graph()
        # with tf.keras.backend.get_session() as sess:
        # Import the previously export meta graph.
        # name = path.join(path, name)
        # loaded_model = tf.train.import_meta_graph(name + '.meta')
        # # tf.keras.backend.clear_session()
        # sess = tf.keras.backend.get_session()
        # loaded_model.restore(sess, tf.train.latest_checkpoint(dir + "./"))
        # json_file = open(name+'actor'+'.json', 'r')
        # loaded_model_json = json_file.read()
        # self.actor = model_from_json(loaded_model_json)
        # json_file.close()

        # load weights into new model
        self.actor.load_weights(path+'actor'+".h5")
        # self.actor.compile(optimizer=Adam(lr=self.learning_rate))

        # json_file = open(name+'critic'+'.json', 'r')
        # loaded_model_json = json_file.read()
        # self.critic = model_from_json(loaded_model_json)
        # json_file.close()

        # load weights into new model
        self.critic.load_weights(path+'critic'+".h5")
        # self.critic.compile(optimizer=Adam(lr=self.learning_rate))
        print("Loaded model from disk")

    def _save_network(self, path):
        # sess = tf.keras.backend.get_session()  # op_input_list=(self.actor.get_layers(), self.critic.get_layers())
        # self.saver = tf.train.Saver()
        # path = path + "-" + str(reward)
        # self.saver.save(sess, name)
        # serialize model to JSON
        # model_json = self.actor.to_json()
        # with open(name+'actor'+".json", "w") as json_file:
        #     json_file.write(model_json)
        # serialize weights to HDF5
        self.actor.save_weights(path + 'actor' + ".h5")

        # model_json = self.critic.to_json()
        # with open(name+'critic'+".json", "w") as json_file:
        #     json_file.write(model_json)
        # serialize weights to HDF5
        self.critic.save_weights(path + 'critic' + ".h5")
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
        if self.img_input:
            actor_net = net_building.build_conv_net(net_architecture, self.state_size, actor=True)
        elif self.stack:
            actor_net = net_building.build_stack_net(net_architecture, self.state_size, actor=True)
        else:
            actor_net = net_building.build_nn_net(net_architecture, self.state_size, actor=True)

        advantage = Input(shape=(1,))
        old_prediction = Input(shape=(self.n_actions,))
        rewards = Input(shape=(1,))
        values = Input(shape=(1,))

        if not define_output_layer:
            actor_net.add(Dense(self.n_actions, name='output', activation=last_activation))

        actor_model = Model(inputs=[actor_net.inputs, advantage, old_prediction, rewards, values], outputs=[actor_net.outputs])

        actor_optimizer = tf.keras.optimizers.Adam(self.actor_lr)
        actor_model.compile(optimizer=actor_optimizer,
                            loss=[self.loss_selected(advantage=advantage,
                                                     old_prediction=old_prediction,
                                                     rewards=rewards,
                                                     values=values,
                                                     stddev=self.stddev_loss_calculation)])
        actor_model.summary()

        # Building actor
        if self.img_input:
            critic_model = net_building.build_conv_net(net_architecture, self.state_size, critic=True)
        elif self.stack:
            critic_model = net_building.build_stack_net(net_architecture, self.state_size, critic=True)
        else:
            critic_model = net_building.build_nn_net(net_architecture, self.state_size, critic=True)

        if not define_output_layer:
            critic_model.add(Dense(1))
        critic_optimizer = tf.keras.optimizers.Adam(self.critic_lr)

        critic_model.compile(optimizer=critic_optimizer, loss='mse')

        return actor_model, critic_model

    def proximal_policy_optimization_loss_continuous(self, advantage, old_prediction, rewards, values, stddev):

        def loss(y_true, y_pred):
            """
            f(x) = (1/σ√2π)exp(-(1/2σ^2)(x−μ)^2)
            X∼N(μ, σ)
            """
            # var = K.square(self.exploration_noise*self.epsilon)
            var = K.square(stddev)
            pi = 3.1415926

            # σ√2π
            denom = K.sqrt(2 * pi * var)

            # exp(-((x−μ)^2/2σ^2))
            prob_num = K.exp(- K.square(y_true - y_pred) / (2 * var))
            old_prob_num = K.exp(- K.square(y_true - old_prediction) / (2 * var))

            # exp(-((x−μ)^2/2σ^2))/(σ√2π)
            new_prob = prob_num / denom
            old_prob = old_prob_num / denom

            # ratio = K.exp(K.log(new_prob + 1e-10) - K.log(old_prob + 1e-10))
            ratio = (new_prob) / (old_prob + 1e-20)

            p1 = ratio * advantage
            p2 = K.clip(ratio, min_value=1 - self.loss_clipping, max_value=1 + self.loss_clipping) * advantage
            actor_loss = K.mean(K.minimum(p1, p2))
            critic_loss =  K.mean(K.square(rewards - values))
            entropy = K.mean(-(new_prob * K.log(new_prob + 1e-10)))

            return -actor_loss + self.critic_discount * critic_loss - self.entropy_beta * entropy

        return loss

    def proximal_policy_optimization_loss_discrete(self, advantage, old_prediction, rewards, values, stddev=None):

        def loss(y_true, y_pred):
            new_prob = K.sum(y_true * y_pred, axis=-1)
            old_prob = K.sum(y_true * old_prediction, axis=-1)

            ratio = K.exp(K.log(new_prob + 1e-10) - K.log(old_prob + 1e-10))

            p1 = ratio * advantage
            p2 = K.clip(ratio, min_value=1 - self.loss_clipping, max_value=1 + self.loss_clipping) * advantage
            actor_loss = - K.mean(K.minimum(p1, p2))
            critic_loss = self.critic_discount * K.mean(K.square(rewards - values))
            entropy = - self.entropy_beta * K.mean(-(new_prob * K.log(new_prob + 1e-10)))

            return actor_loss + critic_loss + entropy

        return loss


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
            prob_num = np.exp((-1./2.) * np.square((y_true - y_pred) / (stddev)))
            old_prob_num = np.exp((-1./2.) * np.square((y_true - old_prediction) / (stddev)))

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

    def bc_fit_legacy(self, expert_traj, epochs, batch_size, learning_rate=1e-3, shuffle=False, optimizer=Adam(), loss='mse',
        validation_split=0.15):

        expert_traj_s = np.array([x[0] for x in expert_traj])
        expert_traj_a = np.array([x[1] for x in expert_traj])
        expert_traj_a = self._actions_to_onehot(expert_traj_a)
        dummy_advantage = np.zeros((expert_traj_a.shape[0], 1))
        dummy_old_prediction = np.zeros(expert_traj_a.shape)
        dummy_rewards = np.zeros((expert_traj_a.shape[0], 1))
        dummy_values = np.zeros((expert_traj_a.shape[0], 1))
        optimizer.lr = learning_rate
        self.actor.compile(optimizer=optimizer, loss=loss)
        hist = self.actor.fit([expert_traj_s, dummy_advantage, dummy_old_prediction, dummy_rewards, dummy_values],
                       [expert_traj_a], batch_size=batch_size, shuffle=shuffle, epochs=epochs, verbose=2,
                       validation_split=validation_split)

        if self.tensorboard_dir is not None:
            loss = hist.history['loss']
            acc = hist.history['acc']
            val_loss = hist.history['val_loss']
            val_acc = hist.history['val_acc']

            for i in range(len(loss)):
                dict = {self.tf_bc_loss: loss[i],
                        self.tf_bc_acc: acc[i],
                        self.tf_bc_val_loss: val_loss[i],
                        self.tf_bc_val_acc: val_acc[i]}
                summary = self.sess.run(self.bc_summary, dict)
                iter = i+self.total_bc_steps
                self.summary_writer.add_summary(summary, iter)

            self.total_bc_steps += epochs

    def _actions_to_onehot(self, actions):
        return actions

    def _reduce_epsilon(self):
        if isinstance(self.epsilon_decay, float):
            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay
        else:
            self.epsilon = self.epsilon_decay(self.epsilon, self.epsilon_min)



