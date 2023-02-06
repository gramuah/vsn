import datetime
import os
import time
import base64
import marshal
import dill
import types
import json

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from RL_Agent.base.utils.networks.networks_interface import RLNetModel, TrainingHistory
from RL_Agent.base.utils.networks import networks, losses, returns_calculations, tensor_board_loss_functions, dqn_utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Flatten


class PPONet(RLNetModel):
    def __init__(self, actor_net, critic_net, checkpoint_path=None, checkpoints_to_keep=10,
                 save_every_iterations=100, tensorboard_dir=None):
        super().__init__(tensorboard_dir)

        self.save_every_iterations = save_every_iterations
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.total_epochs = 0
        self.loss_func_actor = None
        self.loss_func_critic = None
        self.optimizer_actor = None
        self.optimizer_critic = None
        self.metrics = None
        self.calculate_advantages = None

        current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")

        if checkpoint_path is not None:
            self.actor_checkpoint = tf.train.Checkpoint(model=self.actor_net)
            self.actor_manager = tf.train.CheckpointManager(self.actor_checkpoint,
                                                            os.path.join(checkpoint_path,
                                                                         current_time + '_checkpoints', 'actor',
                                                                         ),
                                                            checkpoint_name='actor',
                                                            max_to_keep=checkpoints_to_keep)

            self.critic_checkpoint = tf.train.Checkpoint(model=self.critic_net)
            self.critic_manager = tf.train.CheckpointManager(self.critic_checkpoint,
                                                             os.path.join(checkpoint_path,
                                                                          current_time + '_checkpoints', 'critic',
                                                                          ),
                                                             checkpoint_name='critic',
                                                             max_to_keep=checkpoints_to_keep)

    def compile(self, loss, optimizer, metrics=tf.keras.metrics.MeanSquaredError()):
        self.loss_func_actor = loss[0]
        self.loss_func_critic = loss[1]
        self.optimizer_actor = optimizer[0]
        self.optimizer_critic = optimizer[1]
        self.calculate_advantages = returns_calculations.gae
        self.metrics = metrics

    def summary(self):
        pass

    def predict(self, x):
        y_ = self._predict(x)
        return y_.numpy()

    def predict_values(self, x):
        y_ = self._predict_values(x)
        return y_.numpy()

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, x):
        """ Predict the output sentence for a given input sentence
            Args:
                test_source_text: input sentence (raw string)

            Returns:
                The encoder's attention vectors
                The decoder's bottom attention vectors
                The decoder's middle attention vectors
                The input string array (input sentence split by ' ')
                The output string array
            """
        y_ = self.actor_net(tf.cast(x, tf.float32), training=False)
        return y_

    @tf.function(experimental_relax_shapes=True)
    def _predict_values(self, x):
        y_ = self.critic_net(tf.cast(x, tf.float32), training=False)
        return y_

    def train_step(self, x, old_prediction, y, returns, advantages, stddev=None, loss_clipping=0.3,
                   critic_discount=0.5, entropy_beta=0.001):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        return self._train_step(x, old_prediction, y, returns, advantages, stddev, loss_clipping,
                                critic_discount, entropy_beta)

    # @tf.function(experimental_relax_shapes=True)
    def _train_step(self, x, old_prediction, y, returns, advantages, stddev=None, loss_clipping=0.3,
                    critic_discount=0.5, entropy_beta=0.001):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            values = self.critic_net(x, training=True)
            y_ = self.actor_net(x, training=True)
            loss_actor, [act_comp_loss, critic_comp_loss, entropy_comp_loss] = self.loss_func_actor(y, y_, advantages,
                                                                                                    old_prediction,
                                                                                                    returns, values,
                                                                                                    stddev,
                                                                                                    loss_clipping,
                                                                                                    critic_discount,
                                                                                                    entropy_beta)
            loss_critic = self.loss_func_critic(returns, values)

        self.metrics.update_state(y, y_)

        variables_actor = self.actor_net.trainable_variables
        variables_critic = self.critic_net.trainable_variables
        gradients_actor, gradients_critic = tape.gradient([loss_actor, loss_critic],
                                                          [variables_actor, variables_critic])
        self.optimizer_actor.apply_gradients(zip(gradients_actor, variables_actor))
        self.optimizer_critic.apply_gradients(zip(gradients_critic, variables_critic))

        return [loss_actor, loss_critic], \
               [gradients_actor, gradients_critic], \
               [variables_actor, variables_critic], \
               returns, \
               advantages, \
               [act_comp_loss, critic_comp_loss, entropy_comp_loss]

    def fit(self, obs, next_obs, actions, rewards, done, epochs, batch_size, validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        act_probs = kargs[0]
        mask = kargs[1]
        stddev = kargs[2]
        loss_clipping = kargs[3]
        critic_discount = kargs[4]
        entropy_beta = kargs[5]
        gamma = kargs[6]
        lmbda = kargs[7]

        # Calculate returns and advantages
        returns = []
        advantages = []

        # TODO: [CARLOS] check if this split makes sense at all (specially the +1). Maybe using a ceiling instead of
        #   int in order to fit the rest of the observations.

        batch_obs = np.array_split(obs, int(rewards.shape[0] / batch_size) + 1)
        batch_rewards = np.array_split(rewards, int(rewards.shape[0] / batch_size) + 1)
        batch_mask = np.array_split(mask, int(rewards.shape[0] / batch_size) + 1)

        for b_o, b_r, b_m in zip(batch_obs, batch_rewards, batch_mask):
            values = self.predict_values(b_o)
            ret, adv = self.calculate_advantages(values, b_m, b_r, gamma, lmbda)
            returns.extend(ret)
            advantages.extend(adv)

        dataset = tf.data.Dataset.from_tensor_slices((tf.cast(obs, tf.float32),
                                                      tf.cast(act_probs, tf.float32),
                                                      tf.cast(rewards, tf.float32),
                                                      tf.cast(actions, tf.float32),
                                                      tf.cast(mask, tf.float32),
                                                      tf.cast(returns, tf.float32),
                                                      tf.cast(advantages, tf.float32)))

        if shuffle:
            dataset = dataset.shuffle(len(obs), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        if self.train_summary_writer is not None:
            with self.train_summary_writer.as_default():
                self.rl_loss_sumaries([np.array(returns),
                                       np.array(advantages),
                                       actions,
                                       act_probs,
                                       stddev,
                                       np.array(mask)],
                                      ['returns',
                                       'advantages',
                                       'actions',
                                       'act_probabilities'
                                       'stddev',
                                       'dones']
                                      , self.total_epochs)

        history_actor = TrainingHistory()
        history_critic = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            loss = [0., 0.]
            act_comp_loss = 0.
            critic_comp_loss = 0.
            entropy_comp_loss = 0.
            for batch, (batch_obs,
                        batch_act_probs,
                        batch_rewards,
                        batch_actions,
                        batch_mask,
                        batch_returns,
                        batch_advantages) in enumerate(dataset.take(-1)):
                loss, \
                gradients, \
                variables, \
                returns, \
                advantages, \
                [act_comp_loss, critic_comp_loss, entropy_comp_loss] = self.train_step(batch_obs,
                                                                                       batch_act_probs,
                                                                                       batch_actions,
                                                                                       batch_returns,
                                                                                       batch_advantages,
                                                                                       stddev=tf.cast(stddev,
                                                                                                      tf.float32),
                                                                                       loss_clipping=tf.cast(
                                                                                           loss_clipping, tf.float32),
                                                                                       critic_discount=tf.cast(
                                                                                           critic_discount, tf.float32),
                                                                                       entropy_beta=tf.cast(
                                                                                           entropy_beta, tf.float32))

            if verbose:
                print(
                    'Epoch {}\t Loss Actor\Critic {:.4f}\{:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, loss[0].numpy(), loss[1].numpy(), self.metrics.result(),
                        time.time() - start_time))
                start_time = time.time()

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    self.loss_sumaries([loss[0],
                                        loss[1],
                                        act_comp_loss,
                                        critic_comp_loss,
                                        entropy_comp_loss,
                                        critic_discount * critic_comp_loss,
                                        entropy_beta * entropy_comp_loss],
                                       ['actor_model_loss (-a_l + c*c_l - b*e_l)',
                                        'critic_model_loss',
                                        'actor_component (a_l)',
                                        'critic_component (c_l)',
                                        'entropy_component (e_l)',
                                        '(c*c_l)',
                                        '(b*e_l)'],
                                       self.total_epochs)

            self.total_epochs += 1

            history_actor.history['loss'].append(loss[0].numpy())
            history_critic.history['loss'].append(loss[1].numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)

        return history_actor, history_critic

    @tf.function(experimental_relax_shapes=True)
    def _bc_train_step(self, x, y):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            y_ = self.actor_net(x, training=True)
            loss = self.bc_loss_func(y, y_)

        self.bc_metrics.update_state(y, y_)

        variables = self.actor_net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.bc_optimizer.apply_gradients(zip(gradients, variables))

        return loss, gradients, variables

    @tf.function(experimental_relax_shapes=True)
    def _bc_evaluate(self, x, y):
        y_ = self.actor_net(x, training=False)

        loss = self.bc_loss_func(y, y_)

        self.bc_metrics.update_state(y, y_)

        return loss

    def save(self, path):
        # Serializar función calculate_advanteges
        calculate_advantages_globals = dill.dumps(self.calculate_advantages.__globals__)
        calculate_advantages_globals = base64.b64encode(calculate_advantages_globals).decode('ascii')
        calculate_advantages_code = marshal.dumps(self.calculate_advantages.__code__)
        calculate_advantages_code = base64.b64encode(calculate_advantages_code).decode('ascii')

        # Serializar función loss_sumaries
        loss_sumaries_globals = dill.dumps(self.loss_sumaries.__globals__)
        loss_sumaries_globals = base64.b64encode(loss_sumaries_globals).decode('ascii')
        loss_sumaries_code = marshal.dumps(self.loss_sumaries.__code__)
        loss_sumaries_code = base64.b64encode(loss_sumaries_code).decode('ascii')

        # Serializar función rl_loss_sumaries
        rl_loss_sumaries_globals = dill.dumps(self.rl_loss_sumaries.__globals__)
        rl_loss_sumaries_globals = base64.b64encode(rl_loss_sumaries_globals).decode('ascii')
        rl_loss_sumaries_code = marshal.dumps(self.rl_loss_sumaries.__code__)
        rl_loss_sumaries_code = base64.b64encode(rl_loss_sumaries_code).decode('ascii')

        # Serializar función rl_sumaries
        rl_sumaries_globals = dill.dumps(self.rl_sumaries.__globals__)
        rl_sumaries_globals = base64.b64encode(rl_sumaries_globals).decode('ascii')
        rl_sumaries_code = marshal.dumps(self.rl_sumaries.__code__)
        rl_sumaries_code = base64.b64encode(rl_sumaries_code).decode('ascii')

        # saves actor and critic networks
        self.save_checkpoint(path)

        # TODO: Qeda guardar las funciones de pérdida. De momento confio en las que hay definidad como estandar.
        data = {
            "train_log_dir": self.train_log_dir,
            "total_epochs": self.total_epochs,
            "calculate_advantages_globals": calculate_advantages_globals,
            "calculate_advantages_code": calculate_advantages_code,
            "loss_sumaries_globals": loss_sumaries_globals,
            "loss_sumaries_code": loss_sumaries_code,
            "rl_loss_sumaries_globals": rl_loss_sumaries_globals,
            "rl_loss_sumaries_code": rl_loss_sumaries_code,
            "rl_sumaries_globals": rl_sumaries_globals,
            "rl_sumaries_code": rl_sumaries_code,
        }

        with open(os.path.join(path, 'model_data.json'), 'w') as f:
            json.dump(data, f)

    def export_to_protobuf(self, path):
        # Serializar función calculate_advanteges
        calculate_advantages_globals = dill.dumps(self.calculate_advantages.__globals__)
        calculate_advantages_globals = base64.b64encode(calculate_advantages_globals).decode('ascii')
        calculate_advantages_code = marshal.dumps(self.calculate_advantages.__code__)
        calculate_advantages_code = base64.b64encode(calculate_advantages_code).decode('ascii')

        # Serializar función loss_sumaries
        loss_sumaries_globals = dill.dumps(self.loss_sumaries.__globals__)
        loss_sumaries_globals = base64.b64encode(loss_sumaries_globals).decode('ascii')
        loss_sumaries_code = marshal.dumps(self.loss_sumaries.__code__)
        loss_sumaries_code = base64.b64encode(loss_sumaries_code).decode('ascii')

        # Serializar función rl_loss_sumaries
        rl_loss_sumaries_globals = dill.dumps(self.rl_loss_sumaries.__globals__)
        rl_loss_sumaries_globals = base64.b64encode(rl_loss_sumaries_globals).decode('ascii')
        rl_loss_sumaries_code = marshal.dumps(self.rl_loss_sumaries.__code__)
        rl_loss_sumaries_code = base64.b64encode(rl_loss_sumaries_code).decode('ascii')

        # Serializar función rl_sumaries
        rl_sumaries_globals = dill.dumps(self.rl_sumaries.__globals__)
        rl_sumaries_globals = base64.b64encode(rl_sumaries_globals).decode('ascii')
        rl_sumaries_code = marshal.dumps(self.rl_sumaries.__code__)
        rl_sumaries_code = base64.b64encode(rl_sumaries_code).decode('ascii')

        # saves actor and critic networks
        self._save_network(path)

        # tf.saved_model.save(self.loss_func_actor, os.path.join(path, 'loss_func_actor'))
        # tf.saved_model.save(self.loss_func_critic, os.path.join(path, 'loss_func_critic'))
        tf.saved_model.save(self.optimizer_actor, os.path.join(path, 'optimizer_actor'))
        tf.saved_model.save(self.optimizer_critic, os.path.join(path, 'optimizer_critic'))
        tf.saved_model.save(self.metrics, os.path.join(path, 'metrics'))

        # TODO [SERGIO] Queda guardar las funciones de pérdida. De momento confio en las que hay definidad como estandar
        data = {
            "train_log_dir": self.train_log_dir,
            "total_epochs": self.total_epochs,
            "calculate_advantages_globals": calculate_advantages_globals,
            "calculate_advantages_code": calculate_advantages_code,
            "loss_sumaries_globals": loss_sumaries_globals,
            "loss_sumaries_code": loss_sumaries_code,
            "rl_loss_sumaries_globals": rl_loss_sumaries_globals,
            "rl_loss_sumaries_code": rl_loss_sumaries_code,
            "rl_sumaries_globals": rl_sumaries_globals,
            "rl_sumaries_code": rl_sumaries_code,
        }

        with open(os.path.join(path, 'model_data.json'), 'w') as f:
            json.dump(data, f)

    def restore(self, path):
        with open(os.path.join(path, 'model_data.json'), 'r') as f:
            data = json.load(f)

        calculate_advantages_code = base64.b64decode(data['calculate_advantages_code'])
        calculate_advantages_globals = base64.b64decode(data['calculate_advantages_globals'])

        loss_sumaries_code = base64.b64decode(data['loss_sumaries_code'])
        loss_sumaries_globals = base64.b64decode(data['loss_sumaries_globals'])

        rl_loss_sumaries_code = base64.b64decode(data['rl_loss_sumaries_code'])
        rl_loss_sumaries_globals = base64.b64decode(data['rl_loss_sumaries_globals'])

        rl_sumaries_code = base64.b64decode(data['rl_sumaries_code'])
        rl_sumaries_globals = base64.b64decode(data['rl_sumaries_globals'])

        calculate_advantages_globals = dill.loads(calculate_advantages_globals)
        calculate_advantages_globals = self.process_globals(calculate_advantages_globals)
        calculate_advantages_code = marshal.loads(calculate_advantages_code)
        self.calculate_advantages = types.FunctionType(calculate_advantages_code, calculate_advantages_globals,
                                                       "calculate_advantages_func")

        loss_sumaries_globals = dill.loads(loss_sumaries_globals)
        loss_sumaries_globals = self.process_globals(loss_sumaries_globals)
        loss_sumaries_code = marshal.loads(loss_sumaries_code)
        self.loss_sumaries = types.FunctionType(loss_sumaries_code, loss_sumaries_globals, "loss_sumaries_func")

        rl_loss_sumaries_globals = dill.loads(rl_loss_sumaries_globals)
        rl_loss_sumaries_globals = self.process_globals(rl_loss_sumaries_globals)
        rl_loss_sumaries_code = marshal.loads(rl_loss_sumaries_code)
        self.rl_loss_sumaries = types.FunctionType(rl_loss_sumaries_code, rl_loss_sumaries_globals,
                                                   "rl_loss_sumaries_func")

        rl_sumaries_globals = dill.loads(rl_sumaries_globals)
        rl_sumaries_globals = self.process_globals(rl_sumaries_globals)
        rl_sumaries_code = marshal.loads(rl_sumaries_code)
        self.rl_sumaries = types.FunctionType(rl_sumaries_code, rl_sumaries_globals, "rl_sumaries_func")

        self.total_epochs = data['total_epochs']
        self.train_log_dir = data['train_log_dir']

        if self.train_log_dir is not None:
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        else:
            self.train_summary_writer = None
        # self.optimizer_actor = tf.saved_model.load(os.path.join(path, 'optimizer_actor'))
        # self.optimizer_critic = tf.saved_model.load(os.path.join(path, 'optimizer_critic'))
        # self.metricst = tf.saved_model.load(os.path.join(path, 'metrics'))

        # TODO: falta cargar loss_func_actor y loss_func_critic
        # tf.saved_model.save(self.loss_func_actor, os.path.join(path, 'loss_func_actor'))
        # tf.saved_model.save(self.loss_func_critic, os.path.join(path, 'loss_func_critic'))

        self.load_checkpoint(path)

    def restore_from_protobuf(self, path):
        with open(os.path.join(path, 'model_data.json'), 'r') as f:
            data = json.load(f)

        self.actor_net = tf.saved_model.load(os.path.join(path, 'actor'))

    def save_checkpoint(self, path=None):
        if path is None:
            # Save a checkpoint
            self.actor_manager.save()
            self.critic_manager.save()
        else:
            actor_chkpoint = tf.train.Checkpoint(model=self.actor_net,
                                                 optimizer=self.optimizer_actor,
                                                 # loss_func_actor=self.loss_func_actor,
                                                 metrics=self.metrics,
                                                 )
            actor_manager = tf.train.CheckpointManager(actor_chkpoint,
                                                       os.path.join(path, 'actor'),
                                                       checkpoint_name='actor',
                                                       max_to_keep=1)
            actor_manager.save()

            critic_chkpoint = tf.train.Checkpoint(model=self.critic_net,
                                                  optimizer=self.optimizer_critic,
                                                  # loss_func_critic=self.loss_func_critic
                                                  )
            critic_manager = tf.train.CheckpointManager(critic_chkpoint,
                                                        os.path.join(path, 'critic'),
                                                        checkpoint_name='critic',
                                                        max_to_keep=1)
            critic_manager.save()

    def load_checkpoint(self, path=None, checkpoint_to_restore='latest'):
        if path is None:
            if checkpoint_to_restore == 'latest':
                self.actor_checkpoint.restore(self.actor_manager.latest_checkpoint)
                self.critic_checkpoint.restore(self.critic_manager.latest_checkpoint)
            else:
                raise NotImplementedError
        else:
            actor_checkpoint = tf.train.Checkpoint(model=self.actor_net,
                                                   optimizer=self.optimizer_actor,
                                                   metrics=self.metrics)
            actor_manager = tf.train.CheckpointManager(actor_checkpoint,
                                                       os.path.join(path, 'actor'),
                                                       checkpoint_name='actor',
                                                       max_to_keep=1)

            critic_checkpoint = tf.train.Checkpoint(model=self.critic_net,
                                                    optimizer=self.optimizer_critic,
                                                    )
            critic_manager = tf.train.CheckpointManager(critic_checkpoint,
                                                        os.path.join(path, 'critic'),
                                                        checkpoint_name='critic',
                                                        max_to_keep=1)
            if checkpoint_to_restore == 'latest':
                actor_checkpoint.restore(actor_manager.latest_checkpoint)
                critic_checkpoint.restore(critic_manager.latest_checkpoint)
            elif type(checkpoint_to_restore) is int:
                actor_checkpoint.restore(actor_manager.checkpoints[checkpoint_to_restore])
                critic_checkpoint.restore(critic_manager.checkpoints[checkpoint_to_restore])
            else:
                raise TypeError('Checkpoints_to_restore variable has to be either str or int')

    def _save_network(self, path):
        """
        Saves the neural networks of the agent.
        :param path: (str) path to folder to store the network
        :param checkpoint: (bool) If True the network is stored as Tensorflow checkpoint, otherwise the network is
                                    stored in protobuffer format.
        """
        # if checkpoint:
        #     # Save a checkpoint
        #     actor_chkpoint = tf.train.Checkpoint(model=self.actor_net)
        #     actor_manager = tf.train.CheckpointManager(actor_chkpoint,
        #                                                os.path.join(path, 'actor', 'checkpoint'),
        #                                                checkpoint_name='actor',
        #                                                max_to_keep=3)
        #     save_path = actor_manager.save()
        #
        #     critic_chkpoint = tf.train.Checkpoint(model=self.critic_net)
        #     critic_manager = tf.train.CheckpointManager(critic_chkpoint,
        #                                                os.path.join(path, 'critic', 'checkpoint'),
        #                                                checkpoint_name='critic',
        #                                                max_to_keep=3)
        #     critic_manager.save()
        # else:
        # Save as protobuffer
        tf.saved_model.save(self.actor_net, os.path.join(path, 'actor'))
        tf.saved_model.save(self.critic_net, os.path.join(path, 'critic'))

        print("Saved model to disk")
        print(datetime.datetime.now())


class DQNNet(RLNetModel):
    def __init__(self, net, chckpoint_path=None, chckpoints_to_keep=10, tensorboard_dir=None):
        super().__init__(tensorboard_dir)

        self.net = net
        self.target_net = tf.keras.models.clone_model(net)

        # self._tensorboard_util(tensorboard_dir)
        # if tensorboard_dir is not None:
        #     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #     train_log_dir = os.path.join(tensorboard_dir, 'gradient_tape/' + current_time + '/train')
        #     self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # else:
        #     self.train_summary_writer = None

        self.total_epochs = 0
        self.loss_func = None
        self.optimizer = None
        self.metrics = None
        # self.loss_sumaries = tensor_board_loss_functions.loss_sumaries
        # self.rl_loss_sumaries = tensor_board_loss_functions.rl_loss_sumaries
        # self.rl_sumaries = tensor_board_loss_functions.rl_sumaries

        if chckpoint_path is not None:
            self.net_chkpoint = tf.train.Checkpoint(model=self.net)
            self.net_manager = tf.train.CheckpointManager(self.net_chkpoint,
                                                          os.path.join(chckpoint_path, 'checkpoint'),
                                                          checkpoint_name='actor',
                                                          max_to_keep=chckpoints_to_keep)

    def compile(self, loss, optimizer, metrics=tf.keras.metrics.Accuracy()):
        self.loss_func = loss[0]
        self.optimizer = optimizer[0]
        self.metrics = metrics

    def summary(self):
        pass

    def predict(self, x):
        y_ = self._predict(x)
        return y_.numpy()

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, x):
        """ Predict the output sentence for a given input sentence
            Args:
                test_source_text: input sentence (raw string)

            Returns:
                The encoder's attention vectors
                The decoder's bottom attention vectors
                The decoder's middle attention vectors
                The input string array (input sentence split by ' ')
                The output string array
            """
        y_ = self.net(tf.cast(x, tf.float32), training=False)
        return y_

    @tf.function(experimental_relax_shapes=True)
    def train_step(self, obs, target):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            y_ = self.net(obs, training=True)
            loss, loss_components = self.loss_func(target, y_)
        self.metrics.update_state(target, y_)

        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, gradients, variables, loss_components

    def fit(self, obs, next_obs, actions, rewards, done, epochs, batch_size,
            validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        gamma = kargs[0]

        # Calculate target Q values for optimization
        pred = self.net.predict(obs)
        target_pred = self.target_net.predict(next_obs)
        target = dqn_utils.dqn_calc_target(done, rewards, next_obs, gamma, target_pred)

        for i in range(target.shape[0]):
            pred[i][actions[i]] = target[i]

        dataset = tf.data.Dataset.from_tensor_slices((obs,
                                                      pred))

        if shuffle:
            dataset = dataset.shuffle(len(obs), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            loss = 0.
            for batch, (bach_obs,
                        bach_target) in enumerate(dataset.take(-1)):
                loss, gradients, variables, loss_components = self.train_step(bach_obs, bach_target)

                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print('Epoch {}\t Batch {}\t Loss {:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), self.metrics.result(), time.time() - start_time))
                    start_time = time.time()

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    self.loss_sumaries([loss], ['loss'], self.total_epochs)
                    self.rl_loss_sumaries([np.float32(np.expand_dims(actions, axis=-1))], ['actions'],
                                          self.total_epochs)

            self.total_epochs += 1

            history.history['loss'].append(loss.numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)

        return history

    @tf.function(experimental_relax_shapes=True)
    def _bc_train_step(self, x, y):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            y_ = self.net(x, training=True)
            loss = self.bc_loss_func(y, y_)

        self.bc_metrics.update_state(y, y_)

        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.bc_optimizer.apply_gradients(zip(gradients, variables))

        return loss, gradients, variables

    @tf.function(experimental_relax_shapes=True)
    def _bc_evaluate(self, x, y):
        y_ = self.net(x, training=False)

        loss = self.bc_loss_func(y, y_)

        self.bc_metrics.update_state(y, y_)

        return loss

    def save(self, path):
        # Serializar función calculate_advanteges
        # calculate_advantages_globals = dill.dumps(self.calculate_advantages.__globals__)
        # calculate_advantages_globals = base64.b64encode(calculate_advantages_globals).decode('ascii')
        # calculate_advantages_code = marshal.dumps(self.calculate_advantages.__code__)
        # calculate_advantages_code = base64.b64encode(calculate_advantages_code).decode('ascii')

        # Serializar función loss_sumaries
        loss_sumaries_globals = dill.dumps(self.loss_sumaries.__globals__)
        loss_sumaries_globals = base64.b64encode(loss_sumaries_globals).decode('ascii')
        loss_sumaries_code = marshal.dumps(self.loss_sumaries.__code__)
        loss_sumaries_code = base64.b64encode(loss_sumaries_code).decode('ascii')

        # Serializar función rl_loss_sumaries
        rl_loss_sumaries_globals = dill.dumps(self.rl_loss_sumaries.__globals__)
        rl_loss_sumaries_globals = base64.b64encode(rl_loss_sumaries_globals).decode('ascii')
        rl_loss_sumaries_code = marshal.dumps(self.rl_loss_sumaries.__code__)
        rl_loss_sumaries_code = base64.b64encode(rl_loss_sumaries_code).decode('ascii')

        # Serializar función rl_sumaries
        rl_sumaries_globals = dill.dumps(self.rl_sumaries.__globals__)
        rl_sumaries_globals = base64.b64encode(rl_sumaries_globals).decode('ascii')
        rl_sumaries_code = marshal.dumps(self.rl_sumaries.__code__)
        rl_sumaries_code = base64.b64encode(rl_sumaries_code).decode('ascii')

        # saves actor and critic networks
        self.save_checkpoint(path)

        # TODO: Queda guardar las funciones de pérdida. De momento confio en las que hay definidas como estandar.
        data = {
            "train_log_dir": self.train_log_dir,
            "total_epochs": self.total_epochs,
            # "calculate_advantages_globals": calculate_advantages_globals,
            # "calculate_advantages_code": calculate_advantages_code,
            "loss_sumaries_globals": loss_sumaries_globals,
            "loss_sumaries_code": loss_sumaries_code,
            "rl_loss_sumaries_globals": rl_loss_sumaries_globals,
            "rl_loss_sumaries_code": rl_loss_sumaries_code,
            "rl_sumaries_globals": rl_sumaries_globals,
            "rl_sumaries_code": rl_sumaries_code,
        }

        with open(os.path.join(path, 'model_data.json'), 'w') as f:
            json.dump(data, f)

    def save_checkpoint(self, path=None):
        if path is None:
            # Save a checkpoint
            self.net_manager.save()
        else:
            net_chkpoint = tf.train.Checkpoint(model=self.net,
                                               optimizer=self.optimizer,
                                               # loss_func=self.loss_func,
                                               metrics=self.metrics,
                                               )
            net_manager = tf.train.CheckpointManager(net_chkpoint,
                                                     os.path.join(path, 'net'),
                                                     checkpoint_name='net',
                                                     max_to_keep=1)
            net_manager.save()

    def restore(self, path):
        with open(os.path.join(path, 'model_data.json'), 'r') as f:
            data = json.load(f)

        loss_sumaries_code = base64.b64decode(data['loss_sumaries_code'])
        loss_sumaries_globals = base64.b64decode(data['loss_sumaries_globals'])

        rl_loss_sumaries_code = base64.b64decode(data['rl_loss_sumaries_code'])
        rl_loss_sumaries_globals = base64.b64decode(data['rl_loss_sumaries_globals'])

        rl_sumaries_code = base64.b64decode(data['rl_sumaries_code'])
        rl_sumaries_globals = base64.b64decode(data['rl_sumaries_globals'])

        loss_sumaries_globals = dill.loads(loss_sumaries_globals)
        loss_sumaries_globals = self.process_globals(loss_sumaries_globals)
        loss_sumaries_code = marshal.loads(loss_sumaries_code)
        self.loss_sumaries = types.FunctionType(loss_sumaries_code, loss_sumaries_globals, "loss_sumaries_func")

        rl_loss_sumaries_globals = dill.loads(rl_loss_sumaries_globals)
        rl_loss_sumaries_globals = self.process_globals(rl_loss_sumaries_globals)
        rl_loss_sumaries_code = marshal.loads(rl_loss_sumaries_code)
        self.rl_loss_sumaries = types.FunctionType(rl_loss_sumaries_code, rl_loss_sumaries_globals,
                                                   "rl_loss_sumaries_func")

        rl_sumaries_globals = dill.loads(rl_sumaries_globals)
        rl_sumaries_globals = self.process_globals(rl_sumaries_globals)
        rl_sumaries_code = marshal.loads(rl_sumaries_code)
        self.rl_sumaries = types.FunctionType(rl_sumaries_code, rl_sumaries_globals, "rl_sumaries_func")

        self.total_epochs = data['total_epochs']
        self.train_log_dir = data['train_log_dir']

        if self.train_log_dir is not None:
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        else:
            self.train_summary_writer = None

        # self.optimizer_actor = tf.saved_model.load(os.path.join(path, 'optimizer_actor'))
        # self.optimizer_critic = tf.saved_model.load(os.path.join(path, 'optimizer_critic'))
        # self.metricst = tf.saved_model.load(os.path.join(path, 'metrics'))

        # TODO: falta cargar loss_func_actor y loss_func_critic
        # tf.saved_model.save(self.loss_func_actor, os.path.join(path, 'loss_func_actor'))
        # tf.saved_model.save(self.loss_func_critic, os.path.join(path, 'loss_func_critic'))

        self.load_checkpoint(path)

    def load_checkpoint(self, path=None, chckpoint_to_restore='latest'):
        if path is None:
            if chckpoint_to_restore == 'latest':
                self.net_chkpoint.restore(self.net_manager.latest_checkpoint)
            else:
                chck = self.net_manager.checkpoints
        else:
            net_chkpoint = tf.train.Checkpoint(model=self.net,
                                               optimizer=self.optimizer,
                                               # loss_func_actor=self.loss_func_actor,
                                               metrics=self.metrics)
            net_manager = tf.train.CheckpointManager(net_chkpoint,
                                                     os.path.join(path, 'net'),
                                                     checkpoint_name='net',
                                                     max_to_keep=1)
            net_chkpoint.restore(net_manager.latest_checkpoint)

    def get_weights(self):
        weights = []
        for layer in self.net.layers:
            weights.append(layer.get_weights())
        return weights

    def set_weights(self, weights):
        for layer, w in zip(self.target_net.layers, weights):
            layer.set_weights(w)

    def copy_model_to_target(self):
        for net_layer, target_layer in zip(self.net.layers, self.target_net.layers):
            target_layer.set_weights(net_layer.get_weights())


class DDQNNet(DQNNet):
    def __init__(self, net, tensorboard_dir=None):
        super().__init__(net, tensorboard_dir)

    def fit(self, obs, next_obs, actions, rewards, done, epochs, batch_size,
            validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        gamma = kargs[0]

        # Calculate target Q values for optimization
        pred = self.net.predict(obs)
        target_pred = self.target_net.predict(next_obs)
        target = dqn_utils.ddqn_calc_target(done, rewards, next_obs, gamma, pred, target_pred)

        for i in range(target.shape[0]):
            pred[i][actions[i]] = target[i]

        dataset = tf.data.Dataset.from_tensor_slices((obs,
                                                      pred))

        if shuffle:
            dataset = dataset.shuffle(len(obs), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            loss = 0.
            for batch, (bach_obs,
                        bach_target) in enumerate(dataset.take(-1)):
                loss, gradients, variables, loss_components = self.train_step(bach_obs, bach_target)

                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print('Epoch {}\t Batch {}\t Loss {:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), self.metrics.result(), time.time() - start_time))
                    start_time = time.time()

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    self.loss_sumaries([loss], ['loss'], self.total_epochs)
                    self.rl_loss_sumaries([np.float32(np.expand_dims(actions, axis=-1))], ['actions'],
                                          self.total_epochs)
            self.total_epochs += 1

            history.history['loss'].append(loss.numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)

        return history


class DDDQNNet(DDQNNet):
    def __init__(self, net, tensorboard_dir=None):
        super().__init__(net, tensorboard_dir)


class DPGNet(RLNetModel):
    def __init__(self, net, chckpoint_path=None, chckpoints_to_keep=10, tensorboard_dir=None):
        super().__init__(tensorboard_dir)

        self.net = net

        # self._tensorboard_util(tensorboard_dir)

        # if tensorboard_dir is not None:
        #     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #     train_log_dir = os.path.join(tensorboard_dir, 'gradient_tape/' + current_time + '/train')
        #     self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # else:
        #     self.train_summary_writer = None

        self.total_epochs = 0
        self.loss_func = None
        self.optimizer = None
        self.metrics = None
        # self.loss_sumaries = tensor_board_loss_functions.loss_sumaries
        # self.rl_loss_sumaries = tensor_board_loss_functions.rl_loss_sumaries
        # self.rl_sumaries = tensor_board_loss_functions.rl_sumaries
        if chckpoint_path is not None:
            self.net_chkpoint = tf.train.Checkpoint(model=self.net)
            self.net_manager = tf.train.CheckpointManager(self.net_chkpoint,
                                                          os.path.join(chckpoint_path, 'checkpoint'),
                                                          checkpoint_name='actor',
                                                          max_to_keep=chckpoints_to_keep)

    def compile(self, loss, optimizer, metrics=tf.keras.metrics.MeanSquaredError()):
        self.loss_func = loss[0]
        self.optimizer = optimizer[0]
        self.metrics = metrics
        self.calculate_returns = returns_calculations.discount_and_norm_rewards

    def summary(self):
        pass

    def predict(self, x):
        y_ = self._predict(x)
        return y_.numpy()

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, x):
        """ Predict the output sentence for a given input sentence
            Args:
                test_source_text: input sentence (raw string)

            Returns:
                The encoder's attention vectors
                The decoder's bottom attention vectors
                The decoder's middle attention vectors
                The input string array (input sentence split by ' ')
                The output string array
            """
        y_ = self.net(tf.cast(x, tf.float32), training=False)
        return y_

    # @tf.function(experimental_relax_shapes=True)
    def train_step(self, obs, actions, returns):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            y_ = self.net(obs, training=True)
            loss, loss_components = self.loss_func(y_, actions, returns)
        self.metrics.update_state(actions, y_)

        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss, gradients, variables, loss_components

    def fit(self, obs, next_obs, actions, rewards, done, epochs, batch_size, validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        gamma = kargs[0]
        returns = self.calculate_returns(rewards, np.logical_not(done), gamma)
        dataset = tf.data.Dataset.from_tensor_slices((np.float32(obs),
                                                      np.float32(actions),
                                                      np.float32(returns)))

        if shuffle:
            dataset = dataset.shuffle(len(obs), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            loss = 0.
            for batch, (bach_obs,
                        bach_actions,
                        bach_returns) in enumerate(dataset.take(-1)):
                loss, gradients, variables, loss_components = self.train_step(bach_obs, bach_actions, bach_returns)

                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print('Epoch {}\t Batch {}\t Loss {:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), self.metrics.result(), time.time() - start_time))
                    start_time = time.time()

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    self.loss_sumaries([loss], ['loss'], self.total_epochs)
                    self.rl_loss_sumaries([np.float32(np.expand_dims(actions, axis=-1)),
                                           np.float32(np.expand_dims(returns, axis=-1))],
                                          ['actions',
                                           'returns'],
                                          self.total_epochs)
            self.total_epochs += 1

            history.history['loss'].append(loss.numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)
        return history

    @tf.function(experimental_relax_shapes=True)
    def _bc_train_step(self, x, y):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            y_ = self.net(x, training=True)
            loss = self.bc_loss_func(y, y_)

        self.bc_metrics.update_state(y, y_)

        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.bc_optimizer.apply_gradients(zip(gradients, variables))

        return loss, gradients, variables

    @tf.function(experimental_relax_shapes=True)
    def _bc_evaluate(self, x, y):
        y_ = self.net(x, training=False)

        loss = self.bc_loss_func(y, y_)

        self.bc_metrics.update_state(y, y_)

        return loss

    def save(self, path):
        # Serializar función calculate_advanteges
        # calculate_advantages_globals = dill.dumps(self.calculate_advantages.__globals__)
        # calculate_advantages_globals = base64.b64encode(calculate_advantages_globals).decode('ascii')
        # calculate_advantages_code = marshal.dumps(self.calculate_advantages.__code__)
        # calculate_advantages_code = base64.b64encode(calculate_advantages_code).decode('ascii')

        # Serializar función loss_sumaries
        loss_sumaries_globals = dill.dumps(self.loss_sumaries.__globals__)
        loss_sumaries_globals = base64.b64encode(loss_sumaries_globals).decode('ascii')
        loss_sumaries_code = marshal.dumps(self.loss_sumaries.__code__)
        loss_sumaries_code = base64.b64encode(loss_sumaries_code).decode('ascii')

        # Serializar función rl_loss_sumaries
        rl_loss_sumaries_globals = dill.dumps(self.rl_loss_sumaries.__globals__)
        rl_loss_sumaries_globals = base64.b64encode(rl_loss_sumaries_globals).decode('ascii')
        rl_loss_sumaries_code = marshal.dumps(self.rl_loss_sumaries.__code__)
        rl_loss_sumaries_code = base64.b64encode(rl_loss_sumaries_code).decode('ascii')

        # Serializar función rl_sumaries
        rl_sumaries_globals = dill.dumps(self.rl_sumaries.__globals__)
        rl_sumaries_globals = base64.b64encode(rl_sumaries_globals).decode('ascii')
        rl_sumaries_code = marshal.dumps(self.rl_sumaries.__code__)
        rl_sumaries_code = base64.b64encode(rl_sumaries_code).decode('ascii')

        # saves actor and critic networks
        self.save_checkpoint(path)

        # TODO: Qeda guardar las funciones de pérdida. De momento confio en las que hay definidad como estandar.
        data = {
            "train_log_dir": self.train_log_dir,
            "total_epochs": self.total_epochs,
            # "calculate_advantages_globals": calculate_advantages_globals,
            # "calculate_advantages_code": calculate_advantages_code,
            "loss_sumaries_globals": loss_sumaries_globals,
            "loss_sumaries_code": loss_sumaries_code,
            "rl_loss_sumaries_globals": rl_loss_sumaries_globals,
            "rl_loss_sumaries_code": rl_loss_sumaries_code,
            "rl_sumaries_globals": rl_sumaries_globals,
            "rl_sumaries_code": rl_sumaries_code,
        }

        with open(os.path.join(path, 'model_data.json'), 'w') as f:
            json.dump(data, f)

    def save_checkpoint(self, path=None):
        if path is None:
            # Save a checkpoint
            self.net_manager.save()
        else:
            net_chkpoint = tf.train.Checkpoint(model=self.net,
                                               optimizer=self.optimizer,
                                               # loss_func=self.loss_func,
                                               metrics=self.metrics,
                                               )
            net_manager = tf.train.CheckpointManager(net_chkpoint,
                                                     os.path.join(path, 'net'),
                                                     checkpoint_name='net',
                                                     max_to_keep=1)
            net_manager.save()

    def restore(self, path):
        with open(os.path.join(path, 'model_data.json'), 'r') as f:
            data = json.load(f)

        loss_sumaries_code = base64.b64decode(data['loss_sumaries_code'])
        loss_sumaries_globals = base64.b64decode(data['loss_sumaries_globals'])

        rl_loss_sumaries_code = base64.b64decode(data['rl_loss_sumaries_code'])
        rl_loss_sumaries_globals = base64.b64decode(data['rl_loss_sumaries_globals'])

        rl_sumaries_code = base64.b64decode(data['rl_sumaries_code'])
        rl_sumaries_globals = base64.b64decode(data['rl_sumaries_globals'])

        loss_sumaries_globals = dill.loads(loss_sumaries_globals)
        loss_sumaries_globals = self.process_globals(loss_sumaries_globals)
        loss_sumaries_code = marshal.loads(loss_sumaries_code)
        self.loss_sumaries = types.FunctionType(loss_sumaries_code, loss_sumaries_globals, "loss_sumaries_func")

        rl_loss_sumaries_globals = dill.loads(rl_loss_sumaries_globals)
        rl_loss_sumaries_globals = self.process_globals(rl_loss_sumaries_globals)
        rl_loss_sumaries_code = marshal.loads(rl_loss_sumaries_code)
        self.rl_loss_sumaries = types.FunctionType(rl_loss_sumaries_code, rl_loss_sumaries_globals,
                                                   "rl_loss_sumaries_func")

        rl_sumaries_globals = dill.loads(rl_sumaries_globals)
        rl_sumaries_globals = self.process_globals(rl_sumaries_globals)
        rl_sumaries_code = marshal.loads(rl_sumaries_code)
        self.rl_sumaries = types.FunctionType(rl_sumaries_code, rl_sumaries_globals, "rl_sumaries_func")

        self.total_epochs = data['total_epochs']
        self.train_log_dir = data['train_log_dir']

        if self.train_log_dir is not None:
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        else:
            self.train_summary_writer = None

        # self.optimizer_actor = tf.saved_model.load(os.path.join(path, 'optimizer_actor'))
        # self.optimizer_critic = tf.saved_model.load(os.path.join(path, 'optimizer_critic'))
        # self.metricst = tf.saved_model.load(os.path.join(path, 'metrics'))

        # TODO: falta cargar loss_func_actor y loss_func_critic
        # tf.saved_model.save(self.loss_func_actor, os.path.join(path, 'loss_func_actor'))
        # tf.saved_model.save(self.loss_func_critic, os.path.join(path, 'loss_func_critic'))

        self.load_checkpoint(path)

    def load_checkpoint(self, path=None, chckpoint_to_restore='latest'):
        if path is None:
            if chckpoint_to_restore == 'latest':
                self.net_chkpoint.restore(self.net_manager.latest_checkpoint)
            else:
                chck = self.net_manager.checkpoints
        else:
            net_chkpoint = tf.train.Checkpoint(model=self.net,
                                               optimizer=self.optimizer,
                                               # loss_func_actor=self.loss_func_actor,
                                               metrics=self.metrics)
            net_manager = tf.train.CheckpointManager(net_chkpoint,
                                                     os.path.join(path, 'net'),
                                                     checkpoint_name='net',
                                                     max_to_keep=1)
            net_chkpoint.restore(net_manager.latest_checkpoint)

    def get_weights(self):
        weights = []
        for layer in self.net.layers:
            weights.append(layer.get_weights())
        return weights

    def set_weights(self, weights):
        for layer, w in zip(self.net.layers, weights):
            layer.set_weights(w)


class DDPGNet(RLNetModel):
    def __init__(self, actor_net, critic_net, chckpoint_path=None, chckpoints_to_keep=10, tensorboard_dir=None):
        super().__init__(tensorboard_dir)

        self.actor_net = actor_net
        self.critic_net = critic_net

        self.actor_target_net = tf.keras.models.clone_model(actor_net)
        self.critic_target_net = tf.keras.models.clone_model(critic_net)

        # self._tensorboard_util(tensorboard_dir)
        # if tensorboard_dir is not None:
        #     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #     train_log_dir = os.path.join(tensorboard_dir, 'gradient_tape/' + current_time + '/train')
        #     self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # else:
        #     self.train_summary_writer = None

        self.total_epochs = 0
        self.loss_actor = None
        self.loss_critic = None
        self.optimizer = None
        self.metrics = None
        # self.loss_sumaries = tensor_board_loss_functions.loss_sumaries
        # self.rl_loss_sumaries = tensor_board_loss_functions.rl_loss_sumaries
        # self.rl_sumaries = tensor_board_loss_functions.rl_sumaries

        if chckpoint_path is not None:
            self.actor_chkpoint = tf.train.Checkpoint(model=self.actor_net)
            self.actor_manager = tf.train.CheckpointManager(self.actor_chkpoint,
                                                            os.path.join(chckpoint_path, 'actor', 'checkpoint'),
                                                            checkpoint_name='actor',
                                                            max_to_keep=chckpoints_to_keep)

            self.critic_chkpoint = tf.train.Checkpoint(model=self.critic_net)
            self.critic_manager = tf.train.CheckpointManager(self.critic_chkpoint,
                                                             os.path.join(chckpoint_path, 'critic', 'checkpoint'),
                                                             checkpoint_name='critic',
                                                             max_to_keep=chckpoints_to_keep)

    def compile(self, loss, optimizer, metrics=tf.keras.metrics.Accuracy()):
        self.loss_actor = loss[0]
        self.loss_critic = loss[1]
        self.optimizer_actor = optimizer[0]
        self.optimizer_critic = optimizer[1]
        self.metrics = metrics

    def summary(self):
        pass

    def predict(self, x):
        y_ = self._predict(x)
        return y_.numpy()

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, x):
        """ Predict the output sentence for a given input sentence
            Args:
                test_source_text: input sentence (raw string)

            Returns:
                The encoder's attention vectors
                The decoder's bottom attention vectors
                The decoder's middle attention vectors
                The input string array (input sentence split by ' ')
                The output string array
            """
        y_ = self.actor_net(tf.cast(x, tf.float32), training=False)
        return y_

    # @tf.function(experimental_relax_shapes=True)
    def train_step(self, obs, actions, next_obs, rewards, done, gamma):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            p_target = self.actor_target_net(next_obs)
            q_target = self.critic_target_net([next_obs, p_target])
            q_target = rewards + (1 - done) * gamma * q_target
            q_ = self.critic_net([obs, actions])
            loss_critic, loss_components_actor = self.loss_critic(q_, q_target)

        variables_critic = self.critic_net.trainable_variables
        gradients_critic = tape.gradient(loss_critic, variables_critic)
        self.optimizer_critic.apply_gradients(zip(gradients_critic, variables_critic))

        with tf.GradientTape() as tape:
            a_ = self.actor_net(obs)
            q_ = self.critic_net([obs, a_])
            loss_actor, loss_components_critic = self.loss_actor(q_)

        variables_actor = self.actor_net.trainable_variables
        gradients_actor = tape.gradient(loss_actor, variables_actor)
        self.optimizer_actor.apply_gradients(zip(gradients_actor, variables_actor))

        return [loss_actor, loss_critic], [gradients_actor, gradients_critic], [variables_actor, variables_critic], \
               [loss_components_actor, loss_components_critic]

    # @tf.function(experimental_relax_shapes=True)
    # def train_step(self, obs, actions, next_obs, rewards, gamma):
    #     """ Execute one training step (forward pass + backward pass)
    #     Args:
    #         source_seq: source sequences
    #         target_seq_in: input target sequences (<start> + ...)
    #         target_seq_out: output target sequences (... + <end>)
    #
    #     Returns:
    #         The loss value of the current pass
    #     """
    #     with tf.GradientTape() as tape:
    #         p_target = self.actor_target_net(obs)
    #         q_target = self.critic_target_net([next_obs, p_target])
    #         q_target = rewards + gamma * q_target
    #         # TODO: actions or self.predict(obs)
    #         a_ = self.actor_net(obs)
    #         q_ = self.critic_net([obs, a_])
    #         loss_critic, loss_components_actor = self.loss_critic(q_target, q_)
    #         loss_actor, loss_components_critic = self.loss_actor(q_)
    #
    #     variables_actor = self.actor_net.trainable_variables
    #     variables_critic = self.critic_net.trainable_variables
    #     gradients_actor, gradients_critic = tape.gradient([loss_actor, loss_critic],
    #                                                       [variables_actor, variables_critic])
    #
    #     self.optimizer_actor.apply_gradients(zip(gradients_actor, variables_actor))
    #     self.optimizer_critic.apply_gradients(zip(gradients_critic, variables_critic))
    #
    #     return [loss_actor, loss_critic], [gradients_actor, gradients_critic], [variables_actor, variables_critic], \
    #            [loss_components_actor, loss_components_critic]

    def fit(self, obs, next_obs, actions, rewards, done, epochs, batch_size, validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        gamma = kargs[0]
        tau = kargs[1]

        dataset = tf.data.Dataset.from_tensor_slices((obs,
                                                      next_obs,
                                                      actions,
                                                      rewards,
                                                      done))

        if shuffle:
            dataset = dataset.shuffle(len(obs), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            loss = 0.
            for batch, (bach_obs,
                        bach_next_obs,
                        bach_actions,
                        bach_rewards,
                        batch_done) in enumerate(dataset.take(-1)):

                loss, gradients, variables, loss_components = self.train_step(bach_obs,
                                                                              bach_actions,
                                                                              bach_next_obs,
                                                                              bach_rewards,
                                                                              batch_done,
                                                                              gamma)

                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print(
                        'Epoch {}\t Batch {}\t Loss Actor, Critic {:.4f}, {:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                            e + 1, batch, loss[0].numpy(), loss[1].numpy(), self.metrics.result(),
                            time.time() - start_time))
                    start_time = time.time()

                if self.train_summary_writer is not None:
                    with self.train_summary_writer.as_default():
                        self.loss_sumaries([loss[0], loss[1]], ['loss actor', 'loss critic'], self.total_epochs)
                        self.rl_loss_sumaries([np.float32(np.expand_dims(actions, axis=-1))],
                                              ['actions'],
                                              self.total_epochs)
            self.total_epochs += 1

            history.history['loss'].append(loss[0].numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)

        self.soft_replace(tau)
        return history

    def save(self, path):
        # Serializar función calculate_advanteges
        # calculate_advantages_globals = dill.dumps(self.calculate_advantages.__globals__)
        # calculate_advantages_globals = base64.b64encode(calculate_advantages_globals).decode('ascii')
        # calculate_advantages_code = marshal.dumps(self.calculate_advantages.__code__)
        # calculate_advantages_code = base64.b64encode(calculate_advantages_code).decode('ascii')

        # Serializar función loss_sumaries
        loss_sumaries_globals = dill.dumps(self.loss_sumaries.__globals__)
        loss_sumaries_globals = base64.b64encode(loss_sumaries_globals).decode('ascii')
        loss_sumaries_code = marshal.dumps(self.loss_sumaries.__code__)
        loss_sumaries_code = base64.b64encode(loss_sumaries_code).decode('ascii')

        # Serializar función rl_loss_sumaries
        rl_loss_sumaries_globals = dill.dumps(self.rl_loss_sumaries.__globals__)
        rl_loss_sumaries_globals = base64.b64encode(rl_loss_sumaries_globals).decode('ascii')
        rl_loss_sumaries_code = marshal.dumps(self.rl_loss_sumaries.__code__)
        rl_loss_sumaries_code = base64.b64encode(rl_loss_sumaries_code).decode('ascii')

        # Serializar función rl_sumaries
        rl_sumaries_globals = dill.dumps(self.rl_sumaries.__globals__)
        rl_sumaries_globals = base64.b64encode(rl_sumaries_globals).decode('ascii')
        rl_sumaries_code = marshal.dumps(self.rl_sumaries.__code__)
        rl_sumaries_code = base64.b64encode(rl_sumaries_code).decode('ascii')

        # saves actor and critic networks
        self.save_checkpoint(path)

        # TODO: Qeda guardar las funciones de pérdida. De momento confio en las que hay definidad como estandar.
        data = {
            "train_log_dir": self.train_log_dir,
            "total_epochs": self.total_epochs,
            # "calculate_advantages_globals": calculate_advantages_globals,
            # "calculate_advantages_code": calculate_advantages_code,
            "loss_sumaries_globals": loss_sumaries_globals,
            "loss_sumaries_code": loss_sumaries_code,
            "rl_loss_sumaries_globals": rl_loss_sumaries_globals,
            "rl_loss_sumaries_code": rl_loss_sumaries_code,
            "rl_sumaries_globals": rl_sumaries_globals,
            "rl_sumaries_code": rl_sumaries_code,
        }

        with open(os.path.join(path, 'model_data.json'), 'w') as f:
            json.dump(data, f)

    @tf.function(experimental_relax_shapes=True)
    def _bc_train_step(self, x, y):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            y_ = self.actor_net(x, training=True)
            loss = self.bc_loss_func(y, y_)

        self.bc_metrics.update_state(y, y_)

        variables = self.actor_net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.bc_optimizer.apply_gradients(zip(gradients, variables))

        return loss, gradients, variables

    @tf.function(experimental_relax_shapes=True)
    def _bc_evaluate(self, x, y):
        y_ = self.actor_net(x, training=False)

        loss = self.bc_loss_func(y, y_)

        self.bc_metrics.update_state(y, y_)

        return loss

    def save_checkpoint(self, path=None):
        if path is None:
            # Save a checkpoint
            self.actor_manager.save()
            self.critic_manager.save()
        else:
            actor_chkpoint = tf.train.Checkpoint(model=self.actor_net,
                                                 optimizer=self.optimizer_actor,
                                                 # loss_func_actor=self.loss_func_actor,
                                                 metrics=self.metrics,
                                                 )
            actor_manager = tf.train.CheckpointManager(actor_chkpoint,
                                                       os.path.join(path, 'actor'),
                                                       checkpoint_name='actor',
                                                       max_to_keep=1)
            actor_manager.save()

            critic_chkpoint = tf.train.Checkpoint(model=self.critic_net,
                                                  optimizer=self.optimizer_critic,
                                                  # loss_func_critic=self.loss_func_critic
                                                  )
            critic_manager = tf.train.CheckpointManager(critic_chkpoint,
                                                        os.path.join(path, 'critic'),
                                                        checkpoint_name='critic',
                                                        max_to_keep=1)
            critic_manager.save()

    def restore(self, path):
        with open(os.path.join(path, 'model_data.json'), 'r') as f:
            data = json.load(f)

        loss_sumaries_code = base64.b64decode(data['loss_sumaries_code'])
        loss_sumaries_globals = base64.b64decode(data['loss_sumaries_globals'])

        rl_loss_sumaries_code = base64.b64decode(data['rl_loss_sumaries_code'])
        rl_loss_sumaries_globals = base64.b64decode(data['rl_loss_sumaries_globals'])

        rl_sumaries_code = base64.b64decode(data['rl_sumaries_code'])
        rl_sumaries_globals = base64.b64decode(data['rl_sumaries_globals'])

        loss_sumaries_globals = dill.loads(loss_sumaries_globals)
        loss_sumaries_globals = self.process_globals(loss_sumaries_globals)
        loss_sumaries_code = marshal.loads(loss_sumaries_code)
        self.loss_sumaries = types.FunctionType(loss_sumaries_code, loss_sumaries_globals, "loss_sumaries_func")

        rl_loss_sumaries_globals = dill.loads(rl_loss_sumaries_globals)
        rl_loss_sumaries_globals = self.process_globals(rl_loss_sumaries_globals)
        rl_loss_sumaries_code = marshal.loads(rl_loss_sumaries_code)
        self.rl_loss_sumaries = types.FunctionType(rl_loss_sumaries_code, rl_loss_sumaries_globals,
                                                   "rl_loss_sumaries_func")

        rl_sumaries_globals = dill.loads(rl_sumaries_globals)
        rl_sumaries_globals = self.process_globals(rl_sumaries_globals)
        rl_sumaries_code = marshal.loads(rl_sumaries_code)
        self.rl_sumaries = types.FunctionType(rl_sumaries_code, rl_sumaries_globals, "rl_sumaries_func")

        self.total_epochs = data['total_epochs']
        self.train_log_dir = data['train_log_dir']

        if self.train_log_dir is not None:
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        else:
            self.train_summary_writer = None

        # self.optimizer_actor = tf.saved_model.load(os.path.join(path, 'optimizer_actor'))
        # self.optimizer_critic = tf.saved_model.load(os.path.join(path, 'optimizer_critic'))
        # self.metricst = tf.saved_model.load(os.path.join(path, 'metrics'))

        # TODO: falta cargar loss_func_actor y loss_func_critic
        # tf.saved_model.save(self.loss_func_actor, os.path.join(path, 'loss_func_actor'))
        # tf.saved_model.save(self.loss_func_critic, os.path.join(path, 'loss_func_critic'))

        self.load_checkpoint(path)

    def load_checkpoint(self, path=None, chckpoint_to_restore='latest'):
        if path is None:
            if chckpoint_to_restore == 'latest':
                self.actor_chkpoint.restore(self.actor_manager.latest_checkpoint)
                self.critic_chkpoint.restore(self.critic_manager.latest_checkpoint)
            else:
                chck = self.actor_manager.checkpoints
        else:
            actor_chkpoint = tf.train.Checkpoint(model=self.actor_net,
                                                 optimizer=self.optimizer_actor,
                                                 # loss_func_actor=self.loss_func_actor,
                                                 metrics=self.metrics)
            actor_manager = tf.train.CheckpointManager(actor_chkpoint,
                                                       os.path.join(path, 'actor'),
                                                       checkpoint_name='actor',
                                                       max_to_keep=1)
            actor_chkpoint.restore(actor_manager.latest_checkpoint)

            critic_chkpoint = tf.train.Checkpoint(model=self.critic_net,
                                                  optimizer=self.optimizer_critic,
                                                  # loss_func_critic=self.loss_func_critic
                                                  )
            critic_manager = tf.train.CheckpointManager(critic_chkpoint,
                                                        os.path.join(path, 'critic'),
                                                        checkpoint_name='critic',
                                                        max_to_keep=1)
            critic_chkpoint.restore(critic_manager.latest_checkpoint)

    def get_weights(self):
        weights = []
        for layer in self.actor_net.layers:
            weights.append(layer.get_weights())
        return weights

    def set_weights(self, weights):
        for layer, w in zip(self.actor_net.layers, weights):
            layer.set_weights(w)

    @tf.function(experimental_relax_shapes=True)
    def soft_replace(self, tau):
        actor_w = self.actor_net.trainable_variables
        actor_t_w = self.actor_target_net.trainable_variables
        critic_w = self.critic_net.trainable_variables
        critic_t_w = self.critic_target_net.trainable_variables

        for a_w, at_w, c_w, ct_w in zip(actor_w, actor_t_w, critic_w, critic_t_w):
            if isinstance(at_w, list) or isinstance(a_w, list):
                for target, main in zip(at_w, a_w):
                    target = (1 - tau) * target + tau * main
            else:
                at_w = (1 - tau) * at_w + tau * a_w
            if isinstance(ct_w, list) or isinstance(c_w, list):
                for target, main in zip(ct_w, c_w):
                    target = (1 - tau) * target + tau * main
            else:
                ct_w = (1 - tau) * ct_w + tau * c_w


class A2CNetDiscrete(RLNetModel):
    def __init__(self, actor_net, critic_net, tensorboard_dir=None):
        super().__init__(tensorboard_dir)

        self.actor_net = actor_net
        self.critic_net = critic_net

        # if tensorboard_dir is not None:
        #     current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        #     train_log_dir = os.path.join(tensorboard_dir, 'gradient_tape/' + current_time + '/train')
        #     self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        # else:
        #     self.train_summary_writer = None

        self.total_epochs = 0
        self.loss_func_actor = None
        self.loss_func_critic = None
        self.optimizer_actor = None
        self.optimizer_critic = None
        self.metrics = None
        self.calculate_returns = None
        # self.loss_sumaries = tensor_board_loss_functions.loss_sumaries
        # self.rl_loss_sumaries = tensor_board_loss_functions.rl_loss_sumaries
        # self.rl_sumaries = tensor_board_loss_functions.rl_sumaries

    def compile(self, loss, optimizer, metrics=tf.keras.metrics.Accuracy()):
        self.loss_func_actor = loss[0]
        self.loss_func_critic = loss[1]
        self.optimizer_actor = optimizer[0]
        self.optimizer_critic = optimizer[1]
        self.calculate_returns = returns_calculations.discount_and_norm_rewards
        self.metrics = metrics

    def summary(self):
        pass

    def predict(self, x):
        y_ = self._predict(x)
        return y_.numpy()

    @tf.function(experimental_relax_shapes=True)
    def _predict(self, x):
        """ Predict the output sentence for a given input sentence
            Args:
                test_source_text: input sentence (raw string)

            Returns:
                The encoder's attention vectors
                The decoder's bottom attention vectors
                The decoder's middle attention vectors
                The input string array (input sentence split by ' ')
                The output string array
            """
        y_ = self.actor_net(tf.cast(x, tf.float32), training=False)
        return y_

    # @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y, returns, entropy_beta=0.001):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            values = self.critic_net(x, training=True)
            y_ = self.actor_net(x, training=True)

            normal_dist = (y_ * y) + 1e-10  # +1e-10 to prevent zero values
            log_prob = tf.math.log(normal_dist)

            loss_critic, loss_components_critic = self.loss_func_critic(returns, values)
            td = returns - values
            entropy = -(y_ * tf.math.log(y_ + 1e-10))  # +1e-10 to prevent zero values

            loss_actor, [act_comp_loss, entropy_comp_loss] = self.loss_func_actor(log_prob, td, entropy_beta, entropy)

        self.metrics.update_state(y, y_)

        variables_actor = self.actor_net.trainable_variables
        variables_critic = self.critic_net.trainable_variables
        gradients_actor, gradients_critic = tape.gradient([loss_actor, loss_critic],
                                                          [variables_actor, variables_critic])
        self.optimizer_actor.apply_gradients(zip(gradients_actor, variables_actor))
        self.optimizer_critic.apply_gradients(zip(gradients_critic, variables_critic))

        return [loss_actor, loss_critic], \
               [gradients_actor, gradients_critic], \
               [variables_actor, variables_critic], \
               returns, \
               [[act_comp_loss, entropy_comp_loss], loss_components_critic]

    def fit(self, obs, next_obs, actions, rewards, done, epochs, batch_size, validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        gamma = kargs[0]
        entropy_beta = kargs[1]
        n_step_return = kargs[2]

        returns = self.calculate_returns(rewards, np.logical_not(done), gamma, norm=True, n_step_return=n_step_return)

        dataset = tf.data.Dataset.from_tensor_slices((np.float32(obs),
                                                      np.float32(next_obs),
                                                      np.float32(rewards),
                                                      np.float32(actions),
                                                      np.float32(done),
                                                      np.float32(returns)))

        if shuffle:
            dataset = dataset.shuffle(len(obs), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history_actor = TrainingHistory()
        history_critic = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            act_comp_loss = 0.
            critic_comp_loss = 0.
            entropy_comp_loss = 0.
            loss = 0.
            for batch, (batch_obs,
                        batch_next_obs,
                        batch_rewards,
                        batch_actions,
                        batch_done,
                        batch_returns) in enumerate(dataset.take(-1)):
                loss, \
                gradients, \
                variables, \
                returns, \
                [[act_comp_loss, entropy_comp_loss],
                 loss_components_critic] = self.train_step(batch_obs,
                                                           batch_actions,
                                                           batch_returns,
                                                           entropy_beta=entropy_beta)

                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print(
                        'Epoch {}\t Batch {}\t Loss Actor\Critic {:.4f}\{:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                            e + 1, batch, loss[0].numpy(), loss[1].numpy(), self.metrics.result(),
                            time.time() - start_time))
                    start_time = time.time()

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    self.loss_sumaries([loss[0],
                                        loss[1],
                                        tf.math.reduce_mean(act_comp_loss),
                                        tf.math.reduce_mean(entropy_comp_loss),
                                        tf.math.reduce_mean(entropy_beta * entropy_comp_loss)],
                                       ['actor_model_loss (a_l + b*e_l)',
                                        'critic_model_loss (mse)',
                                        'actor_loss_component (a_l)',
                                        'entropy_loss_component (e_l)',
                                        '(b*el)'], self.total_epochs)
                    self.rl_loss_sumaries([np.float32(np.expand_dims(actions, axis=-1)),
                                           np.float32(np.expand_dims(returns, axis=-1))],
                                          ['actions',
                                           'returns'],
                                          self.total_epochs)
            self.total_epochs += 1

            history_actor.history['loss'].append(loss[0].numpy())
            history_critic.history['loss'].append(loss[1].numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)
        return history_actor, history_critic

    @tf.function(experimental_relax_shapes=True)
    def _bc_train_step(self, x, y):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            y_ = self.actor_net(x, training=True)
            loss = self.bc_loss_func(y, y_)

        self.bc_metrics.update_state(y, y_)

        variables = self.actor_net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.bc_optimizer.apply_gradients(zip(gradients, variables))

        return loss, gradients, variables

    @tf.function(experimental_relax_shapes=True)
    def _bc_evaluate(self, x, y):
        y_ = self.actor_net(x, training=False)

        loss = self.bc_loss_func(y, y_)

        self.bc_metrics.update_state(y, y_)

        return loss

    def save(self, path):
        # Serializar función calculate_advanteges
        # calculate_advantages_globals = dill.dumps(self.calculate_advantages.__globals__)
        # calculate_advantages_globals = base64.b64encode(calculate_advantages_globals).decode('ascii')
        # calculate_advantages_code = marshal.dumps(self.calculate_advantages.__code__)
        # calculate_advantages_code = base64.b64encode(calculate_advantages_code).decode('ascii')

        # Serializar función loss_sumaries
        loss_sumaries_globals = dill.dumps(self.loss_sumaries.__globals__)
        loss_sumaries_globals = base64.b64encode(loss_sumaries_globals).decode('ascii')
        loss_sumaries_code = marshal.dumps(self.loss_sumaries.__code__)
        loss_sumaries_code = base64.b64encode(loss_sumaries_code).decode('ascii')

        # Serializar función rl_loss_sumaries
        rl_loss_sumaries_globals = dill.dumps(self.rl_loss_sumaries.__globals__)
        rl_loss_sumaries_globals = base64.b64encode(rl_loss_sumaries_globals).decode('ascii')
        rl_loss_sumaries_code = marshal.dumps(self.rl_loss_sumaries.__code__)
        rl_loss_sumaries_code = base64.b64encode(rl_loss_sumaries_code).decode('ascii')

        # Serializar función rl_sumaries
        rl_sumaries_globals = dill.dumps(self.rl_sumaries.__globals__)
        rl_sumaries_globals = base64.b64encode(rl_sumaries_globals).decode('ascii')
        rl_sumaries_code = marshal.dumps(self.rl_sumaries.__code__)
        rl_sumaries_code = base64.b64encode(rl_sumaries_code).decode('ascii')

        # saves actor and critic networks
        self.save_checkpoint(path)

        # TODO: Qeda guardar las funciones de pérdida. De momento confio en las que hay definidad como estandar.
        data = {
            "train_log_dir": self.train_log_dir,
            "total_epochs": self.total_epochs,
            # "calculate_advantages_globals": calculate_advantages_globals,
            # "calculate_advantages_code": calculate_advantages_code,
            "loss_sumaries_globals": loss_sumaries_globals,
            "loss_sumaries_code": loss_sumaries_code,
            "rl_loss_sumaries_globals": rl_loss_sumaries_globals,
            "rl_loss_sumaries_code": rl_loss_sumaries_code,
            "rl_sumaries_globals": rl_sumaries_globals,
            "rl_sumaries_code": rl_sumaries_code,
        }

        with open(os.path.join(path, 'model_data.json'), 'w') as f:
            json.dump(data, f)

    def restore(self, path):
        with open(os.path.join(path, 'model_data.json'), 'r') as f:
            data = json.load(f)

        calculate_advantages_code = base64.b64decode(data['calculate_advantages_code'])
        calculate_advantages_globals = base64.b64decode(data['calculate_advantages_globals'])

        loss_sumaries_code = base64.b64decode(data['loss_sumaries_code'])
        loss_sumaries_globals = base64.b64decode(data['loss_sumaries_globals'])

        rl_loss_sumaries_code = base64.b64decode(data['rl_loss_sumaries_code'])
        rl_loss_sumaries_globals = base64.b64decode(data['rl_loss_sumaries_globals'])

        rl_sumaries_code = base64.b64decode(data['rl_sumaries_code'])
        rl_sumaries_globals = base64.b64decode(data['rl_sumaries_globals'])

        calculate_advantages_globals = dill.loads(calculate_advantages_globals)
        calculate_advantages_globals = self.process_globals(calculate_advantages_globals)
        calculate_advantages_code = marshal.loads(calculate_advantages_code)
        self.calculate_advantages = types.FunctionType(calculate_advantages_code, calculate_advantages_globals,
                                                       "calculate_advantages_func")

        loss_sumaries_globals = dill.loads(loss_sumaries_globals)
        loss_sumaries_globals = self.process_globals(loss_sumaries_globals)
        loss_sumaries_code = marshal.loads(loss_sumaries_code)
        self.loss_sumaries = types.FunctionType(loss_sumaries_code, loss_sumaries_globals, "loss_sumaries_func")

        rl_loss_sumaries_globals = dill.loads(rl_loss_sumaries_globals)
        rl_loss_sumaries_globals = self.process_globals(rl_loss_sumaries_globals)
        rl_loss_sumaries_code = marshal.loads(rl_loss_sumaries_code)
        self.rl_loss_sumaries = types.FunctionType(rl_loss_sumaries_code, rl_loss_sumaries_globals,
                                                   "rl_loss_sumaries_func")

        rl_sumaries_globals = dill.loads(rl_sumaries_globals)
        rl_sumaries_globals = self.process_globals(rl_sumaries_globals)
        rl_sumaries_code = marshal.loads(rl_sumaries_code)
        self.rl_sumaries = types.FunctionType(rl_sumaries_code, rl_sumaries_globals, "rl_sumaries_func")

        self.total_epochs = data['total_epochs']
        self.train_log_dir = data['train_log_dir']

        if self.train_log_dir is not None:
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        else:
            self.train_summary_writer = None
        # self.optimizer_actor = tf.saved_model.load(os.path.join(path, 'optimizer_actor'))
        # self.optimizer_critic = tf.saved_model.load(os.path.join(path, 'optimizer_critic'))
        # self.metricst = tf.saved_model.load(os.path.join(path, 'metrics'))

        # TODO: falta cargar loss_func_actor y loss_func_critic
        # tf.saved_model.save(self.loss_func_actor, os.path.join(path, 'loss_func_actor'))
        # tf.saved_model.save(self.loss_func_critic, os.path.join(path, 'loss_func_critic'))

        self.load_checkpoint(path)

    def save_checkpoint(self, path=None):
        if path is None:
            # Save a checkpoint
            self.actor_manager.save()
            self.critic_manager.save()
        else:
            actor_chkpoint = tf.train.Checkpoint(model=self.actor_net,
                                                 optimizer=self.optimizer_actor,
                                                 # loss_func_actor=self.loss_func_actor,
                                                 metrics=self.metrics,
                                                 )
            actor_manager = tf.train.CheckpointManager(actor_chkpoint,
                                                       os.path.join(path, 'actor'),
                                                       checkpoint_name='actor',
                                                       max_to_keep=1)
            actor_manager.save()

            critic_chkpoint = tf.train.Checkpoint(model=self.critic_net,
                                                  optimizer=self.optimizer_critic,
                                                  # loss_func_critic=self.loss_func_critic
                                                  )
            critic_manager = tf.train.CheckpointManager(critic_chkpoint,
                                                        os.path.join(path, 'critic'),
                                                        checkpoint_name='critic',
                                                        max_to_keep=1)
            critic_manager.save()

    def load_checkpoint(self, path=None, chckpoint_to_restore='latest'):
        if path is None:
            if chckpoint_to_restore == 'latest':
                self.actor_chkpoint.restore(self.actor_manager.latest_checkpoint)
                self.critic_chkpoint.restore(self.critic_manager.latest_checkpoint)
            else:
                chck = self.actor_manager.checkpoints
        else:
            actor_chkpoint = tf.train.Checkpoint(model=self.actor_net,
                                                 optimizer=self.optimizer_actor,
                                                 # loss_func_actor=self.loss_func_actor,
                                                 metrics=self.metrics)
            actor_manager = tf.train.CheckpointManager(actor_chkpoint,
                                                       os.path.join(path, 'actor'),
                                                       checkpoint_name='actor',
                                                       max_to_keep=1)
            actor_chkpoint.restore(actor_manager.latest_checkpoint)

            critic_chkpoint = tf.train.Checkpoint(model=self.critic_net,
                                                  optimizer=self.optimizer_critic,
                                                  # loss_func_critic=self.loss_func_critic
                                                  )
            critic_manager = tf.train.CheckpointManager(critic_chkpoint,
                                                        os.path.join(path, 'critic'),
                                                        checkpoint_name='critic',
                                                        max_to_keep=1)
            critic_chkpoint.restore(critic_manager.latest_checkpoint)


class A2CNetContinuous(A2CNetDiscrete):
    def __init__(self, actor_net, critic_net, tensorboard_dir=None):
        super().__init__(actor_net, critic_net, tensorboard_dir)

    def predict(self, x):
        y_ = self._predict(x)
        y_ = np.random.normal(y_[0].numpy(), y_[1].numpy())
        return y_

    # @tf.function(experimental_relax_shapes=True)
    def train_step(self, x, y, returns, entropy_beta=0.001):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            values = self.critic_net(x, training=True)
            y_ = self.actor_net(x, training=True)

            normal_dist = tfp.distributions.Normal(y_[0], y_[1])

            log_prob = normal_dist.log_prob(y)
            adv = returns - values
            entropy = normal_dist.entropy()

            loss_actor, [act_comp_loss, entropy_comp_loss] = self.loss_func_actor(log_prob, adv, entropy_beta, entropy)
            loss_critic, loss_components_critic = self.loss_func_critic(returns, values)

        y_sampled = normal_dist.sample((1,))[0]
        self.metrics.update_state(y, y_sampled)

        variables_actor = self.actor_net.trainable_variables
        variables_critic = self.critic_net.trainable_variables
        gradients_actor, gradients_critic = tape.gradient([loss_actor, loss_critic],
                                                          [variables_actor, variables_critic])
        self.optimizer_actor.apply_gradients(zip(gradients_actor, variables_actor))
        self.optimizer_critic.apply_gradients(zip(gradients_critic, variables_critic))

        return [loss_actor, loss_critic], \
               [gradients_actor, gradients_critic], \
               [variables_actor, variables_critic], \
               returns, \
               [[act_comp_loss, entropy_comp_loss, y_[0], y_[1]], loss_components_critic]

    def fit(self, obs, next_obs, actions, rewards, done, epochs, batch_size, validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        gamma = kargs[0]
        entropy_beta = kargs[1]
        n_step_return = kargs[2]

        returns = self.calculate_returns(rewards, np.logical_not(done), gamma, norm=False, n_step_return=n_step_return)

        dataset = tf.data.Dataset.from_tensor_slices((np.float32(obs),
                                                      np.float32(next_obs),
                                                      np.float32(rewards),
                                                      np.float32(actions),
                                                      np.float32(done),
                                                      np.float32(returns)))

        if shuffle:
            dataset = dataset.shuffle(len(obs), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history_actor = TrainingHistory()
        history_critic = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            act_comp_loss = 0.
            critic_comp_loss = 0.
            entropy_comp_loss = 0.
            loss = 0.
            for batch, (batch_obs,
                        batch_next_obs,
                        batch_rewards,
                        batch_actions,
                        batch_done,
                        batch_returns) in enumerate(dataset.take(-1)):
                loss, \
                gradients, \
                variables, \
                returns, \
                [[act_comp_loss, entropy_comp_loss, pred_act, pred_std],
                 loss_components_critic] = self.train_step(batch_obs,
                                                           batch_actions,
                                                           batch_returns,
                                                           entropy_beta=entropy_beta)

                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print(
                        'Epoch {}\t Batch {}\t Loss Actor\Critic {:.4f}\{:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                            e + 1, batch, loss[0].numpy(), loss[1].numpy(), self.metrics.result(),
                            time.time() - start_time))
                    start_time = time.time()

                if self.train_summary_writer is not None:
                    with self.train_summary_writer.as_default():
                        self.loss_sumaries([loss[0],
                                            loss[1],
                                            tf.math.reduce_mean(act_comp_loss),
                                            tf.math.reduce_mean(entropy_comp_loss),
                                            tf.math.reduce_mean(entropy_beta * entropy_comp_loss)],
                                           ['actor_model_loss (a_l + b*e_l)',
                                            'critic_model_loss (mse)',
                                            'actor_loss_component (a_l)',
                                            'entropy_loss_component (e_l)',
                                            '(b*el)'], self.total_epochs)
                        self.rl_loss_sumaries([np.float32(np.expand_dims(actions, axis=-1)),
                                               np.float32(np.expand_dims(returns, axis=-1)),
                                               np.float32(np.expand_dims(pred_act, axis=-1)),
                                               np.float32(np.expand_dims(pred_std, axis=-1))],
                                              ['actions',
                                               'returns',
                                               'predicted action',
                                               'predicted stddev'],
                                              self.total_epochs)
            self.total_epochs += 1

            history_actor.history['loss'].append(loss[0].numpy())
            history_critic.history['loss'].append(loss[1].numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)
        return history_actor, history_critic

    def bc_train_step(self, x, y):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        stddev = tf.math.reduce_std(y, axis=0)
        stddev = np.array([stddev for i in range(y.shape[0])])
        y = np.array([y, stddev])
        return self._bc_train_step(x, y)

    @tf.function(experimental_relax_shapes=True)
    def _bc_train_step(self, x, y):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            y_ = self.actor_net(x, training=True)
            loss = self.bc_loss_func(y, y_)

        self.bc_metrics.update_state(y, y_)

        variables = self.actor_net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.bc_optimizer.apply_gradients(zip(gradients, variables))

        return loss, gradients, variables

    def bc_evaluate(self, x, y):
        stddev = tf.math.reduce_std(y, axis=0)
        stddev = np.array([stddev for i in range(y.shape[0])])
        y = np.array([y, stddev])

        return self._bc_evaluate(x, y)

    @tf.function(experimental_relax_shapes=True)
    def _bc_evaluate(self, x, y):
        y_ = self.actor_net(x, training=False)

        loss = self.bc_loss_func(y, y_)

        self.bc_metrics.update_state(y, y_)

        return loss


class A2CNetQueueDiscrete(A2CNetDiscrete):
    def fit(self, obs, next_obs, actions, rewards, done, epochs, batch_size, validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        gamma = kargs[0]
        entropy_beta = kargs[1]
        n_step_return = kargs[2]
        returns = kargs[3]

        dataset = tf.data.Dataset.from_tensor_slices((np.float32(obs),
                                                      np.float32(next_obs),
                                                      np.float32(rewards),
                                                      np.float32(actions),
                                                      np.float32(done),
                                                      np.float32(returns)))

        if shuffle:
            dataset = dataset.shuffle(len(obs), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history_actor = TrainingHistory()
        history_critic = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            loss = 0.
            for batch, (batch_obs,
                        batch_next_obs,
                        batch_rewards,
                        batch_actions,
                        batch_done,
                        batch_returns) in enumerate(dataset.take(-1)):
                loss, \
                gradients, \
                variables, \
                returns, \
                [[act_comp_loss, entropy_comp_loss],
                 loss_components_critic] = self.train_step(batch_obs,
                                                           batch_actions,
                                                           batch_returns,
                                                           entropy_beta=entropy_beta)

                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print(
                        'Epoch {}\t Batch {}\t Loss Actor\Critic {:.4f}\{:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                            e + 1, batch, loss[0].numpy(), loss[1].numpy(), self.metrics.result(),
                            time.time() - start_time))
                    start_time = time.time()

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    self.loss_sumaries([loss[0],
                                        loss[1],
                                        tf.math.reduce_mean(act_comp_loss),
                                        tf.math.reduce_mean(entropy_comp_loss),
                                        tf.math.reduce_mean(entropy_beta * entropy_comp_loss)],
                                       ['actor_model_loss (a_l + b*e_l)',
                                        'critic_model_loss (mse)',
                                        'actor_loss_component (a_l)',
                                        'entropy_loss_component (e_l)',
                                        '(b*el)'], self.total_epochs)
                    self.rl_loss_sumaries([np.float32(np.expand_dims(actions, axis=-1)),
                                           np.float32(np.expand_dims(returns, axis=-1))],
                                          ['actions',
                                           'returns'],
                                          self.total_epochs)
            self.total_epochs += 1

            history_actor.history['loss'].append(loss[0].numpy())
            history_critic.history['loss'].append(loss[1].numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)
        return history_actor, history_critic


class A2CNetQueueContinuous(A2CNetContinuous):
    def fit(self, obs, next_obs, actions, rewards, done, epochs, batch_size, validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        gamma = kargs[0]
        entropy_beta = kargs[1]
        n_step_return = kargs[2]
        returns = kargs[3]

        dataset = tf.data.Dataset.from_tensor_slices((np.float32(obs),
                                                      np.float32(next_obs),
                                                      np.float32(rewards),
                                                      np.float32(actions),
                                                      np.float32(done),
                                                      np.float32(returns)))

        if shuffle:
            dataset = dataset.shuffle(len(obs), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history_actor = TrainingHistory()
        history_critic = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            loss = 0.
            for batch, (batch_obs,
                        batch_next_obs,
                        batch_rewards,
                        batch_actions,
                        batch_done,
                        batch_returns) in enumerate(dataset.take(-1)):
                loss, \
                gradients, \
                variables, \
                returns, \
                [[act_comp_loss, entropy_comp_loss, pred_act, pred_std],
                 loss_components_critic] = self.train_step(batch_obs,
                                                           batch_actions,
                                                           batch_returns,
                                                           entropy_beta=entropy_beta)

                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print(
                        'Epoch {}\t Batch {}\t Loss Actor\Critic {:.4f}\{:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                            e + 1, batch, loss[0].numpy(), loss[1].numpy(), self.metrics.result(),
                            time.time() - start_time))
                    start_time = time.time()

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    self.loss_sumaries([loss[0],
                                        loss[1],
                                        tf.math.reduce_mean(act_comp_loss),
                                        tf.math.reduce_mean(entropy_comp_loss),
                                        tf.math.reduce_mean(entropy_beta * entropy_comp_loss)],
                                       ['actor_model_loss (a_l + b*e_l)',
                                        'critic_model_loss (mse)',
                                        'actor_loss_component (a_l)',
                                        'entropy_loss_component (e_l)',
                                        '(b*el)'], self.total_epochs)
                    self.rl_loss_sumaries([np.float32(np.expand_dims(actions, axis=-1)),
                                           np.float32(np.expand_dims(returns, axis=-1)),
                                           np.float32(np.expand_dims(pred_act, axis=-1)),
                                           np.float32(np.expand_dims(pred_std, axis=-1))],
                                          ['actions',
                                           'returns',
                                           'predicted action',
                                           'predicted stddev'],
                                          self.total_epochs)
            self.total_epochs += 1

            history_actor.history['loss'].append(loss[0].numpy())
            history_critic.history['loss'].append(loss[1].numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)
        return history_actor, history_critic


class HabitatPPONet(PPONet):
    """
    Define Custom Net for habitat
    """

    def __init__(self, input_shape, actor_net, critic_net, tensorboard_dir=None):
        super().__init__(actor_net(input_shape), critic_net(input_shape), tensorboard_dir=tensorboard_dir)

    # TODO: [Sergio]: Standardize the inputs to _train_step(). We have two inputs (x[0]=rgb y x[1]=objectgoal) but in
    #   a generic problem we may have a different number of inputs.
    def predict(self, x):
        y_ = self._predict(np.array(x[0]), np.array(x[1]))
        return y_.numpy()

    # @tf.function(experimental_relax_shapes=False)
    def _predict(self, x1, x2):
        """ Predict the output sentence for a given input sentence
            Args:
                test_source_text: input sentence (raw string)

            Returns:
                The encoder's attention vectors
                The decoder's bottom attention vectors
                The decoder's middle attention vectors
                The input string array (input sentence split by ' ')
                The output string array
            """
        # out = self.actor_net(tf.cast(np.array(x[0]), tf.float32), tf.cast(np.array(x[1]), tf.float32), training=False)
        out = self.actor_net([x1, x2], training=False)

        return out

    # TODO: [Sergio]: Standardize the inputs to _train_step(). We have two inputs (x[0]=rgb y x[1]=objectgoal) but in
    #   a generic problem we may have a different number of inputs.
    def predict_values(self, x):
        y_ = self._predict_values(np.array(x[0]), np.array(x[1]))
        return y_.numpy()

    @tf.function(experimental_relax_shapes=False)
    def _predict_values(self, x1, x2):
        out = self.critic_net([x1, x2], training=False)
        return out

    def fit(self, obs, next_obs, actions, rewards, done, epochs, batch_size, validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        act_probs = kargs[0]
        mask = kargs[1]
        stddev = kargs[2]
        loss_clipping = kargs[3]
        critic_discount = kargs[4]
        entropy_beta = kargs[5]
        gamma = kargs[6]
        lmbda = kargs[7]

        # Calculate returns and advantages
        returns = []
        advantages = []

        # TODO: [CARLOS] check if this split makes sense at all (specially the +1). Maybe using a ceiling instead of
        #   int in order to fit the rest of the observations.
        batch_obs = np.array_split(obs[0], int(rewards.shape[0] / batch_size) + 1)
        batch_target = np.array_split(obs[1], int(rewards.shape[0] / batch_size) + 1)
        batch_rewards = np.array_split(rewards, int(rewards.shape[0] / batch_size) + 1)
        batch_mask = np.array_split(mask, int(rewards.shape[0] / batch_size) + 1)

        for b_o, b_t, b_r, b_m in zip(batch_obs, batch_target, batch_rewards, batch_mask):
            values = self.predict_values([b_o, b_t])
            ret, adv = self.calculate_advantages(values, b_m, b_r, gamma, lmbda)
            returns.extend(ret)
            advantages.extend(adv)

        dataset = tf.data.Dataset.from_tensor_slices((tf.cast(obs[0], tf.float32),
                                                      tf.cast(obs[1], tf.float32),
                                                      tf.cast(act_probs, tf.float32),
                                                      tf.cast(rewards, tf.float32),
                                                      tf.cast(actions, tf.float32),
                                                      tf.cast(mask, tf.float32),
                                                      tf.cast(returns, tf.float32),
                                                      tf.cast(advantages, tf.float32)))

        if shuffle:
            dataset = dataset.shuffle(len(obs[0]), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        if self.train_summary_writer is not None:
            with self.train_summary_writer.as_default():
                self.rl_loss_sumaries([np.array(returns),
                                       np.array(advantages),
                                       actions,
                                       act_probs,
                                       stddev],
                                      ['returns',
                                       'advantages',
                                       'actions',
                                       'act_probabilities'
                                       'stddev']
                                      , self.total_epochs)

        history_actor = TrainingHistory()
        history_critic = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            loss = [0., 0.]
            act_comp_loss = 0.
            critic_comp_loss = 0.
            entropy_comp_loss = 0.
            for batch, (batch_obs,
                        batch_target,
                        batch_act_probs,
                        batch_rewards,
                        batch_actions,
                        batch_mask,
                        batch_returns,
                        batch_advantages) in enumerate(dataset.take(-1)):
                loss, \
                gradients, \
                variables, \
                returns, \
                advantages, \
                [act_comp_loss, critic_comp_loss, entropy_comp_loss] = self.train_step(
                    [tf.cast(batch_obs, tf.float32), tf.cast(batch_target, tf.float32)],
                    tf.cast(batch_act_probs, tf.float32),
                    tf.cast(batch_actions, tf.float32),
                    tf.cast(batch_returns, tf.float32),
                    tf.cast(batch_advantages, tf.float32),
                    stddev=tf.cast(stddev,
                                   tf.float32),
                    loss_clipping=tf.cast(
                        loss_clipping,
                        tf.float32),
                    critic_discount=tf.cast(
                        critic_discount,
                        tf.float32),
                    entropy_beta=tf.cast(
                        entropy_beta,
                        tf.float32))

            if verbose:
                print(
                    'Epoch {}\t Loss Actor\Critic {:.4f}\{:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, loss[0].numpy(), loss[1].numpy(), self.metrics.result(),
                        time.time() - start_time))
                start_time = time.time()

            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    self.loss_sumaries([loss[0],
                                        loss[1],
                                        act_comp_loss,
                                        critic_comp_loss,
                                        entropy_comp_loss,
                                        critic_discount * critic_comp_loss,
                                        entropy_beta * entropy_comp_loss],
                                       ['actor_model_loss (-a_l + c*c_l - b*e_l)',
                                        'critic_model_loss',
                                        'actor_component (a_l)',
                                        'critic_component (c_l)',
                                        'entropy_component (e_l)',
                                        '(c*c_l)',
                                        '(b*e_l)'],
                                       self.total_epochs)

            self.total_epochs += 1

            history_actor.history['loss'].append(loss[0].numpy())
            history_critic.history['loss'].append(loss[1].numpy())

            if callbacks is not None:
                for cb in callbacks:
                    cb.on_epoch_end(e)

        return history_actor, history_critic

    def train_step(self, x, old_prediction, y, returns, advantages, stddev=None, loss_clipping=0.3,
                   critic_discount=0.5, entropy_beta=0.001):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        # TODO: [Sergio]: Standardize the inputs to _train_step(). We have two inputs (x[0]=rgb y x[1]=objectgoal) but in
        #   a generic problem we may have a different number of inputs.
        return self._train_step(x[0], x[1], old_prediction, y, returns, advantages, stddev, loss_clipping,
                                critic_discount, entropy_beta)

    # TODO: [Sergio]: Standardize the inputs to _train_step(). We have two inputs (x[0]=rgb y x[1]=objectgoal) but in
    #   a generic problem we may have a different number of inputs.
    @tf.function(experimental_relax_shapes=True)
    def _train_step(self, x_rgb, x_objgoal, old_prediction, y, returns, advantages, stddev=None, loss_clipping=0.3,
                    critic_discount=0.5, entropy_beta=0.001):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            values = self.critic_net([x_rgb, x_objgoal], training=True)
            y_ = self.actor_net([x_rgb, x_objgoal], training=True)
            loss_actor, [act_comp_loss, critic_comp_loss, entropy_comp_loss] = self.loss_func_actor(y, y_,
                                                                                                    advantages,
                                                                                                    old_prediction,
                                                                                                    returns, values,
                                                                                                    stddev,
                                                                                                    loss_clipping,
                                                                                                    critic_discount,
                                                                                                    entropy_beta)
            loss_critic = self.loss_func_critic(returns, values)

        self.metrics.update_state(y, y_)

        variables_actor = self.actor_net.trainable_variables
        variables_critic = self.critic_net.trainable_variables
        gradients_actor, gradients_critic = tape.gradient([loss_actor, loss_critic],
                                                          [variables_actor, variables_critic])
        self.optimizer_actor.apply_gradients(zip(gradients_actor, variables_actor))
        self.optimizer_critic.apply_gradients(zip(gradients_critic, variables_critic))

        return [loss_actor, loss_critic], \
               [gradients_actor, gradients_critic], \
               [variables_actor, variables_critic], \
               returns, \
               advantages, \
               [act_comp_loss, critic_comp_loss, entropy_comp_loss]


class MazePPONet(PPONet):
    """
    Define Custom Net for Maze
    """

    def __init__(self,
                 input_shape,
                 actor_net,
                 critic_net,
                 tensorboard_dir=None,
                 checkpoint_path=None,
                 checkpoints_to_keep=None,
                 save_every_iterations=100):
        super().__init__(actor_net(input_shape),
                         critic_net(input_shape),
                         tensorboard_dir=tensorboard_dir,
                         checkpoints_to_keep=checkpoints_to_keep,
                         save_every_iterations=save_every_iterations,
                         checkpoint_path=checkpoint_path)
