import time
import datetime
import os
import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
from RL_Agent.base.utils.networks import tensor_board_loss_functions


class RLNetInterfaz(object, metaclass=ABCMeta):
    """
    A class for defining your own computation graph without worrying about the reinforcement learning procedure. Her
    e you are able to completely configure your neural network, optimizer, loss function, metrics and almost do anything
    else tensorflow allows. You can even use tensorboard to monitor the behaviour of the computation graph further than
    the metrics recorded by this library.
    """

    def __init__(self):
        super().__init__()
        self.optimizer = None   # Optimization algorithm form tensorflow or keras
        self.loss_func = None
        self.metrics = None

    @abstractmethod
    def compile(self, loss, optimizer, metrics=None):
        """
        Compile the neural network. Usually used for define loss function, optimizer and metrics.
        :param loss: Loss function from tensorflow, keras or user defined.
        :param optimizer:  Optimizer from tensorflow, keras or user defined.
        :param metrics: Metrics from tensorflow, keras or user defined.
        """
        self.loss_func = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def summary(self):
        pass

    @abstractmethod
    def predict(self, x):
        """
        Makes a prediction over the input x.
        :param x: (numpy nd array) input to neural network
        :return: (numpy nd array) output of the neural network. If tensorflow is working in eager mode try .numpy() over
                 the tensor returned from the neural network.
        """
        pass

    @abstractmethod
    def fit(self, obs, next_obs, actions, rewards, done, epochs, batch_size, validation_split=0.,
            shuffle=True, verbose=1, callbacks=None, kargs=[]):
        """
        Method for training the neural network.
        :param obs: (numpy nd array) Agent observations.
        :param next_obs: (numpy nd array) Agent next observations (the observations in the next time step to obs).
        :param actions: (numpy nd array) Agent actions. One hot encoded for discrete action spaces.
        :param rewards: (numpy 1D array of floats) Rewards related to transitions from obs to next_obs.
        :param done: (numpy 1D array of bools) True values correspond to final estate experience.
        :params advantages: (numpy 1D array of floats) Advantages calculated for each experience.
        :param epochs: (int) Training epochs.
        :param batch_size: (int) Training bach size.
        :param validation_split: (float in [0, 1]) Rate of data used for validation.
        :param shuffle: (bool) Shuffle or not the data.
        :param verbose: (int) If 0 do not print anything, greater numbers than zero print more or less info.
        :param callbacks: (function) Callbacks to apply during  training.
        :param kargs: (list) Other arguments.
        :returns: RL_Agent.base.utils.network_interface.TariningHistory object.
        """

    @abstractmethod
    def save(self, path):
        """ Serialize the class for saving with RL_Agent.base.utils.agent_saver.py utilities.
        :return: serialized data
        """

    @abstractmethod
    def bc_compile(self, loss, optimizer, metrics=None):
        """
        Compile the neural network. Usually used for define loss function, optimizer and metrics.
        :param loss: Loss function from tensorflow, keras or user defined.
        :param optimizer:  Optimizer from tensorflow, keras or user defined.
        :param metrics: Metrics from tensorflow, keras or user defined.
        """

    @abstractmethod
    def bc_fit(self, x, y, epochs, batch_size,  validation_split=0., shuffle=True, verbose=1, callbacks=None,
               kargs=[]):
        """
        :return:
        """

    def get_weights(self):
        """
        Returns the weights of all neural network variables in a numpy nd array. This method must be implemented when
        using DQN, DDQN, DDDQN or DDPG because these agents has transference of information between the main networks
        and target networks.
        An example of implementation when using keras Sequential models for defining the network:
        ###########################################
            weights = []
            for layer in self.net.layers:
                weights.append(layer.get_weights())
            return np.array(weights)
        ###########################################
        """

    def set_weights(self, weights):
        """
        Set the weights of all neural network variables. Input is a numpy nd array. This method must be implemented when
        using DQN, DDQN, DDDQN or DDPG because these agents has transference of information between the main networks
        and target networks.
        An example of implementation when using keras Sequential models for defining the network:
        ###########################################
            for layer, w in zip(self.net.layers, weights):
                layer.set_weights(w)
        ###########################################
        """

    def copy_model_to_target(self):
        """
        Copy the main network/s weights into the target network/s after each episode. All main and target networks must
        be defined inside the instantiation of the RLNetModel. Not all the agent included in this library uses target
        networks, in those cases the implementation of this method consist of doing nothing. Mainly DQN based methods
        use to require the use of target networks. DDPG algo use target networks but they use te be updated after each
        training step and do not necessarily requieres the implementation of this methos.
        An example of implementation when using keras Sequential models for a DQN problem::
        ###########################################
        for net_layer, target_layer in zip(self.net.layers, self.target_net.layers):
            target_layer.set_weights(net_layer.get_weights())
        ###########################################
        """


    # @abstractmethod
    # def save_weights(self, path):
    #     """ Save the neural network weights in h5 if a keras model is used or as tensorflow checkpoints if other
    #     tensorflow modules are used.
    #     :param path: (str) path to save the neural network
    #     """
    #
    # @abstractmethod
    # def load_weights(self, path):
    #     """ Load the neural network weights from an h5 if a keras model is used or from a tensorflow checkpoints if
    #     other tensorflow modules are used.
    #     :param path: (str) path to load from the neural network
    #     """

class RLNetModel(RLNetInterfaz):
    """
    A class for defining your own computation graph without worrying about the reinforcement learning procedure. Her
    e you are able to completely configure your neural network, optimizer, loss function, metrics and almost do anything
    else tensorflow allows. You can even use tensorboard to monitor the behaviour of the computation graph further than
    the metrics recorded by this library.
    """

    def __init__(self, tensorboard_dir):
        super().__init__()
        self.optimizer = None   # Optimization algorithm form tensorflow or keras
        self.loss_func = None
        self.metrics = None
        self._tensorboard_util(tensorboard_dir)
        self.loss_sumaries = tensor_board_loss_functions.loss_sumaries
        self.rl_loss_sumaries = tensor_board_loss_functions.rl_loss_sumaries
        self.rl_sumaries = tensor_board_loss_functions.rl_sumaries

        self.total_epochs = 0

    def _tensorboard_util(self, tensorboard_dir):
        if tensorboard_dir is not None:
            current_time = datetime.datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            self.train_log_dir = os.path.join(tensorboard_dir, current_time)
            self.train_summary_writer = tf.summary.create_file_writer(self.train_log_dir)
        else:
            self.train_log_dir = None
            self.train_summary_writer = None

    def process_globals(self, custom_globals):
        globs = globals()
        for key in globs:
            for cust_key in custom_globals:
                if key == cust_key:
                    custom_globals[cust_key] = globs[key]
                    break
        return custom_globals

    def bc_fit(self, x, y, epochs, batch_size,  validation_split=0., shuffle=True, verbose=1, callbacks=None,
               kargs=[]):

        if validation_split > 0.:
            # Validation split for expert traj
            n_val_split = int(x.shape[0] * validation_split)
            val_idx = np.random.choice(x.shape[0], n_val_split, replace=False)
            train_mask = np.array([False if i in val_idx else True for i in
                                   range(x.shape[0])])

            x_val = x[val_idx]
            y_val = y[val_idx]

            x_train = np.array(x[train_mask])
            y_train = np.array(y[train_mask])
        else:
            x_train = np.array(x)
            y_train = np.array(y)


        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        if validation_split > 0.:
            val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))

        if shuffle:
            dataset = dataset.shuffle(x.shape[0], reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)
        if validation_split > 0.:
            val_dataset = val_dataset.batch(batch_size)

        history = TrainingHistory()

        start_time = time.time()

        for e in range(epochs):
            loss_mean = []
            metrics_mean = []
            for batch, (x, y) in enumerate(dataset.take(-1)):
                loss, gradients, variables = self.bc_train_step(x, y)
                loss_mean.append(loss)
                metric = self.bc_metrics.result()
                metrics_mean.append(metric)

                if batch_size > 5:
                    show_each = int(int((x_train.shape[0]/4))/batch_size)
                else:
                    show_each = 1
                if batch % show_each == 0 and verbose == 1:
                    print(('Epoch {}\t Batch {}\t Loss  {:.4f} ' + self.bc_metrics.name +
                           ' {:.4f} Elapsed time {:.2f}s').format(e + 1,
                                                                  batch,
                                                                  loss.numpy(),
                                                                  metric,
                                                                  time.time() - start_time))
                    start_time = time.time()
            loss_mean = np.mean(loss_mean)
            metrics_mean = np.mean(metrics_mean)

            if validation_split > 0.:
                val_loss = []
                val_metrics = []
                for batch, (x, y) in enumerate(val_dataset.take(-1)):
                    val_loss.append(self.bc_evaluate(x, y))
                    val_metrics.append(self.bc_metrics.result())
                mean_val_loss = np.mean(val_loss)
                val_metrics_mean = np.mean(val_metrics)

            if verbose >= 1:
                if validation_split > 0.:
                    print(('epoch {}\t loss  {:.4f} ' + self.bc_metrics.name + ' {:.4f}' +
                           ' val_loss  {:.4f} val_' + self.bc_metrics.name + ' {:.4f}').format(e + 1,
                                                                                            loss_mean,
                                                                                            metrics_mean,
                                                                                            mean_val_loss,
                                                                                            val_metrics_mean))
                else:
                    print(('Epoch {}\t Loss  {:.4f} ' + self.bc_metrics.name + ' {:.4f}').format(e + 1,
                                                                                              loss_mean,
                                                                                              metrics_mean))
            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    if validation_split > 0.:
                        self.loss_sumaries([loss_mean, metrics_mean,
                                            mean_val_loss, val_metrics_mean],
                                           ['discriminator_loss', self.bc_metrics.name,
                                            'discriminator_val_loss',
                                            'val_' + self.bc_metrics.name],
                                           self.total_epochs)
                    else:
                        self.loss_sumaries([loss_mean, metrics_mean],
                                           ['discriminator_loss', self.metrics.name],
                                           self.total_epochs)
        self.total_epochs += 1

        history.history['loss'].append(loss_mean)
        history.history['acc'].append(metrics_mean)

        if validation_split > 0.:
            history.history['val_loss'].append(mean_val_loss)
            history.history['val_acc'].append(val_metrics_mean)

        if callbacks is not None:
            for cb in callbacks:
                cb.on_epoch_end(e)


        return history

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

        return self._bc_train_step(x, y)

    @tf.function(experimental_relax_shapes=True)
    def _bc_train_step(self, x, y):
       pass

    def bc_evaluate(self, x, y):
        """
        Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        return self._bc_evaluate(x, y)

    def bc_compile(self, loss, optimizer, metrics=None):
        """
        Compile the neural network. Usually used for define loss function, optimizer and metrics.
        :param loss: Loss function from tensorflow, keras or user defined.
        :param optimizer:  Optimizer from tensorflow, keras or user defined.
        :param metrics: Metrics from tensorflow, keras or user defined.
        """
        self.bc_loss_func = loss[0]
        self.bc_optimizer = optimizer[0]
        self.bc_metrics = metrics

    @tf.function(experimental_relax_shapes=True)
    def _bc_evaluate(self, x, y):
        pass

class TrainingHistory():
    def __init__(self):
        self.history = {'loss': [],
                        'acc': [],
                        'val_loss': [],
                        'val_acc': []}

class RLSequentialModel(object):
    """
        encoder_size: encoder d_model in the paper (depth size of the model)
        encoder_n_layers: encoder number of layers (Multi-Head Attention + FNN)
        encoder_h: encoder number of attention heads
    """

    def __init__(self, sequential_net, tensorboard_dir=None):
        self.net = sequential_net
        if tensorboard_dir is not None:
            current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            train_log_dir = os.path.join(tensorboard_dir, 'logs/gradient_tape/' + current_time + '/train')
            test_log_dir = os.path.join(tensorboard_dir, 'logs/gradient_tape/' + current_time + '/test')
            self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            # self.test_summary_writer = tf.summary.create_file_writer(test_log_dir)
        else:
            self.train_summary_writer = None
        self.total_epochs = 0
        self.loss_func = None
        self.optimizer = None
        self.metrics = None

    def add(self, layer):
        self.net = tf.keras.models.Sequential([self.net, layer])
        # self.net.add(layer)

    def compile(self, loss, optimizer, metrics=None):
        self.loss_func = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def summary(self):
        pass

    def predict(self, x):
        y_ = self._predict(x)
        return y_.numpy()

    @tf.function
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
        # if test_source_text is None:
        #     test_source_text = self.raw_data_en[np.random.choice(len(raw_data_en))]
        y_ = self.net(tf.cast(x, tf.float32), training=False)
        return y_

    def evaluate(self, x, y, batch_size=32, verbose=0):
        dataset = tf.data.Dataset.from_tensor_slices((x, y))

        dataset = dataset.shuffle(len(x), reshuffle_each_iteration=True).batch(batch_size)

        loss = 0.
        acc = 0.
        for batch, (x, y) in enumerate(dataset.take(-1)):
            l = self.validate_step(x, y)
            loss += l
            acc += self.metrics.result()
        return loss / (batch + 1), acc / (batch + 1)

    @tf.function
    def validate_step(self, x, y):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        y_ = self.net(x, training=False)
        loss = self.loss_func(y, y_)
        self.metrics.update_state(y, y_)
        return loss

    @tf.function
    def train_step(self, x, y):
        """ Execute one training step (forward pass + backward pass)
        Args:
            source_seq: source sequences
            target_seq_in: input target sequences (<start> + ...)
            target_seq_out: output target sequences (... + <end>)

        Returns:
            The loss value of the current pass
        """
        with tf.GradientTape() as tape:
            y_ = self.net(x, training=True)
            loss = self.loss_func(y, y_)
        self.metrics.update_state(y, y_)

        variables = self.net.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))

        return loss

    def fit(self, x, y, epochs, batch_size=64, validation_split=0.15, shuffle=True, verbose=1, callbacks=None):

        if validation_split > 0.0:
            validation_split = int(x.shape[0] * validation_split)
            val_idx = np.random.choice(x.shape[0], validation_split, replace=False)
            train_mask = np.array([False if i in val_idx else True for i in range(x.shape[0])])

            test_samples = np.int(val_idx.shape[0])
            train_samples = np.int(train_mask.shape[0] - test_samples)

            val_input_data = np.float32(x[val_idx])
            val_target_data = y[val_idx]

            train_input_data = np.float32(x[train_mask])
            train_target_data = y[train_mask]

        else:
            train_input_data = tf.float32(x)
            train_target_data = y

        dataset = tf.data.Dataset.from_tensor_slices((train_input_data, train_target_data))

        if shuffle:
            dataset = dataset.shuffle(len(train_input_data), reshuffle_each_iteration=True).batch(batch_size)
        else:
            dataset = dataset.batch(batch_size)

        history = TrainingHistory()

        starttime = time.time()
        for e in range(epochs):
            for batch, (batch_train_input_data, batch_train_target_data) in enumerate(dataset.take(-1)):
                loss = self.train_step(batch_train_input_data, batch_train_target_data)
                if batch % int(batch_size / 5) == 0 and verbose == 1:
                    print('Epoch {}\t Batch {}\t Loss {:.4f} Acc {:.4f} Elapsed time {:.2f}s'.format(
                        e + 1, batch, loss.numpy(), self.metrics.result(), time.time() - starttime))
                    starttime = time.time()
            if self.train_summary_writer is not None:
                with self.train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=self.total_epochs)
                    tf.summary.scalar('accuracy', self.metrics.result(), step=self.total_epochs)
                    self.extract_variable_summaries(self.net, self.total_epochs)
            self.total_epochs += 1

            history.history['loss'].append(loss)
            try:
                if validation_split > 0.0 and verbose == 1:
                    val_loss = self.evaluate(val_input_data, val_target_data, batch_size)
                    history.history['val_loss'].append(val_loss[0])
                    # with self.test_summary_writer.as_default():
                    #     tf.summary.scalar('loss', val_loss[0], step=self.total_epochs)
                    #     tf.summary.scalar('accuracy', val_loss[1], step=self.total_epochs)
                    print('Epoch {}\t val_loss {:.4f}, val_acc {:.4f}'.format(
                        e + 1, val_loss[0].numpy(), val_loss[1].numpy()))
            except Exception as e:
                print(e)
                continue

            for cb in callbacks:
                cb.on_epoch_end(e)
        return history

    def extract_variable_summaries(self, net, epoch):
        # Set all the required tensorboard summaries
        pass

    def variable_summaries(self, name, var, e):
        """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
        with tf.name_scope(str(name)):
            with tf.name_scope('summaries'):
                histog_summary = tf.summary.histogram('histogram', var, step=e)