from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Flatten
import tensorflow as tf
from termcolor import colored
import numpy as np

def read_net_params(net_architecture, actor=False, critic=False, action=False, value=False, common=False):
    id_1 = 'conv_layers'
    id_2 = 'kernel_num'
    id_3 = 'kernel_size'
    id_4 = 'kernel_strides'
    id_5 = 'conv_activation'
    id_6 = 'dense_lay'
    id_7 = 'n_neurons'
    id_8 = 'dense_activation'
    id_9 = 'use_custom_network'
    id_10 = 'custom_network'
    id_11 = 'define_custom_output_layer'
    id_12 = 'tf_custom_model'
    id_13 = 'use_tf_custom_model'

    if actor or critic:
        if actor:
            prefix = 'actor_'
        elif critic:
            prefix = 'critic_'

        id_1 = prefix + id_1
        id_2 = prefix + id_2
        id_3 = prefix + id_3
        id_4 = prefix + id_4
        id_5 = prefix + id_5
        id_6 = prefix + id_6
        id_7 = prefix + id_7
        id_8 = prefix + id_8
        # id_9 is unique
        id_10 = prefix + id_10
        # id_11 is unique
        # id_12 is unique
        # id_13 is unique

    if action or value or common:
        if action:
            prefix = 'action_'
        elif value:
            prefix = 'value_'
        elif common:
            prefix = 'common_'

        id_1 = prefix + id_1
        id_2 = prefix + id_2
        id_3 = prefix + id_3
        id_4 = prefix + id_4
        id_5 = prefix + id_5
        id_6 = prefix + id_6
        id_7 = prefix + id_7
        id_8 = prefix + id_8
        # id_9 is unique
        id_10 = prefix + id_10
        # id_11 is unique
        # id_12 is unique
        # id_13 is unique

    # TODO: configurar correctamente los warnings
    if (id_1 and id_2 and id_3 and id_4 and id_5 in net_architecture) \
            and (net_architecture[id_1] and net_architecture[id_2] and net_architecture[id_3] and net_architecture[id_4]
                 and net_architecture[id_5] is not None):

        n_conv_layers = net_architecture[id_1]
        kernel_num = net_architecture[id_2]
        kernel_size = net_architecture[id_3]
        strides = net_architecture[id_4]
        conv_activation = net_architecture[id_5]
        # print('Convolutional layers selected: {conv_layers:\t\t', n_conv_layers,
        #       '\n\t\t\t\t\t\t\t    kernel_num:\t\t\t', kernel_num, '\n\t\t\t\t\t\t\t    kernel_size:\t\t', kernel_size,
        #       '\n\t\t\t\t\t\t\t    kernel_strides:\t\t', strides, '\n\t\t\t\t\t\t\t    conv_activation:\t',
        #       conv_activation, '}')
    else:
        n_conv_layers = None
        kernel_num = None
        kernel_size = None
        strides = None
        conv_activation = None
        # print(colored('WARNING: If you want to specify convolutional layers you must set all the values for the '
        #               'following keys: conv_layers, kernel_num, kernel_size, kernel_strides and conv_activation',
        #               'yellow'))

    if (id_6 and id_7 and id_8 in net_architecture) and (net_architecture[id_6] and net_architecture[id_7]
                                                         and net_architecture[id_8] is not None):
        n_dense_layers = net_architecture[id_6]
        n_neurons = net_architecture[id_7]
        dense_activation = net_architecture[id_8]
        # print('Dense layers selected: {dense_lay:\t\t\t', n_dense_layers, '\n\t\t\t\t\t    n_neurons:\t\t\t', n_neurons,
        #       '\n\t\t\t\t\t    dense_activation:\t', dense_activation, '}')
    else:
        n_dense_layers = None
        n_neurons = None
        dense_activation = None
        # print(colored('WARNING: If you want to specify dense layers you must set all the values for the following keys:'
        #               ' dense_lay, n_neurons and dense_activation', 'yellow'))

    if (id_9 and id_10 in net_architecture) and (net_architecture[id_9] and net_architecture[id_10] is not None):
        use_custom_net = net_architecture[id_9]
        custom_net = net_architecture[id_10]
        define_custom_output_layer = net_architecture[id_11]
        # print('Custom network option selected: {use_custom_network: ', use_custom_net, ', custom_network: ',
        #       custom_net, '}')
    else:
        use_custom_net = False
        custom_net = None
        define_custom_output_layer = None
        # print(colored('WARNING: If you want to use a custom neural net you must set the values for all the following '
        #               'keys: '
        #               'use_custom_network, custom_network', 'yellow'))

    if (id_12 and id_13 in net_architecture) and (net_architecture[id_12] and net_architecture[id_13] is not None):
        use_custom_net = net_architecture[id_13]
        custom_net = net_architecture[id_12]

    return n_conv_layers, kernel_num, kernel_size, strides, conv_activation, n_dense_layers, n_neurons, \
           dense_activation, use_custom_net, custom_net, define_custom_output_layer

def build_conv_net(net_architecture, input_shape, actor=False, critic=False, common=False):
    n_conv_layers, kernel_num, kernel_size, strides, conv_activation, \
    n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net, define_custom_output_layer = read_net_params(net_architecture, actor, critic, common=common)

    if use_custom_net:
        return custom_net(input_shape)
    else:
        model = Sequential()
        model.add(Conv2D(kernel_num[0], kernel_size=kernel_size[0], input_shape=input_shape,
                         strides=(strides[0], strides[0]), padding='same',
                         activation=conv_activation[0]))
        for i in range(1, n_conv_layers):
            model.add(Conv2D(kernel_num[i], kernel_size=kernel_size[i], strides=(strides[i], strides[i]),
                             padding='same', activation=conv_activation[i]))

        model.add(Flatten())

        if n_dense_layers is not None:
            for i in range(n_dense_layers):
                model.add(Dense(n_neurons[i], activation=dense_activation[i]))

        return model


def build_stack_net(net_architecture, input_shape, actor=False, critic=False, common=False):
    _, _, _, _, _, n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net, define_custom_output_layer = read_net_params(net_architecture, actor, critic, common=common)

    if use_custom_net:
        return custom_net(input_shape)
    else:
        model = Sequential()
        model.add(Flatten(input_shape=input_shape))
        if n_dense_layers is not None:
            for i in range(n_dense_layers):
                model.add(Dense(n_neurons[i], input_dim=input_shape, activation=dense_activation[i]))

        return model


def build_nn_net(net_architecture, input_shape, actor=False, critic=False, action=False, value=False, common=False):
    _, _, _, _, _, n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net, define_custom_output_layer = read_net_params(net_architecture, actor, critic, action=action, value=value, common=common)

    if use_custom_net:
        return custom_net((input_shape,))
    else:
        model = Sequential()
        model.add(Dense(n_neurons[0], input_dim=input_shape, activation=dense_activation[0]))

        for i in range(1, n_dense_layers):
            model.add(Dense(n_neurons[i], activation=dense_activation[i]))
        return model


def build_ddpg_conv_critic(net_architecture, input_shape, s, a):
    n_conv_layers, kernel_num, kernel_size, strides, conv_activation, \
    n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net, define_custom_output_layer = read_net_params(net_architecture, actor=False, critic=True)

    if use_custom_net:
        return custom_net(input_shape, s, a)
    else:
        conv_obs = Conv2D(kernel_num[0], kernel_size=kernel_size[0], input_shape=input_shape,
                          strides=(strides[0], strides[0]), padding='same',
                          activation=conv_activation[0])(s)
        for i in range(1, n_conv_layers):
            conv_obs = Conv2D(kernel_num[i], kernel_size=kernel_size[i], strides=(strides[i], strides[i]),
                              padding='same', activation=conv_activation[i])(conv_obs)

        flat = tf.keras.layers.Flatten()(conv_obs)
        bias = len(n_neurons) > 2

        lay_obs = Dense(n_neurons[0], activation=dense_activation[0], use_bias=bias)(flat)
        for i in range(1, n_dense_layers - 1):
            lay_obs = Dense(n_neurons[i], activation=dense_activation[i], use_bias=False)(lay_obs)
        lay_obs = Dense(n_neurons[-1], activation='linear', use_bias=False)(lay_obs)

        lay_act = Dense(n_neurons[0], input_dim=input_shape, activation=dense_activation[0], use_bias=bias)(a)
        for i in range(1, n_dense_layers - 1):
            lay_act = Dense(n_neurons[i], activation=dense_activation[i], use_bias=False)(lay_act)
        lay_act = Dense(n_neurons[-1], activation='linear', use_bias=False)(lay_act)

        merge = tf.keras.layers.Add()([lay_obs, lay_act])
        b_init = tf.constant_initializer(0.1)
        b = tf.get_variable(name='bias', shape=[n_neurons[-1]], initializer=b_init)
        output = tf.nn.relu(merge + b)

        return output


def build_ddpg_stack_critic(net_architecture, input_shape, s, a):
    _, _, _, _, _, n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net, define_custom_output_layer = read_net_params(net_architecture, actor=False, critic=True)


    if use_custom_net:
        return custom_net(input_shape, s, a)
    else:
        flat = tf.keras.layers.Flatten(input_shape=input_shape)(s)
        bias = len(n_neurons) > 2

        lay_obs = Dense(n_neurons[0], activation=dense_activation[0], use_bias=bias)(flat)
        for i in range(1, n_dense_layers - 1):
            lay_obs = Dense(n_neurons[i], activation=dense_activation[i], use_bias=False)(lay_obs)
        lay_obs = Dense(n_neurons[-1], activation='linear', use_bias=False)(lay_obs)

        lay_act = Dense(n_neurons[0], input_dim=input_shape, activation=dense_activation[0], use_bias=bias)(a)
        for i in range(1, n_dense_layers - 1):
            lay_act = Dense(n_neurons[i], activation=dense_activation[i], use_bias=False)(lay_act)
        lay_act = Dense(n_neurons[-1], activation='linear', use_bias=False)(lay_act)

        merge = tf.keras.layers.Add()([lay_obs, lay_act])
        b_init = tf.constant_initializer(0.1)
        b = tf.get_variable(name='bias', shape=[n_neurons[-1]], initializer=b_init)
        output = tf.nn.relu(merge + b)

        return output


def build_ddpg_nn_critic(net_architecture, input_shape, s, a):
    _, _, _, _, _, n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net, define_custom_output_layer = read_net_params(net_architecture, actor=False, critic=True)

    if use_custom_net:
        return custom_net(input_shape, s, a)
    else:
        bias = len(n_neurons) > 2

        lay_obs = Dense(n_neurons[0], input_dim=input_shape, activation=dense_activation[0], use_bias=bias)(s)
        for i in range(1, n_dense_layers - 1):
            lay_obs = Dense(n_neurons[i], activation=dense_activation[i], use_bias=False)(lay_obs)
        lay_obs = Dense(n_neurons[-1], activation='linear', use_bias=False)(lay_obs)

        lay_act = Dense(n_neurons[0], input_dim=input_shape, activation=dense_activation[0], use_bias=bias)(a)
        for i in range(1, n_dense_layers - 1):
            lay_act = Dense(n_neurons[i], activation=dense_activation[i], use_bias=False)(lay_act)
        lay_act = Dense(n_neurons[-1], activation='linear', use_bias=False)(lay_act)

        merge = tf.keras.layers.Add()([lay_obs, lay_act])
        b_init = tf.constant_initializer(0.1)
        b = tf.get_variable(name='bias', shape=[n_neurons[-1]], initializer=b_init)
        output = tf.nn.relu(merge + b)

    return output


def build_ddpg_conv_critic_tf(net_architecture, input_shape, actor_net):
    n_conv_layers, kernel_num, kernel_size, strides, conv_activation, \
    n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net, define_custom_output_layer = read_net_params(net_architecture, actor=False, critic=True)

    if use_custom_net:
        try:
            return custom_net(input_shape, actor_net)
        except:
            from termcolor import colored
            print(colored('Custom critic network for DDPG requires two arguments as input: input_shape and actor_net. '
                          'Use a function to create the network with the head function_name(input_shape, actor_net)',
                          'red'))
    else:
        model_obs = Sequential()
        model_obs.add(Conv2D(kernel_num[0], kernel_size=kernel_size[0], input_shape=input_shape,
                          strides=(strides[0], strides[0]), padding='same',
                          activation=conv_activation[0]))
        for i in range(1, n_conv_layers):
            model_obs.add(Conv2D(kernel_num[i], kernel_size=kernel_size[i],
                                 strides=(strides[i], strides[i]), padding='same',
                                 activation=conv_activation[i]))

        model_obs.add(Flatten(input_shape=input_shape))
        if n_dense_layers is not None:
            for i in range(n_dense_layers - 1):
                model_obs.add(Dense(n_neurons[i], input_dim=input_shape, activation=dense_activation[i]))

        model_act = Sequential()
        if n_dense_layers is not None:
            for i in range(n_dense_layers - 1):
                actor_shape = (np.array(actor_net.output.shape.as_list())[1])
                model_act.add(Dense(n_neurons[i], input_dim=actor_shape, activation=dense_activation[i]))

        merge = tf.keras.layers.Concatenate()([model_obs.output, model_act.output])
        output = Dense(n_neurons[-1], input_dim=input_shape, activation=dense_activation[-1])(merge)
        model = Model(inputs=[model_obs.input, model_act.input], outputs=output)
        return model

def build_ddpg_stack_critic_tf(net_architecture, input_shape, actor_net):
    _, _, _, _, _, n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net, define_custom_output_layer = read_net_params(net_architecture, actor=False, critic=True)

    if use_custom_net:
        try:
            return custom_net(input_shape, actor_net)
        except:
            from termcolor import colored
            print(colored('Custom critic network for DDPG requires two arguments as input: input_shape and actor_net. '
                          'Use a function to create the network with the head function_name(input_shape, actor_net)',
                          'red'))
    else:
        model_obs = Sequential()
        model_obs.add(Flatten(input_shape=input_shape))
        if n_dense_layers is not None:
            for i in range(n_dense_layers-1):
                model_obs.add(Dense(n_neurons[i], input_dim=input_shape, activation=dense_activation[i]))

        model_act = Sequential()
        if n_dense_layers is not None:
            for i in range(n_dense_layers-1):
                actor_shape = (np.array(actor_net.output.shape.as_list())[1])
                model_act.add(Dense(n_neurons[i], input_dim=actor_shape, activation=dense_activation[i]))

        merge = tf.keras.layers.Concatenate()([model_obs.output, model_act.output])
        output = Dense(n_neurons[-1], input_dim=input_shape, activation=dense_activation[-1])(merge)
        # model = Model Sequential([output])
        model = Model(inputs=[model_obs.input, model_act.input], outputs=output)

        return model


def build_ddpg_nn_critic_tf(net_architecture, input_shape, actor_net, n_actions):
    _, _, _, _, _, n_dense_layers, n_neurons, dense_activation, use_custom_net, \
    custom_net, define_custom_output_layer = read_net_params(net_architecture, actor=False, critic=True)

    if use_custom_net:
        try:
            return custom_net(input_shape, n_actions)
        except:
            from termcolor import colored
            print(colored('Custom critic network for DDPG requires two arguments as input: input_shape and actor_net. '
                          'Use a function to create the network with the head function_name(input_shape, actor_net)',
                          'red'))
    else:
        model_obs = Sequential()
        if n_dense_layers is not None:
            for i in range(n_dense_layers - 1):
                model_obs.add(Dense(n_neurons[i], input_dim=input_shape, activation=dense_activation[i]))

        model_act = Sequential()
        if n_dense_layers is not None:
            for i in range(n_dense_layers - 1):
                actor_shape = (np.array(actor_net.output.shape.as_list())[1])
                model_act.add(Dense(n_neurons[i], input_dim=actor_shape, activation=dense_activation[i]))

        merge = tf.keras.layers.Concatenate()([model_obs.output, model_act.output])
        output = Dense(n_neurons[-1], input_dim=input_shape, activation=dense_activation[-1])(merge)
        model = Model(inputs=[model_obs.input, model_act.input], outputs=output)
        return model

