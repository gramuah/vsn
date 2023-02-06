from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, BatchNormalization, Activation, Conv1D, Permute, LSTM
# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.constraints import MaxNorm


import environments.carla.vae_1 as vae
import os
# encoder, decoder, _vae = vae.load_model(os.path.abspath('saved_models/Carracing_vae/beta2_encoder_32_64_128_128'),
#                  os.path.abspath('saved_models/Carracing_vae/beta2_decoder_32_64_128_128'), (96, 96, 3))

########################################################################################################################
#                                                 DQN model
########################################################################################################################

def dqn_model(input_shape):
    dqn_model = Sequential()

    # dqn_model.add(Flatten(input_shape=input_shape))

    dqn_model.add(Dense(128, input_dim=input_shape, activation='relu'))
    dqn_model.add(Dropout(0.2))
    dqn_model.add(Dense(128, activation='relu', kernel_constraint=MaxNorm(3)))
    dqn_model.add(Dropout(0.2))

    # dqn_model.add(Dense(128, input_dim=input_shape, activation='linear', use_bias=False))
    # dqn_model.add(BatchNormalization())
    # dqn_model.add(Activation("relu"))
    #
    # # dqn_model.add(Dropout(0.2))
    # dqn_model.add(Dense(128, input_dim=input_shape, activation='linear', use_bias=False))
    # dqn_model.add(BatchNormalization())
    # dqn_model.add(Activation("relu"))
    # # dqn_model.add(Dropout(0.2))

    return dqn_model

########################################################################################################################
#                                                 DDQN model
########################################################################################################################

def ddqn_model(input_shape):
    ddqn_model = Sequential()

    ddqn_model.add(Conv2D(32, kernel_size=3, input_shape=input_shape, strides=1, padding='same', activation='relu'))
    ddqn_model.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))
    ddqn_model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    ddqn_model.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))

    ddqn_model.add(Flatten())

    ddqn_model.add(Dense(128, activation='relu'))
    ddqn_model.add(Dense(128, activation='relu'))
    return ddqn_model


########################################################################################################################
#                                                 DDDQN model
########################################################################################################################
def dddqn_model(input_shape):
    dddqn_model_head = Sequential()

    dddqn_model_head.add(Conv2D(32, kernel_size=3, input_shape=input_shape, strides=1, padding='same', activation='relu'))
    dddqn_model_head.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))
    dddqn_model_head.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    dddqn_model_head.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))

    dddqn_model_head.add(Flatten())

    dddqn_value = Dense(64, activation='relu', name="dense_valor_in")(dddqn_model_head.output)
    dddqn_advantage = Dense(64, activation='relu', name="dense_advantage_in")(dddqn_model_head.output)
    dddqn_value = Dense(64, activation='relu', name="dense_valor_0")(dddqn_value)
    dddqn_advantage = Dense(64, activation='relu', name="dense_advantage_0")(dddqn_advantage)

    return dddqn_model_head, dddqn_value, dddqn_advantage


########################################################################################################################
#                                                 DPG model
########################################################################################################################
def dpg_model(input_shape):
    dpg_model = Sequential()

    dpg_model.add(Conv2D(32, kernel_size=3, input_shape=input_shape, strides=1, padding='same', activation='relu'))
    dpg_model.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))
    dpg_model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    dpg_model.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))

    dpg_model.add(Flatten())

    dpg_model.add(Dense(128, activation='relu'))
    dpg_model.add(Dense(128, activation='relu'))
    return dpg_model

########################################################################################################################
#                                                 Actor-Critic model
########################################################################################################################
def actor_model(input_shape):
    actor_model = Sequential()

    # actor_model.add(Conv2D(32, kernel_size=3, input_shape=input_shape, strides=1, padding='same', activation='relu'))
    # actor_model.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))
    # actor_model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    # actor_model.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))

    # actor_model.add(Flatten(input_shape=input_shape))
    actor_model.add(Dropout(0.2))
    actor_model.add(Dense(512, activation='relu'))
    actor_model.add(Dropout(0.4))
    actor_model.add(Dense(512, activation='relu'))
    actor_model.add(Dense(256, activation='relu'))

    return actor_model

def critic_model(input_shape):
    critic_model = Sequential()

    # critic_model.add(Conv2D(32, kernel_size=3, input_shape=input_shape, strides=1, padding='same', activation='relu'))
    # critic_model.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))
    # critic_model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    # critic_model.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))

    # critic_model.add(Flatten(input_shape=input_shape))
    critic_model.add(Dropout(0.2))
    critic_model.add(Dense(512, activation='relu'))
    critic_model.add(Dropout(0.5))
    critic_model.add(Dense(512, activation='relu'))
    critic_model.add(Dense(256, activation='relu'))

    return critic_model



########################################################################################################################
#                                                 DDPG model
########################################################################################################################

def actor_ddpg(input_shape):
    actor_model = Sequential()

    # actor_model.add(Conv2D(32, kernel_size=3, input_shape=input_shape, strides=1, padding='same', activation='relu'))
    # actor_model.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))
    # actor_model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    # actor_model.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))

    # actor_model.add(Flatten(input_shape=input_shape))

    actor_model.add(Dense(32, input_dim=input_shape, activation='relu'))
    actor_model.add(Dense(32, activation='relu'))

    return actor_model

def critic_ddpg(input_shape, s, a):
    # flat = tf.keras.layers.Flatten(input_shape=input_shape)(s)

    lay_obs = Dense(32, input_dim=input_shape, activation='relu')(s)
    # lay_obs = Dense(512, activation='linear', use_bias=False)(lay_obs)
    lay_obs = Dense(32, activation='linear')(lay_obs)

    # lay_act = Dense(64, input_dim=input_shape, activation='relu')(a)
    # lay_act = Dense(512, activation='linear', use_bias=False)(a)
    lay_act = Dense(32, activation='linear')(a)

    merge = tf.keras.layers.Add()([lay_obs, lay_act])
    b_init = tf.constant_initializer(0.1)
    b = tf.get_variable(name='bias', shape=[32], initializer=b_init)
    output = tf.nn.relu(merge + b)
    output = Dense(64, activation='relu')(output)

    return output

# def critic_ddpg(input_shape, s, a):
#     flat = tf.keras.layers.Flatten(input_shape=input_shape)(s)
#
#     lay_obs = Dense(256, activation='relu')(flat)
#     # lay_obs = Dense(512, activation='linear', use_bias=False)(lay_obs)
#     lay_obs = Dense(512, activation='linear')(lay_obs)
#
#     # lay_act = Dense(64, input_dim=input_shape, activation='relu')(a)
#     # lay_act = Dense(512, activation='linear', use_bias=False)(a)
#     lay_act = Dense(512, activation='linear')(a)
#
#     merge = tf.keras.layers.concatenate([lay_obs, lay_act])
#     # b_init = tf.constant_initializer(0.1)
#     # b = tf.get_variable(name='bias', shape=[512], initializer=b_init)
#     # output = tf.nn.relu(merge + b)
#     output = Dense(512, activation='relu')(merge)
#
#     return output


# def critic_ddpg(input_shape, s, a):
#     flat = tf.keras.layers.Flatten(input_shape=input_shape)(s)
#
#     merge = tf.keras.layers.concatenate([flat, a])
#     # b_init = tf.constant_initializer(0.1)
#     # b = tf.get_variable(name='bias', shape=[512], initializer=b_init)
#     # output = tf.nn.relu(merge + b)
#     output = Dense(512, activation='relu')(merge)
#     output = Dense(256, activation='relu')(output)
#
#     return output

def actor_model_drop(input_shape):
    actor_model = Sequential()

    # actor_model.add(Conv2D(32, kernel_size=3, input_shape=input_shape, strides=1, padding='same', activation='relu'))
    # actor_model.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))
    # actor_model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    # actor_model.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))
    # actor_model.add(Permute((2, 1), input_shape=input_shape))
    actor_model.add(Conv1D(64, kernel_size=3, strides=2, input_shape=input_shape, padding='same', activation='relu'))
    actor_model.add(Dropout(0.3))
    actor_model.add(Conv1D(64, kernel_size=3, strides=1, input_shape=input_shape, padding='same', activation='relu'))
    actor_model.add(Dropout(0.3))
    actor_model.add(Flatten(input_shape=input_shape))
    # actor_model.add(Dense(256, activation='tanh'))
    actor_model.add(Dense(2048, activation='relu'))
    actor_model.add(Dropout(0.4))
    actor_model.add(Dense(2048, activation='relu'))
    actor_model.add(Dropout(0.4))
    actor_model.add(Dense(128, activation='relu'))
    # actor_model.add(Dropout(0.2))

    return actor_model

def critic_model_drop(input_shape):
    critic_model = Sequential()

    # critic_model.add(Conv2D(32, kernel_size=3, input_shape=input_shape, strides=1, padding='same', activation='relu'))
    # critic_model.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))
    # critic_model.add(Conv2D(32, kernel_size=3, strides=2, padding='same', activation='relu'))
    # critic_model.add(MaxPooling2D(pool_size=(2, 2), paddimg='same'))
    # critic_model.add(Permute((2, 1), input_shape=input_shape))
    critic_model.add(Conv1D(64, kernel_size=3, strides=1, input_shape=input_shape, padding='same', activation='relu'))
    # critic_model.add(Dropout(0.3))
    # critic_model.add(Conv1D(64, kernel_size=3, strides=2, padding='same', activation='relu'))
    # critic_model.add(Dropout(0.3))
    critic_model.add(Flatten(input_shape=input_shape))
    critic_model.add(Dense(2048, activation='relu'))
    # critic_model.add(Dense(2048, activation='tanh'))
    critic_model.add(Dropout(0.4))
    critic_model.add(Dense(1024, activation='relu'))
    critic_model.add(Dropout(0.4))
    critic_model.add(Dense(128, activation='relu'))
    critic_model.add(Dropout(0.2))

    return critic_model

def actor_model_lstm(input_shape):
    actor_model = Sequential()
    actor_model.add(LSTM(512, input_shape=input_shape, activation='tanh'))
    actor_model.add(Dense(512, activation='tanh'))
    # actor_model.add(Dropout(0.4))
    # actor_model.add(Flatten(input_shape=input_shape))
    # actor_model.add(Dense(1024, activation='tanh', input_shape=input_shape))
    # actor_model.add(Dropout(0.3))
    actor_model.add(Dense(128, activation='tanh'))

    return actor_model


def critic_model_lstm(input_shape):
    critic_model = Sequential()
    critic_model.add(LSTM(512, input_shape=input_shape, activation='tanh'))
    critic_model.add(Dense(512, activation='tanh'))
    # critic_model.add(Dropout(0.4))
    # critic_model.add(Flatten(input_shape=input_shape))
    # critic_model.add(Dense(1024, activation='tanh', input_shape=input_shape))
    # critic_model.add(Dropout(0.3))
    critic_model.add(Dense(128, activation='tanh'))

    return critic_model






