# DQN network input and hidden layers architecture. For dense only discriminator, convolutional parameters will be not readed.
dqn_net = {"conv_layers": 2,
           "kernel_num": [16, 32],
           "kernel_size": [7, 3],
           "kernel_strides": [4, 2],
           "conv_activation": ['relu', 'relu'],
           "dense_lay": 2,
           "n_neurons": [256, 256],
           "dense_activation": ['relu', 'relu'],
           }

# DDQN network input and hidden layers architecture. For dense only discriminator, convolutional parameters will be not readed.
ddqn_net = {"conv_layers": 2,
            "kernel_num": [16, 32],
            "kernel_size": [7, 3],
            "kernel_strides": [4, 2],
            "conv_activation": ['relu', 'relu'],
            "dense_lay": 2,
            "n_neurons": [256, 256],
            "dense_activation": ['relu', 'relu'],
            }

# DDDQN network input and hidden layers architecture. For this algorithm we will consider the same architecture for both
# dense layer flows. For dense only discriminator, convolutional parameters will be not readed.
dddqn_net = {"common_conv_layers": 2,
             "common_kernel_num": [16, 32],
             "common_kernel_size": [7, 3],
             "common_kernel_strides": [4, 2],
             "common_conv_activation": ['relu', 'relu'],

             "action_dense_lay": 2,
             "action_n_neurons": [256, 256],
             "action_dense_activation": ['relu', 'relu'],

             "value_dense_lay": 2,
             "value_n_neurons": [256, 256],
             "value_dense_activation": ['relu', 'relu']
             }


# DPG network input and hidden layers architecture. For dense only discriminator, convolutional parameters will be not readed.
dpg_net = {"conv_layers": 3,
           "kernel_num": [32, 32, 64],
           "kernel_size": [9, 7, 3],
           "kernel_strides": [4, 2, 2],
           "conv_activation": ['relu', 'relu', 'relu'],
           "dense_lay": 2,
           "n_neurons": [256, 256],
           "dense_activation": ['relu', 'relu'],
           }

# a2c network input and hidden layers architecture. This architecture will use two diferent architectures for actor
# and critic nets. For dense only discriminator, convolutional parameters will be not readed.
a2c_net = {"actor_conv_layers": 3,
           "actor_kernel_num": [32, 32, 64],
           "actor_kernel_size": [7, 5, 3],
           "actor_kernel_strides": [4, 2, 1],
           "actor_conv_activation": ['relu', 'relu', 'relu'],
           "actor_dense_lay": 1,
           "actor_n_neurons": [256],
           "actor_dense_activation": ['relu'],

           "critic_conv_layers": 3,
           "critic_kernel_num": [32, 32, 64],
           "critic_kernel_size": [7, 5, 3],
           "critic_kernel_strides": [4, 2, 1],
           "critic_conv_activation": ['relu', 'relu', 'relu'],
           "critic_dense_lay": 1,
           "critic_n_neurons": [128],
           "critic_dense_activation": ['relu'],
           }

# DDPG network input and hidden layers architecture. This algorithm use two diferent architectures for actor
# and critic nets. Last layer activation function of the critic network will be linear even if other activation function
# is specified. For dense only discriminator, convolutional parameters will be not readed.
ddpg_net = {"actor_conv_layers": 3,
           "actor_kernel_num": [32, 32, 64],
           "actor_kernel_size": [7, 5, 3],
           "actor_kernel_strides": [4, 2, 1],
           "actor_conv_activation": ['relu', 'relu', 'relu'],
           "actor_dense_lay": 2,
           "actor_n_neurons": [128, 64],
           "actor_dense_activation": ['relu', 'relu'],

           "critic_conv_layers": 3,
           "critic_kernel_num": [32, 32, 64],
           "critic_kernel_size": [7, 5, 3],
           "critic_kernel_strides": [4, 2, 1],
           "critic_conv_activation": ['relu', 'relu', 'relu'],
           "critic_dense_lay": 2,
           "critic_n_neurons": [128, 64],
           "critic_dense_activation": ['linear', 'linear'],
           }

# a3c network input and hidden layers architecture. This architecture will use two diferent architectures for actor
# and critic nets. For dense only discriminator, convolutional parameters will be not readed.
a3c_net = {"actor_conv_layers": 3,
           "actor_kernel_num": [32, 32, 128],
           "actor_kernel_size": [7, 5, 3],
           "actor_kernel_strides": [4, 2, 1],
           "actor_conv_activation": ['relu', 'relu', 'relu'],
           "actor_dense_lay": 1,
           "actor_n_neurons": [256],
           "actor_dense_activation": ['relu'],

           "critic_conv_layers": 3,
           "critic_kernel_num": [32, 32, 128],
           "critic_kernel_size": [7, 5, 3],
           "critic_kernel_strides": [4, 2, 1],
           "critic_conv_activation": ['relu', 'relu', 'relu'],
           "critic_dense_lay": 1,
           "critic_n_neurons": [128],
           "critic_dense_activation": ['relu'],
           }

# ppo network input and hidden layers architecture. This architecture will use two diferent architectures for actor
# and critic nets. For dense only discriminator, convolutional parameters will be not readed.
ppo_net = {"actor_conv_layers": 3,
           "actor_kernel_num": [32, 32, 64],
           "actor_kernel_size": [7, 5, 3],
           "actor_kernel_strides": [4, 2, 1],
           "actor_conv_activation": ['relu', 'relu', 'relu'],
           "actor_dense_lay": 2,
           "actor_n_neurons": [128, 128],
           "actor_dense_activation": ['tanh', 'tanh'],

           "critic_conv_layers": 3,
           "critic_kernel_num": [32, 32, 64],
           "critic_kernel_size": [7, 5, 3],
           "critic_kernel_strides": [4, 2, 1],
           "critic_conv_activation": ['relu', 'relu', 'relu'],
           "critic_dense_lay": 1,
           "critic_n_neurons": [128, 128],
           "critic_dense_activation": ['tanh', 'tanh'],
           }

# ppo network input and hidden layers architecture. This architecture will use two diferent architectures for actor
# and critic nets. For dense only discriminator, convolutional parameters will be not readed.
irl_net = {"state_conv_layers": 3,
           "state_kernel_num": [32, 32, 64],
           "state_kernel_size": [7, 5, 3],
           "state_kernel_strides": [4, 2, 1],
           "state_conv_activation": ['relu', 'relu', 'relu'],
           "state_dense_lay": 2,
           "state_n_neurons": [128, 128],
           "state_dense_activation": ['tanh', 'tanh'],

           "action_dense_lay": 1,
           "action_n_neurons": [128, 128],
           "action_dense_activation": ['tanh', 'tanh'],

           "common_dense_lay": 2,
           "common_n_neurons": [128, 128],
           "common_dense_activation": ['tanh', 'tanh'],
           "use_custom_network": None,
           "state_custom_network": None,
           "action_custom_network": None,
           "common_custom_network": None
           }
