import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', lambda x: '%.2f' % x)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

import datetime
import os
import sys
import yaml
from RL_Agent.base.utils.networks.action_selection_options import *
from utils.log_utils import Unbuffered
from RL_Problem import rl_problem
from RL_Agent import ppo_agent_discrete_parallel, ppo_agent_discrete
from environments.maze import PyMaze
from RL_Agent.base.utils.networks import networks
from RL_Agent.base.utils.networks.agent_networks import PPONet, MazePPONet
from RL_Agent.base.utils import agent_saver, history_utils
from maze_experiments.utils.neuralnets import *

# Loading yaml configuration files
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    if isinstance(config_file, str) and config_file.endswith('.yaml'):
        with open(config_file) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
else:
    raise Exception('No config.yaml file is provided')

# define agent's neural networks to use
exec('actor_model = ' + config["actor_model"])
exec('critic_model = ' + config["critic_model"])

n_stack = config["n_stack"]
if config["state_size"] == 'None':
    state_size = None
else:
    exec('state_size = ' + config["state_size"])
exec('train_action_selection_options = ' + config["train_action_selection_options"])
exec('action_selection_options = ' + config["action_selection_options"])
if config["preprocess"] == 'None':
    preprocess = None
else:
    exec('preprocess = ' + config["preprocess"])


def custom_model(input_shape):
    return MazePPONet(input_shape,
                      actor_model,
                      critic_model,
                      tensorboard_dir=None,
                      save_every_iterations=None,
                      checkpoints_to_keep=None,
                      checkpoint_path=None)


net_architecture = networks.ppo_net(use_tf_custom_model=True,
                                    tf_custom_model=custom_model)

agent = ppo_agent_discrete.Agent(actor_lr=float(config["actor_lr"]),
                                 critic_lr=float(config["critic_lr"]),
                                 batch_size=config["batch_size"],
                                 memory_size=config["memory_size"],
                                 epsilon=config["epsilon_min"],
                                 epsilon_decay=config["epsilon_decay"],
                                 epsilon_min=config["epsilon_min"],
                                 gamma=config["gamma"],
                                 loss_clipping=config["loss_clipping"],
                                 loss_critic_discount=config["loss_critic_discount"],
                                 loss_entropy_beta=config["loss_entropy_beta"],
                                 lmbda=config["lmbda"],
                                 train_epochs=config["train_epochs"],
                                 net_architecture=net_architecture,
                                 n_stack=n_stack,
                                 is_habitat=config["is_habitat"],
                                 img_input=config["img_input"],
                                 state_size=state_size,
                                 train_action_selection_options=train_action_selection_options,
                                 action_selection_options=action_selection_options)

data = pd.DataFrame(columns=['Total_Episodes', 'Reward', 'Steps', 'Success', 'Epsilon', 'Global_Steps'])

for i in range(100):

    exec('environment = ' + config["environment"])
    environment = environment(forward_step=config['forward_step'],
                              turn_step=config['turn_step'],
                              num_rows=config['num_rows'],
                              num_cols=config['num_cols'],
                              domain_rand=config['domain_rand'],
                              max_steps=config['max_steps'],
                              use_clip=config['use_clip'],
                              sparse_reward=config['sparse_reward'],
                              seed=i)
    # plt.imshow(environment.render_top_view())
    # plt.show()

    problem = rl_problem.Problem(environment, agent)
    agent.model.load_checkpoint(path='../../../resultados/MAZE_PAPER/S5/11/26-10-2022_19-47-59_checkpoints', checkpoint_to_restore='latest')

    problem.test(render=config["render_test"], n_iter=config["test_epochs"], max_step_epi=config['max_steps'])
    # #
    data = data.append(pd.DataFrame(problem.get_histogram_metrics(), columns=list(data)), ignore_index=True)

print(data.describe(include='all'))
