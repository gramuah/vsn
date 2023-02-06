import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import history_utils
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path
import json

def main():
    root = Path(__file__).parent.parent.parent.parent
    if len(sys.argv) > 0:
        path = sys.argv
        path = path[1]
    else:
        path = str(root) + '/monitor_historial.json'
    n_moving_average = 10
    headers = ['episodes', 'reward', 'epi_epochs', 'epsilon', 'global_step']

    plt.ion()
    fig = plt.figure(1)
    fig.suptitle('Reward history', fontsize=16)
    ax1 = fig.add_subplot(221)
    ax1.set_ylabel('RL Reward')
    ax1.set_xlabel('RL Episodes')

    ax1.plot([], [], 'r', label="reward")
    ax1.plot([], [], 'b', label="average " + str(n_moving_average) + " rewards")
    ax1.legend(loc="upper left")

    ax2 = fig.add_subplot(222)
    ax2.set_ylabel('RL Epochs')
    ax2.set_xlabel('Rl Episodes')

    ax2.plot([], [], 'g', label="episode epochs")
    ax2.plot([], [], 'y', label="average " + str(n_moving_average) + "epochs")
    ax2.legend(loc="upper left")

    ax3 = fig.add_subplot(223)
    ax3.set_ylabel('RL Epsilon')
    ax3.set_xlabel('RL Global Steps')

    ax3.plot([], [], 'k', label="Epsilon")

    ax4 = fig.add_subplot(224)
    ax4.set_ylabel('IL Loss')
    ax4.set_xlabel('Discriminator train epochs')

    ax4.plot([], [], 'c', label="Train loss")
    ax4.plot([], [], 'm', label="Val loss")
    ax4.legend(loc="upper left")

    fig.show()

    while True:
        try:
            # data = pd.read_csv(path, sep="\t", names=headers)
            with open(path, 'r') as f:
                data = json.load(f)
            rl_data = data['rl_data']
            episodes = rl_data['episode']
            rewards = rl_data['reward']
            epi_epochs = rl_data['n_epochs']
            epsilon = rl_data['epsilon']
            global_step = rl_data['global_steps']

            il_data = data['il_data']
            il_train_loss = il_data['train_loss']
            il_val_loss = [il_data['val_loss']]
            il_epochs = il_data['epochs']

            # episodes = data.episodes.values
            # rewards = data.reward.values
            # epi_epochs = data.epi_epochs.values
            # epsilon = data.epsilon.values
            # global_step = data.global_step.values

            y_mean = history_utils.moving_average(rewards, n_moving_average)

            ax1.lines[0].set_data(episodes, rewards)
            ax1.lines[1].set_data(episodes, y_mean)

            ax1.relim()  # recompute the data limits
            ax1.autoscale_view()  # automatic axis scaling

            epi_epochs_mean = history_utils.moving_average(epi_epochs, n_moving_average)

            ax2.lines[0].set_data(episodes, epi_epochs)
            ax2.lines[1].set_data(episodes, epi_epochs_mean)
            ax2.relim()  # recompute the data limits
            ax2.autoscale_view()  # automatic axis scaling

            ax3.lines[0].set_data(global_step, epsilon)
            ax3.relim()  # recompute the data limits
            ax3.autoscale_view()  # automatic axis scaling

            ax4.lines[0].set_data(il_epochs, il_train_loss)
            ax4.lines[1].set_data(il_epochs, il_val_loss)
            ax4.relim()  # recompute the data limits
            ax4.autoscale_view()  # automatic axis scaling

        except:
            pass

        fig.canvas.flush_events()
        time.sleep(5)
        if not plt.get_fignums():
            sys.exit()




main()
