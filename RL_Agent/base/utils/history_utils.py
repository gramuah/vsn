import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import json

def plot_reward_hist(hist, n_moving_average):
    x = hist[:, 0]
    y = hist[:, 1]

    y_mean = moving_average(y, n_moving_average)

    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    fig.suptitle('Reward history', fontsize=16)

    ax.plot(x, y, 'r', label="reward")
    ax.plot(x, y_mean, 'b', label="average "+str(n_moving_average)+" rewards")

    ax.legend(loc="upper left")

    ax.set_ylabel('Reward')
    ax.set_xlabel('Episodes')

    plt.show()

def moving_average(values, window):
    values = np.array(values)
    if window%2 !=0:
        expand_dims_f = int(window / 2)
        expand_dims_l = int(window / 2)
    else:
        expand_dims_f = int(window / 2) - 1
        expand_dims_l = int(window / 2)

    for i in range(expand_dims_f):
        values = np.insert(values, 0, values[0])
    for i in range(expand_dims_l):
        values = np.append(values, values[-1])
    weights = np.repeat(1.0, window)/window
    sma = np.convolve(values, weights, 'valid')
    return sma


def write_history(rl_hist=False, il_hist=False, monitor_path=False):

    if monitor_path:
        path = monitor_path
    else:
        root = Path(__file__).parent.parent.parent.parent
        path = str(root) + '/monitor_historial.json'

    data = False
    # file = open(path, "r")
    try:
        with open(path, 'r') as f:
            data = json.load(f)
    except:
        pass
    # file.writelines(str(hist[-1][0]) + "\t" + str(hist[-1][1]) + "\t" +
    #                 str(hist[-1][2]) + "\t" + str(hist[-1][3]) + "\t" + str(hist[-1][4]) + "\n")


    if data:
        rl_data = data['rl_data']
        il_data = data['il_data']
        if rl_hist:
            rl_data['episode'].append(rl_hist[-1][0])
            rl_data['reward'].append(float(rl_hist[-1][1]))
            rl_data['n_epochs'].append(rl_hist[-1][2])
            rl_data['epsilon'].append(float(rl_hist[-1][3]))
            rl_data['global_steps'].append(rl_hist[-1][4])
        if il_hist:
            if len(il_hist[0]) > 1 and len(il_hist[1]) > 1:
                il_data['train_loss'].append(float(il_hist[0][0]))
                il_data['val_loss'].append(float(il_hist[1][0]))
                il_data['train_acc'].append(float(il_hist[0][1]))
                il_data['val_acc'].append(float(il_hist[1][1]))
                il_data['epochs'].append(il_hist[2])
            else:
                il_data['train_loss'].append(float(il_hist[0]))
                il_data['val_loss'].append(float(il_hist[1]))
                il_data['epochs'].append(il_hist[2])
    else:
        if rl_hist:
            rl_data = {
                'episode': [rl_hist[-1][0]],
                'reward': [float(rl_hist[-1][1])],
                'n_epochs': [rl_hist[-1][2]],
                'epsilon': [float(rl_hist[-1][3])],
                'global_steps': [rl_hist[-1][4]]
            }
        else:
            rl_data = {
                'episode': [],
                'reward': [],
                'n_epochs': [],
                'epsilon': [],
                'global_steps': []
            }
        if il_hist:
                if len(il_hist[0]) > 1 and len(il_hist[1]) > 1:
                    il_data = {
                        'train_loss': [float(il_hist[0][0])],
                        'val_loss': [float(il_hist[1][0])],
                        'train_acc': [float(il_hist[0][1])],
                        'val_acc': [float(il_hist[1][1])],
                        'epochs': [il_hist[2]]
                    }
                else:
                    il_data = {
                        'train_loss': [float(il_hist[0])],
                        'val_loss': [float(il_hist[1])],
                        'epochs': [il_hist[2]]
                    }
        else:
            il_data = {
                'train_loss': [],
                'val_loss': [],
                'train_acc': [],
                'val_acc': [],
                'epochs': []
            }

    data = {
        'rl_data': rl_data,
        'il_data': il_data
    }
    with open(path, 'w') as f:
        json.dump(data, f)

    # file.close()
