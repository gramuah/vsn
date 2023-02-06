import numpy as np

def clip_reward_atari(rew):
    if rew > 0:
        return 1
    elif rew < 0:
        return -1
    else:
        return 0

def clip_reward_atari_v2(rew):
    return np.sign(rew)*np.log(1+np.abs(rew))

def clip_pendulum_rew(rew):
    return (rew + 8) / 8
