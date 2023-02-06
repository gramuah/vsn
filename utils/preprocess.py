import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean


def atari_assault_preprocess(obs):
    # Crop and resize the image

    obs = obs[20:196:2, ::2]

    # Convert the image to greyscale
    obs = obs.mean(axis=2)

    # eee = True
    # if eee:
    #     plt.clf()
    #     fig = plt.figure(1)
    #     plt.imshow(obs, cmap="gray")
    #     plt.draw()
    #
    #     plt.pause(10e-50)
    #     plt.show()
    # Improve image contrast
    # obs[50:][obs[50:] == color] = 255

    # normalize between from 0 to 1
    obs = obs / 255.

    return obs.reshape(88, 80, 1)


def atari_pacman_preprocess(obs):
    color = np.array([210, 164, 74]).mean()

    # Crop and resize the image
    img = obs[1:176:2, ::2]

    # Convert the image to greyscale
    img = img.mean(axis=2)

    # Improve image contrast
    img[img == color] = 0

    # Next we normalize the image from -1 to +1
    img = (img - 128) / 128 - 1

    return img.reshape(88, 80, 1)


def to_grayscale(obs):
    # Convert the image to greyscale
    img = obs.mean(axis=2)

    # # Improve image contrast
    # img[img == color] = 0

    return img.reshape(img.shape[0], img.shape[1], 1)


def preproces_car_racing(obs):
    # Crop and resize the image
    # obs = obs[10:-10, 10:-10]
    obs = color.rgb2gray(obs)
    # plt.imshow(obs, cmap='gray')
    # plt.show()
    # obs = obs[::3, ::3]

    # Convert the image to greyscale
    # obs = obs.mean(axis=2)
    obs = resize(obs, (obs.shape[0] // 2, obs.shape[1] // 2), anti_aliasing=False)
    obs = obs[:-6, 3:-3]
    # plt.clf()
    # plt.imshow(obs, cmap='gray')
    # plt.draw()
    # plt.pause(10e-50)
    img = obs / 255.
    # # Improve image contrast
    # img[img == color] = 0

    return img.reshape(img.shape[0], img.shape[1], 1)

def preprocess_habitat_clip(obs: dict):
    """
    Normalize rgb input and one hot objectgoal id
    @param obs: observations returned by habitat
    @return: dict containing the preprocessed observations
    """

    # RGB input already normalized
    rgb = obs['rgb'][0]
    # img = rgb / 255.

    # ObjectGoal input to onehot
    object_goal = obs['objectgoal']
    one_hot_goal = np.zeros((12))
    one_hot_goal[object_goal] = 1.

    return {'rgb': rgb, 'objectgoal': one_hot_goal}

def preprocess_habitat(obs: dict):
    """
    Normalize rgb input and one hot objectgoal id
    @param obs: observations returned by habitat
    @return: dict containing the preprocessed observations
    """

    # RGB input normalized
    rgb = obs['rgb']
    img = rgb / 255.

    # ObjectGoal input to onehot
    object_goal = obs['objectgoal']
    one_hot_goal = np.zeros((12))
    one_hot_goal[object_goal] = 1.

    return {'rgb': img, 'objectgoal': one_hot_goal}

def preprocess_habitat6(obs: dict):
    """
    Normalize rgb input and one hot objectgoal id
    @param obs: observations returned by habitat
    @return: dict containing the preprocessed observations
    """

    # RGB input normalized
    rgb = obs['rgb']
    img = rgb / 255.

    # ObjectGoal input to onehot
    object_goal = obs['objectgoal']
    one_hot_goal = np.zeros((6))
    one_hot_goal[object_goal] = 1.

    return {'rgb': img, 'objectgoal': one_hot_goal}

def preprocess_clip_habitat6(obs: dict):
    """
    Normalize rgb input and one hot objectgoal id
    @param obs: observations returned by habitat
    @return: dict containing the preprocessed observations
    """

    # RGB input normalized
    rgb = obs['rgb'][0]

    # ObjectGoal input to onehot
    object_goal = obs['objectgoal']
    one_hot_goal = np.zeros((6))
    one_hot_goal[object_goal] = 1.

    return {'rgb': rgb, 'objectgoal': one_hot_goal}

def preprocess_clip_habitat6_stack(obs: dict):
    """
    Normalize rgb input and one hot objectgoal id
    @param obs: observations returned by habitat
    @return: dict containing the preprocessed observations
    """

    # RGB input normalized
    rgb = obs['rgb'][0]

    # ObjectGoal input to onehot
    object_goal = obs['objectgoal']
    one_hot_goal = np.zeros((6))
    one_hot_goal[object_goal] = 1.

    return {'rgb': rgb, 'objectgoal': one_hot_goal}