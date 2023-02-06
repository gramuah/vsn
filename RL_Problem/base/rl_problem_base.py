import copy
import numpy as np

from collections import deque
from abc import ABCMeta, abstractmethod
from RL_Agent.base.utils.history_utils import write_history

class RLProblemSuper(object, metaclass=ABCMeta):
    """ Reinforcement Learning Problem.

    This class represent the RL problem to solve formed by an agent and an environment. The RL problem controls and
    defines how, when and where the agent and the environment send information to each other or in other words, the
    events flow between agent and environment.
    Specific problem classes for each agent should extend this class in order to adjust and introduce specific features
    in the events flow.
    """
    def __init__(self, environment, agent):
        """
        Attributes:
        :param environment:    (EnvInterface) Environment selected for this RL problem.
        :param agent:          (AgentInterface) Selected agent for solving the environment.
        """
        self.env = environment
        self.agent = agent

        self.is_habitat = agent.is_habitat
        self.n_stack = agent.n_stack
        self.img_input = agent.img_input

        # Set state_size depending on the input type
        self.state_size = agent.env_state_size

        if self.state_size is None:
            if self.img_input:
                self.state_size = self.env.observation_space.shape
            else:
                self.state_size = self.env.observation_space.shape[0]
            agent.env_state_size = self.state_size

        # Set n_actions depending on the environment format
        try:
            self.n_actions = self.env.action_space.n
        except AttributeError:
            self.n_actions = self.env.action_space.shape[0]

        # Setting default preprocess and clip_norm_reward functions
        self.preprocess = self._preprocess  # Preprocessing function for observations
        self.clip_norm_reward = self._clip_norm_reward  # Clipping reward

        # Total number of steps processed
        self.global_steps = 0
        self.total_episodes = 0

        self.max_rew_mean = -2**1000  # Store the maximum value for reward mean
        self.histogram_metrics = []

    @abstractmethod
    def _build_agent(self):
        """
        This method should call the agent build_agent method. This method needs to be called when a new Rl problem is
        defined or where an already trained agent will be used in a new RL problem.
        """
        pass

    def compile(self):
        self.agent.compile()


    def solve(self, episodes, render=True, render_after=None, max_step_epi=None, skip_states=1, verbose=1,
              discriminator=None, save_live_histogram=False, smooth_rewards=10):
        """ Method for training the agent to solve the environment problem. The reinforcement learning loop is 
        implemented here.

        :param episodes: (int) >= 1. Number of episodes to train.
        :param render: (bool) If True, the environment will show the user interface during the training process.
        :param render_after: (int) >=1 or None. Star rendering the environment after this number of episodes.
        :param max_step_epi: (int) >=1 or None. Maximum number of epochs per episode. Mainly for problems where the
            environment doesn't have a maximum number of epochs specified.
        :param skip_states: (int) >= 1. Frame skipping technique applied in Playing Atari With Deep Reinforcement paper.
            If 1, this technique won't be applied.
        :param verbose: (int) in range [0, 3]. If 0 no training information will be displayed, if 1 lots of
           information will be displayed, if 2 fewer information will be displayed and 3 a minimum of information will
           be displayed.
        :param save_live_histogram: (bool or str) Path for recording live evaluation params. If is set to False, no
            information will be recorded.
        :return:
        """
        self.compile()
        # Inicializar iteraciones globales
        if discriminator is None:
            self.global_steps = 0
            self.total_episodes = 0

        # List of 100 last rewards
        rew_mean_list = deque(maxlen=smooth_rewards)

        # Stacking inputs
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = deque(maxlen=self.n_stack)
            obs_next_queue = deque(maxlen=self.n_stack)
        else:
            obs_queue = None
            obs_next_queue = None

        # For each episode do
        for e in range(episodes):

            # Init episode parameters
            obs = self.env.reset()
            episodic_reward = 0
            steps = 0
            done = False
            success = 0

            # Reading initial state
            obs = self.preprocess(obs)
            # obs = np.zeros((300, 300))

            # Stacking inputs
            if self.n_stack is not None and self.n_stack > 1:
                for i in range(self.n_stack):
                    obs_queue.append(np.zeros(obs.shape))
                    obs_next_queue.append(np.zeros(obs.shape))
                obs_queue.append(obs)
                obs_next_queue.append(obs)

            # While the episode doesn't reach a final state
            while not done:
                if render or ((render_after is not None) and e > render_after):
                    self.env.render()

                # Select an action
                action = self.act_train(obs, obs_queue)

                # Agent act in the environment
                next_obs, reward, done, _ = self.env.step(action)
                if discriminator is not None:
                    if discriminator.stack:
                        reward = discriminator.get_reward(obs_queue, action)[0]
                    else:
                        reward = discriminator.get_reward(obs, action)[0]
                # next_obs = np.zeros((300, 300))
                # next_obs = self.preprocess(next_obs)  # Is made in store_experience now

                # Store the experience in memory
                next_obs, obs_next_queue, reward, done, steps = self.store_experience(action, done, next_obs, obs, obs_next_queue,
                                                                         obs_queue, reward, skip_states, steps)

                # Replay some memories and training the agent
                self.agent.replay()

                # copy next_obs to obs
                obs, obs_queue = self.copy_next_obs(next_obs, obs, obs_next_queue, obs_queue)

                # If max steps value is reached the episode is finished

                episodic_reward += reward
                steps += 1
                done, success = self._max_steps(done, steps, max_step_epi)

                self.global_steps += 1

            # Add reward to the list
            rew_mean_list.append(episodic_reward)
            self.histogram_metrics.append([self.total_episodes, episodic_reward, steps, success, self.agent.epsilon, self.global_steps])

            if save_live_histogram:
                if isinstance(save_live_histogram, str):
                    write_history(rl_hist=self.histogram_metrics, monitor_path=save_live_histogram)
                else:
                    raise Exception('Type of parameter save_live_histories must be string but ' +
                                    str(type(save_live_histogram)) + ' has been received')

            # Copy main model to target model
            self.agent.copy_model_to_target()

            # Print log on scream
            self._feedback_print(self.total_episodes, episodic_reward, steps, success, verbose, rew_mean_list)
            self.total_episodes += 1

            self.agent.save_tensorboar_rl_histogram(self.histogram_metrics)
        return

    def copy_next_obs(self, next_obs, obs, obs_next_queue, obs_queue):
        """
        Make a copy of the current observation ensuring the is no conflicts of two variables pointing common values.
        """
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = copy.copy(obs_next_queue)
        else:
            obs = next_obs
        return obs, obs_queue

    def act_train(self, obs, obs_queue):
        """
        Make the agent select an action in training mode given an observation. Use an input depending if the
        observations are stacked in
        time or not.
        :param obs: (numpy nd array) observation (state).
        :param obs_queue: (numpy nd array) List of observation (states) in sequential time steps.
        :return: (int or [floats]) int if actions are discrete or numpy array of float of action shape if actions are
            continuous)
        """
        if self.n_stack is not None and self.n_stack > 1:
            action = self.agent.act_train(np.array(obs_queue))
        else:
            action = self.agent.act_train(obs)
        return action

    def act(self, obs, obs_queue):
        """
        Make the agent select an action in exploitation mode given an observation. Use an input depending if the
        observations are stacked.
        time or not.
        :param obs: (numpy nd array) observation (states).
        :param obs_queue: (numpy nd array) List of observation (states) in sequential time steps.
        :return: (int or [floats]) int if actions are discrete or numpy array of float of action shape if actions are
            continuous)
        """
        if self.n_stack is not None and self.n_stack > 1:
            action = self.agent.act(np.array(obs_queue))
        else:
            action = self.agent.act(obs)
        return action

    def store_experience(self, action, done, next_obs, obs, obs_next_queue, obs_queue, reward, skip_states, epochs):
        """
        Method for store a experience in the agent memory. A standard experience consist of a tuple (observation,
        action, reward, next observation, done flag).
        :param action: (int, [int] or [float]) Action selected by the agent. Type depend on the action type (discrete or
            continuous) and the agent needs.
        :param done: (bool) Episode is finished flag. True denotes that next_obs or obs_next_queue represent a final
            state.
        :param next_obs: (numpy nd array) Next observation to obs (state).
        :param obs: (numpy nd array) Observation (state).
        :param obs_next_queue: (numpy nd array) List of Next observations to obs_queue (states) in sequential time steps.
        :param obs_queue: (numpy nd array) List of observation (states) in sequential time steps.
        :param reward: (float) Regard value obtained by the agent in the current experience.
        :param skip_states: (int) >= 0. Select the states to skip with the frame skipping technique (explained in
            Playing Atari With Deep Reinforcement paper).
        :param epochs: (int) Episode epochs counter.
        """
        # Execute the frame skipping technique explained in Playing Atari With Deep Reinforcement paper.
        done, next_obs, reward, epochs = self.frame_skipping(action, done, next_obs, reward, skip_states, epochs)

        # Store the experience in memory depending on stacked inputs and observations type
        if self.n_stack is not None and self.n_stack > 1:
            obs_next_queue.append(next_obs)

            if self.img_input:
                obs_satck = np.dstack(obs_queue)
                obs_next_stack = np.dstack(obs_next_queue)
            else:
                obs_satck = np.array(obs_queue)
                obs_next_stack = np.array(obs_next_queue)

            self.agent.remember(obs_satck, action, self.clip_norm_reward(reward), obs_next_stack, done)
        else:
            self.agent.remember(obs, action, self.clip_norm_reward(reward), next_obs, done)
        return next_obs, obs_next_queue, reward, done, epochs

    def frame_skipping(self, action, done, next_obs, reward, skip_states, epochs):
        """
        This method execute the frame skipping technique explained in Playing Atari With Deep Reinforcement paper. It
        consist on repeating the last selected action n times whit the objective of explore a bigger space.
        :param action: (int, [int] or [float]) Action selected by the agent. Type depend on the action type (discrete or
            continuous) and the agent needs.
        :param done: (bool) Episode is finished flag. True denotes that next_obs or obs_next_queue represent a final
            state.
        :param next_obs: (numpy nd array) Next observation to obs (state).
        :param reward: (float) Regard value obtained by the agent in the current experience.
        :param skip_states: (int) >= 0. Select the number of states to skip.
        :param epochs: (int) Spisode epochs counter.
        """
        if skip_states > 1 and not done:
            for i in range(skip_states - 2):
                next_obs_aux1, reward_aux, done_aux, _ = self.env.step(action)
                epochs += 1
                reward += reward_aux
                if done_aux:
                    next_obs_aux2 = next_obs_aux1
                    done = done_aux
                    break

            if not done:
                next_obs_aux2, reward_aux, done_aux, _ = self.env.step(action)
                epochs += 1
                reward += reward_aux
                done = done_aux

            if self.img_input:
                next_obs_aux2 = self.preprocess(next_obs_aux2)
                if skip_states > 2:
                    next_obs_aux1 = self.preprocess(next_obs_aux1)
                    # TODO: esto no se debería hacer con todas las imágenes intermedias? consultar en paper atari dqn
                    next_obs = np.maximum(next_obs_aux2, next_obs_aux1)

                else:
                    next_obs = self.preprocess(next_obs)
                    next_obs = np.maximum(next_obs_aux2, next_obs)
            else:
                next_obs = self.preprocess(next_obs_aux2)
        else:
            next_obs = self.preprocess(next_obs)
        return done, next_obs, reward, epochs

    def test(self, n_iter=10, render=True, verbose=1, callback=None, smooth_rewards=10, discriminator=None, max_step_epi=None):
        """ Test a trained agent using only exploitation mode on the environment.

        :param n_iter: (int) number of test iterations.
        :param render: (bool) If True, the environment will show the user interface during the training process.
        :param verbose: (int) in range [0, 3]. If 0 no training information will be displayed, if 1 lots of
           information will be displayed, if 2 fewer information will be displayed and 3 a minimum of information will
           be displayed.
        :param callback: A extern function that receives a tuple (prev_obs, obs, action, reward, done, info)
        """
        epi_rew_mean = 0
        rew_mean_list = deque(maxlen=smooth_rewards)

        # Stacking inputs
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = deque(maxlen=self.n_stack)
        else:
            obs_queue = None

        # For each episode do
        for e in range(n_iter):
            done = False
            episodic_reward = 0
            success = 0
            steps = 0
            obs = self.env.reset()
            obs = self.preprocess(obs)

            # stacking inputs
            if self.n_stack is not None and self.n_stack > 1:
                for i in range(self.n_stack):
                    obs_queue.append(obs)
                obs_queue.append(obs)

            while not done:
                if render:
                    self.env.render()

                # Select action
                # TODO: poner bien
                # action = self.act(obs, obs_queue)
                action = self.act(obs, obs_queue)

                prev_obs = obs

                obs, reward, done, info = self.env.step(action)
                obs = self.preprocess(obs)

                if discriminator is not None:
                    if discriminator.stack:
                        reward = discriminator.get_reward(obs_queue, action, multithread=False)[0]
                    else:
                        reward = discriminator.get_reward(obs, action, multithread=False)[0]

                if callback is not None:
                    callback(prev_obs, obs, action, reward, done, info)

                episodic_reward += reward
                steps += 1

                if self.n_stack is not None and self.n_stack > 1:
                    obs_queue.append(obs)

                done, success = self._max_steps(done, steps, max_step_epi)

            rew_mean_list.append(episodic_reward)
            self.histogram_metrics.append([self.total_episodes, episodic_reward, steps, success, self.agent.epsilon, self.global_steps])
            self._feedback_print(e, episodic_reward, steps, success, verbose, rew_mean_list, test=True)

        # print('Mean Reward ', epi_rew_mean / n_iter)
        self.env.close()

    def _preprocess(self, obs):
        """
        Preprocessing function by default does nothing to the observation.
        :param obs: (numpy nd array) Observation (state).
        """
        return obs

    def _clip_norm_reward(self, rew):
        """
        Clip and/or normalize the reward. By default does nothing to the reward value.
        :param rew: (float) Regard value obtained by the agent in a experience.
        """
        return rew

    def _max_steps(self, done, steps, max_steps):
        """
        Return True if number of epochs pass a selected number of steps. This allow to set a maximum number of
        iterations for each RL epoch.
        :param done: (bool) Episode is finished flag. True if the episode has finished.
        :param steps: (int) Episode epochs counter.
        :param max_steps: (int) Maximum number of episode epochs. When it is reached param done is set to True.
        """
        if max_steps is not None:
            if done and steps < max_steps:
                return done, 1
        return done, 0

    def _feedback_print(self, episode, episodic_reward, steps, success, verbose, epi_rew_list, test=False):
        """
        Print on terminal information about the training process.
        :param episode: (int) Current episode.
        :param episodic_reward: (float) Cumulative reward of the last episode.
        :param steps: (int) Episode steps counter.
        :param verbose: (int) in range [0, 3]. If 0 no training information will be displayed, if 1 lots of
           information will be displayed, if 2 fewer information will be displayed and 3 a minimum of information will
           be displayed.
        :param epi_rew_list: ([float]) List of reward of the last episode.
        :param test: (bool) Flag for select test mode. True = test mode, False = train mode.
        """
        rew_mean = np.sum(epi_rew_list) / len(epi_rew_list)

        if test:
            episode_str = 'Test episode: '
        else:
            episode_str = 'Episode: '
        if verbose == 1:

            if (episode + 1) % 1 == 0:

                if hasattr(self.env, "current_episode"):
                    # It is a habitat env so we use its info
                    print(episode_str, episode,
                          '| Steps: ', steps,
                          '| Reward: {:.1f}'.format(episodic_reward),
                          '| Smooth Reward: {:.1f}'.format(rew_mean),
                          '| Epsilon: {:.4f}'.format(self.agent.epsilon),
                          '| Current Episode: {}'.format(self.env.current_episode.episode_id),
                          '| Current Scene: {}'.format(self.env.current_episode.scene_id.split('/')[4]),
                          '| SPL: {:.1f}'.format(self.env.get_info(None)['spl']),
                          '| Distance to Goal: {:.1f}'.format(self.env.get_info(None)['distance_to_goal']),
                          '| Episode Success: {:.1f}'.format(self.env.get_success()))
                else:
                    print(episode_str, episode,
                          '| Steps: ', steps,
                          '| Success: ', success,
                          '| Reward: {:.1f}'.format(episodic_reward),
                          '| Smooth Reward: {:.1f}'.format(rew_mean),
                          '| Epsilon: {:.4f}'.format(self.agent.epsilon))

        if verbose == 2:
            print(episode_str, episode + 1, 'Mean Reward: ', rew_mean)
        if verbose == 3:
            print(episode_str, episode + 1)

    # def load_model(self, dir_load="", name_loaded=""):
    #     self.agent._load(dir_load)
    #
    # def save_agent(self, agent_name=None):
    #     if agent_name is None:
    #         agent_name = self.agent.agent_name
    #     with open(agent_name, 'wb') as f:
    #         dill.dump(self.agent, f)

    def get_histogram_metrics(self):
        """
        Return the history of metrics consisting on a array with rows:  [episode number, episode reward, episode epochs,
        epsilon value, global steps]
        return: (2D array)
        """
        return np.array(self.histogram_metrics)

