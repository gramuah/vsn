from RL_Problem.base.rl_problem_base import *
import numpy as np

class PPOProblemBase(RLProblemSuper):
    """
    Proximal Policy Optimization.
    """
    def __init__(self, environment, agent, continuous=False):
        """
        Attributes:
                environment:    Environment selected for this problem
                agent:          Agent to solve the problem: DDPG.
                n_stack:        Int >= 1. If 1, there is no stacked input. Number of time related input stacked.
                img_input:      Bool. If True, input data is an image.
                state_size:     None, Int or Tuple. State dimensions. If None it will be calculated automatically. Int
                                or Tuple format will be useful when preprocessing change the input dimensions.
                model_params:   Dictionary of params like learning rate, batch size, epsilon values, n step returns...
        """
        super().__init__(environment, agent)

        self.episode = 0
        self.val = False
        self.reward = []
        self.reward_over_time = []

        self.gradient_steps = 0

        self.batch_size = agent.batch_size
        self.memory_size = agent.memory_size
        # self.learning_rate = learning_rate

        # List of 10 last rewards
        self.rew_mean_list = deque(maxlen=2)
        self.global_steps = 0
        self.total_episodes = 0

        self.obs_batch = []
        self.actions_batch = []
        self.actions_probs_batch = []
        self.rewards_batch = []
        self.masks_batch = []

        self.continuous = continuous

        if continuous:
            self.action_bound = [self.env.action_space.low, self.env.action_space.high]  # action bounds
        else:
            self.action_bound = None

        if not agent.agent_builded:
            self._build_agent()

        # for IRL
        self.agent_traj = deque(maxlen=10000)
        self.disc_loss = 100

    def _build_agent(self):
        if self.img_input:
            stack = self.n_stack is not None and self.n_stack > 1
            # TODO: Tratar n_stack como ambos channel last and channel first
            if len(self.state_size) > 2:
                state_size = (*self.state_size[:2], self.state_size[-1] * self.n_stack)
            else:
                state_size = self.state_size


        elif self.n_stack is not None and self.n_stack > 1:
            stack = True
            state_size = (self.n_stack, self.state_size)
        else:
            stack = False
            state_size = self.state_size


        self._define_agent(self.n_actions, state_size, stack, action_bound=self.action_bound)
        # self._define_agent(agent=self.agent, state_size=state_size, n_actions=self.n_actions, stack=stack,
        #                           img_input=self.img_input, actor_lr=self.actor_lr, critic_lr=self.critic_lr,
        #                           batch_size=self.batch_size, buffer_size=self.buffer_size, epsilon=self.epsilon,
        #                           epsilon_decay=self.epsilon_decay, epsilon_min=self.epsilon_min,
        #                           action_bound=self.action_bound, net_architecture=self.net_architecture,
        #                           n_threads=self.n_threads)

    def solve(self, episodes, render=True, render_after=None, max_step_epi=None, skip_states=1, verbose=1,
              discriminator=None, expert_traj=None, save_live_histogram=False, smooth_rewards=2):
        """ Algorithm for training the agent to solve the environment problem.

        :param episodes:        Int >= 1. Number of episodes to train.
        :param render:          Bool. If True, the environment will show the user interface during the training process.
        :param render_after:    Int >=1 or None. Star rendering the environment after this number of episodes.
        :param max_step_epi:    Int >=1 or None. Maximum number of epochs per episode. Mainly for problems where the
                                environment
                                doesn't have a maximum number of epochs specified.
        :param skip_states:     Int >= 1. Frame skipping technique  applied in Playing Atari With Deep Reinforcement
                                Learning paper. If 1, this technique won't be applied.
        :param verbose:         Int in range [0, 2]. If 0 no training information will be displayed, if 1 lots of
                                information will be displayed, if 2 fewer information will be displayed.
        :param save_live_histogram: Path for recording live evaluation params
        :return:
        """
        # Inicializar iteraciones globales
        if discriminator is None:
            self.global_steps = 0
            self.total_episodes = 0
        self.episode = 0
        # List of 100 last rewards
        self.rew_mean_list = deque(maxlen=smooth_rewards)

        while self.episode < episodes:
            self.collect_batch(render, render_after, max_step_epi, skip_states, verbose, discriminator, expert_traj,
                               save_live_histogram)
            actor_loss, critic_loss = self.agent.replay()

            print('Actor loss', actor_loss.history['loss'][-1], self.gradient_steps)
            print('Critic loss', critic_loss.history['loss'][-1], self.gradient_steps)

            self.gradient_steps += 1

            self.agent.save_tensorboar_rl_histogram(self.histogram_metrics)
            if self.agent.model.actor_checkpoint and self.gradient_steps % self.agent.model.save_every_iterations == 0:
                print('Saving checkpoint...')
                self.agent.model.save_checkpoint()

    def collect_batch(self, render, render_after, max_step_epi, skip_states, verbose, discriminator=None,
                      expert_traj=None, save_live_histories=False):
        self.obs_batch = []
        self.actions_batch = []
        self.actions_probs_batch = []
        self.rewards_batch = []
        self.masks_batch = []
        # Stacking inputs
        if self.n_stack is not None and self.n_stack > 1:
            obs_queue = deque(maxlen=self.n_stack)
            obs_next_queue = deque(maxlen=self.n_stack)
        else:
            obs_queue = None
            obs_next_queue = None

        while len(self.obs_batch) < self.memory_size:
            tmp_batch = [[], [], [], [], [], [], []]

            # TODO: normalizar como se ejecutan estos test, si se harán en todos los algoritmos y como puede controlar el usuario si se hacen o no.
            # if self.episode % 99 == 0:
            #     self.test(n_iter=4, render=True)

            obs = self.env.reset()
            episodic_reward = 0
            steps = 0
            done = False
            success = 0
            self.reward = []

            obs = self.preprocess(obs)

            # Stacking inputs
            if self.n_stack is not None and self.n_stack > 1:
                for i in range(self.n_stack):
                    # obs_queue.append(np.zeros(obs.shape))
                    # obs_next_queue.append(np.zeros(obs.shape))
                    obs_queue.append(obs)
                    obs_next_queue.append(obs)

                obs_queue.append(obs)
                obs_next_queue.append(obs)

            while not done:  # and len(batch[0])+len(tmp_batch[0]) < self.buffer_size:
                if render or ((render_after is not None) and self.episode > render_after):
                    self.env.render(view='top')

                # Select an action
                action, action_matrix, predicted_action = self.act_train(obs, obs_queue)

                # Agent act in the environment
                next_obs, reward, done, info = self.env.step(action)


                if discriminator is not None:
                    if discriminator.stack:
                        reward = discriminator.get_reward(obs_queue, action)[0]
                    else:
                        reward = discriminator.get_reward(obs, action)[0]

                # Store the experience in episode memory
                # Here we apply the preprocess/formatting function
                next_obs, obs_next_queue, reward, done, steps, mask = self.store_episode_experience(action,
                                                                                                    done,
                                                                                                    next_obs,
                                                                                                    obs,
                                                                                                    obs_next_queue,
                                                                                                    obs_queue,
                                                                                                    reward,
                                                                                                    skip_states,
                                                                                                    steps,
                                                                                                    predicted_action,
                                                                                                    action_matrix)

                # copy next_obs to obs
                obs, obs_queue = self.copy_next_obs(next_obs, obs, obs_next_queue, obs_queue)



                episodic_reward += reward
                steps += 1
                self.global_steps += 1

                done, success = self._max_steps(done, steps, max_step_epi)

            self.reduce_exploration_noise()
            self.episode += 1
            self.total_episodes += 1
            # Add reward to the list
            self.rew_mean_list.append(episodic_reward)
            self.histogram_metrics.append([self.total_episodes, episodic_reward, steps, success, self.agent.epsilon, self.global_steps])

            if save_live_histories:
                if isinstance(save_live_histories, str):
                    write_history(rl_hist=self.histogram_metrics, monitor_path=save_live_histories)
                else:
                    raise Exception('Type of parameter save_live_histories must be string but ' +
                                    str(type(save_live_histories)) + ' has been received')

            # Print log on scream
            self._feedback_print(self.total_episodes, episodic_reward, steps, success, verbose, self.rew_mean_list)

        if discriminator is not None and expert_traj is not None:

            """ Estrategia para entrenar solo cuando la pérdida es mas alta de la cuenta"""
            # TODO: Qizás no proceda mantener esta estrategia para la biblioteca
            if self.disc_loss > 0.01:
                # agent_traj = [[np.array(o), np.array(a)] for o, a in zip(self.obs_batch, self.actions_batch)]
                # discriminator.train(expert_traj, agent_traj)
                # agent_traj = [np.array([o, [np.argmax(a)]]) for o, a in zip(self.obs_batch, self.actions_batch)]
                # discriminator.train(expert_traj, agent_traj)
                if discriminator.use_expert_actions:
                    [self.agent_traj.append([np.array(o), np.array(a)]) for o, a in
                     zip(self.obs_batch, self.actions_batch)]
                else:
                    [self.agent_traj.append(np.array(o) for o in self.obs_batch)]

                train_loss, self.disc_loss = discriminator.train(expert_traj, self.agent_traj)

                self.rewards_batch = [discriminator.get_reward(o, a)[0] for o, a in
                                      zip(self.obs_batch, self.actions_batch)]
                if save_live_histories:
                    if isinstance(save_live_histories, str):
                        write_history(il_hist=[train_loss, self.disc_loss, discriminator.global_epochs], monitor_path=save_live_histories)
                    else:
                        raise Exception('Type of parameter save_live_histories must be string but ' +
                                        str(type(save_live_histories)) + ' has been received')
            else:
                self.disc_loss += 0.0025



        self.agent.remember(self.obs_batch, self.actions_batch, self.actions_probs_batch, self.rewards_batch,
                            self.masks_batch)

    def store_episode_experience(self, action, done, next_obs, obs, obs_next_queue, obs_queue, reward, skip_states, epochs,
                                 predicted_action, action_matrix):
        # Here we apply the preprocess/formatting function
        done, next_obs, reward, epochs = self.frame_skipping(action, done, next_obs, reward, skip_states, epochs)

        mask = not done
        if self.n_stack is not None and self.n_stack > 1:
            obs_next_queue.append(next_obs)

            if self.img_input:
                obs_queue_stack = np.dstack(obs_queue)
            else:
                # obs_next_stack = np.array(obs_next_queue).reshape(self.state_size, self.n_stack)
                obs_queue_stack = np.array(obs_queue)
            self.obs_batch.append(obs_queue_stack)
        else:
            self.obs_batch.append(obs)

        self.actions_batch.append(action_matrix)
        self.actions_probs_batch.append(predicted_action)
        self.rewards_batch.append(reward)
        self.masks_batch.append(mask)

        return next_obs, obs_next_queue, reward, done, epochs, mask

    def reduce_exploration_noise(self):
        """
        For continuous action spaces
        """
        pass

    def _define_agent(self, n_actions, state_size, stack, action_bound):
        """
        Create the agent. It is specified in each class which inherit this one.
        """
        pass