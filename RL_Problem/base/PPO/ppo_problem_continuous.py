from RL_Problem.base.PPO.ppo_problem_base import PPOProblemBase
import numpy as np


class PPOProblem(PPOProblemBase):
    """
    Proximal Policy Optimization.
    """
    def __init__(self, environment, agent, n_stack=1, img_input=False, state_size=None, model_params=None,
                 saving_model_params=None, net_architecture=None):
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

        super().__init__(environment, agent, continuous=True)

    def _define_agent(self, n_actions, state_size, stack, action_bound=None):
        self.agent.build_agent(state_size=state_size, n_actions=n_actions, stack=stack, action_bound=action_bound)

    def reduce_exploration_noise(self):
        """
        For continuous action spaces
        """
        self.agent.exploration_noise = np.random.rand() * 2 / 10