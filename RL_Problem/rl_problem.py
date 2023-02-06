from RL_Agent.base.utils import agent_globals


def Problem(environment, agent):
    """ Method for selecting an algorithm to use
    :param environment: (EnvInterface or Gym environment) Environment selected.
    :param agent: (AgentInterface) Agent selected.
    :return: Built RL problem. Instance of RLProblemSuper.
    """
    if agent.agent_name == agent_globals.names["dqn"]:
        from RL_Problem.base.ValueBased import dqn_problem
        problem = dqn_problem.DQNProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["ddqn"]:
        from RL_Problem.base.ValueBased import dqn_problem
        problem = dqn_problem.DQNProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["dddqn"]:
        from RL_Problem.base.ValueBased import dqn_problem
        problem = dqn_problem.DQNProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["dpg"]:
        from RL_Problem.base.PolicyBased import dpg_problem
        problem = dpg_problem.DPGProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["dpg_continuous"]:
        from RL_Problem.base.PolicyBased import dpg_problem
        problem = dpg_problem.DPGProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["ddpg"]:
        from RL_Problem.base.ActorCritic import ddpg_problem
        problem = ddpg_problem.DDPGPRoblem(environment, agent)

    elif agent.agent_name == agent_globals.names["a2c_discrete"]:
        from RL_Problem.base.ActorCritic import a2c_problem
        problem = a2c_problem.A2CProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["a2c_continuous"]:
        from RL_Problem.base.ActorCritic import a2c_problem
        problem = a2c_problem.A2CProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["a2c_discrete_queue"]:
        from RL_Problem.base.ActorCritic import a2c_problem
        problem = a2c_problem.A2CProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["a2c_continuous_queue"]:
        from RL_Problem.base.ActorCritic import a2c_problem
        problem = a2c_problem.A2CProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["ppo_discrete"]:
        from RL_Problem.base.PPO import ppo_problem_discrete
        problem = ppo_problem_discrete.PPOProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["ppo_discrete_habitat"]:
        from RL_Problem.base.PPO import ppo_problem_discrete_habitat
        problem = ppo_problem_discrete_habitat.PPOProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["ppo_continuous"]:
        from RL_Problem.base.PPO import ppo_problem_continuous
        problem = ppo_problem_continuous.PPOProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["ppo_discrete_multithread"]:
        from RL_Problem.base.PPO import ppo_problem_discrete_parallel
        problem = ppo_problem_discrete_parallel.PPOProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["ppo_discrete_multithread_habitat"]:
        from RL_Problem.base.PPO import ppo_problem_discrete_parallel_habitat
        problem = ppo_problem_discrete_parallel_habitat.PPOProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["ppo_continuous_multithread"]:
        from RL_Problem.base.PPO import ppo_problem_continuous_parallel
        problem = ppo_problem_continuous_parallel.PPOProblem(environment, agent)

    elif agent.agent_name == agent_globals.names["ppo_continuous_multithread_transformer"]:
        from RL_Problem.base.PPO import ppo_problem_continuous_parallel
        problem = ppo_problem_continuous_parallel.PPOProblem(environment, agent)

    return problem

# elif agent.agent_name == agent_globals.names["a3c_continuous"]:
# from RL_Problem.base.ActorCritic import a3c_problem
# problem = a3c_problem.A3CProblem(environment, agent)
#
# elif agent.agent_name == agent_globals.names["a3c_discrete"]:
# from RL_Problem.base.ActorCritic import a3c_problem
#
# problem = a3c_problem.A3CProblem(environment, agent)
#
# elif agent.agent_name == agent_globals.names["a3c_discrete_tf"]:
# from RL_Problem.base.ActorCritic import a3c_problem_tf
#
# problem = a3c_problem_tf.A3CProblem(environment, agent)
#
# elif agent.agent_name == agent_globals.names["ppo_s2s_continuous_multithread"]:
#     from RL_Problem.base.PPO import ppo_problem_continuous_parallel
#     problem = ppo_problem_continuous_parallel.PPOProblem(environment, agent)
# elif agent.agent_name == agent_globals.names["ppo_transformer_agent_continuous_multithread"]:
#     from RL_Problem.base.PPO import ppo_problem_continuous_parallel
#     problem = ppo_problem_continuous_parallel.PPOProblem(environment, agent)
# elif agent.agent_name == agent_globals.names["ppo_transformer_agent_discrete_multithread"]:
#     from RL_Problem.base.PPO import ppo_problem_discrete_parallel
#     problem = ppo_problem_discrete_parallel.PPOProblem(environment, agent)