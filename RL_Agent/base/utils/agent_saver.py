import types

# from RL_Agent import dqn_agent, ddqn_agent, dddqn_agent
# from RL_Agent.legacy_agents import dpg_agent, ddpg_agent
# from RL_Agent.legacy_agents import a2c_agent_discrete, a2c_agent_continuous, a2c_agent_discrete_queue, a2c_agent_continuous_queue
# from RL_Agent import a3c_agent_discrete, a3c_agent_continuous
# from RL_Agent import ppo_agent_discrete, ppo_agent_continuous, ppo_agent_discrete_parallel, \
#     ppo_agent_continuous_parallel
from RL_Agent.base.utils import agent_globals
import pickle
import json
import numpy as np
import time
import base64
import copy
import os
import shutil
import marshal
import dill

def prueba():
    return 1

def save(agent, path):
    # Save network object from RLNet
    assert isinstance(path, str)

    folder = os.path.dirname(path)

    # agent_aux = copy.deepcopy(agent)
    if not os.path.exists(folder) and folder != '':
        os.makedirs(folder)

    agent._save_network(path)

    # file_name = os.path.basename(path)
    tmp_path = 'capoirl_tmp_saving_folder/'



    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    # agent._save_network(tmp_path + 'tmp_model')
    agent_att, custom_net, actor_custom_net, critic_custom_net, common_custom_net, value_custom_net, adv_custom_net \
        = extract_agent_attributes(agent)

    custom_globals = actor_custom_globals = critic_custom_globals = common_custom_globals = value_custom_globals = adv_custom_globals = None

    # save the network
    if agent_att['net_architecture'] is not None and 'use_custom_network' in agent_att['net_architecture'].keys() \
            and agent_att['net_architecture']['use_custom_network']:
        if custom_net is not None:
            custom_globals = dill.dumps(custom_net.__globals__)
            custom_globals = base64.b64encode(custom_globals).decode('ascii')
            custom_net = marshal.dumps(custom_net.__code__)
            custom_net = base64.b64encode(custom_net).decode('ascii')

        # TODO: esclarecer si hace falta hacer este paso con protobuffer y checkpoints en custom nets
        elif actor_custom_net is not None and critic_custom_net is not None:
            actor_custom_globals = dill.dumps(actor_custom_net.__globals__)
            actor_custom_globals = base64.b64encode(actor_custom_globals).decode('ascii')
            actor_custom_net = marshal.dumps(actor_custom_net.__code__)
            actor_custom_net = base64.b64encode(actor_custom_net).decode('ascii')

            critic_custom_globals = dill.dumps(critic_custom_net.__globals__)
            critic_custom_globals = base64.b64encode(critic_custom_globals).decode('ascii')
            critic_custom_net = marshal.dumps(critic_custom_net.__code__)
            critic_custom_net = base64.b64encode(critic_custom_net).decode('ascii')

        elif common_custom_net is not None and value_custom_net is not None and adv_custom_net is not None:
            common_custom_globals = dill.dumps(common_custom_net.__globals__)
            common_custom_globals = base64.b64encode(common_custom_globals).decode('ascii')
            common_custom_net = marshal.dumps(common_custom_net.__code__)
            common_custom_net = base64.b64encode(common_custom_net).decode('ascii')

            value_custom_globals = dill.dumps(value_custom_net.__globals__)
            value_custom_globals = base64.b64encode(value_custom_globals).decode('ascii')
            value_custom_net = marshal.dumps(value_custom_net.__code__)
            value_custom_net = base64.b64encode(value_custom_net).decode('ascii')

            adv_custom_globals = dill.dumps(adv_custom_net.__globals__)
            adv_custom_globals = base64.b64encode(adv_custom_globals).decode('ascii')
            adv_custom_net = marshal.dumps(adv_custom_net.__code__)
            adv_custom_net = base64.b64encode(adv_custom_net).decode('ascii')

    agent_att = pickle.dumps(agent_att)
    agent_att = base64.b64encode(agent_att).decode('ascii')

    data = {
        'agent': agent_att,
        'custom_net': custom_net,
        'custom_globals': custom_globals,
        'actor_custom_net': actor_custom_net,
        'actor_custom_globals': actor_custom_globals,
        'critic_custom_net': critic_custom_net,
        'critic_custom_globals': critic_custom_globals,
        'common_custom_net': common_custom_net,
        'common_custom_globals': common_custom_globals,
        'value_custom_net': value_custom_net,
        'value_custom_globals': value_custom_globals,
        'adv_custom_net': adv_custom_net,
        'adv_custom_globals': adv_custom_globals,
    }

    with open(os.path.join(path, 'agent_data.json'), 'w') as f:
        json.dump(data, f)

    shutil.rmtree(tmp_path)

def export_to_protobuf(agent, path):
    """
    Save the agent neural network into protobuffer format for deploying.
    This do not allows to retrain the agent once is loaded.
    :param agent: RL_Agent to saves.
    :param path: str. Folder for saving the agent
    """
    assert isinstance(path, str)

    folder = os.path.dirname(path)

    # agent_aux = copy.deepcopy(agent)
    if not os.path.exists(folder) and folder != '':
        os.makedirs(folder)

    agent._save_protobuf(path)

    # file_name = os.path.basename(path)
    tmp_path = 'capoirl_tmp_saving_folder/'



    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    # agent._save_network(tmp_path + 'tmp_model')
    agent_att, custom_net, actor_custom_net, critic_custom_net, common_custom_net, value_custom_net, adv_custom_net \
        = extract_agent_attributes(agent)

    custom_globals = actor_custom_globals = critic_custom_globals = common_custom_globals = value_custom_globals = adv_custom_globals = None

    # # save the network
    # if agent_att['net_architecture'] is not None and 'use_custom_network' in agent_att['net_architecture'].keys() \
    #         and agent_att['net_architecture']['use_custom_network']:
    #     if custom_net is not None:
    #         custom_globals = dill.dumps(custom_net.__globals__)
    #         custom_globals = base64.b64encode(custom_globals).decode('ascii')
    #         custom_net = marshal.dumps(custom_net.__code__)
    #         custom_net = base64.b64encode(custom_net).decode('ascii')
    #
    #     # TODO: esclarecer si hace falta hacer este paso con protobuffer y checkpoints en custom nets
    #     elif actor_custom_net is not None and critic_custom_net is not None:
    #         actor_custom_globals = dill.dumps(actor_custom_net.__globals__)
    #         actor_custom_globals = base64.b64encode(actor_custom_globals).decode('ascii')
    #         actor_custom_net = marshal.dumps(actor_custom_net.__code__)
    #         actor_custom_net = base64.b64encode(actor_custom_net).decode('ascii')
    #
    #         critic_custom_globals = dill.dumps(critic_custom_net.__globals__)
    #         critic_custom_globals = base64.b64encode(critic_custom_globals).decode('ascii')
    #         critic_custom_net = marshal.dumps(critic_custom_net.__code__)
    #         critic_custom_net = base64.b64encode(critic_custom_net).decode('ascii')
    #
    #     elif common_custom_net is not None and value_custom_net is not None and adv_custom_net is not None:
    #         common_custom_globals = dill.dumps(common_custom_net.__globals__)
    #         common_custom_globals = base64.b64encode(common_custom_globals).decode('ascii')
    #         common_custom_net = marshal.dumps(common_custom_net.__code__)
    #         common_custom_net = base64.b64encode(common_custom_net).decode('ascii')
    #
    #         value_custom_globals = dill.dumps(value_custom_net.__globals__)
    #         value_custom_globals = base64.b64encode(value_custom_globals).decode('ascii')
    #         value_custom_net = marshal.dumps(value_custom_net.__code__)
    #         value_custom_net = base64.b64encode(value_custom_net).decode('ascii')
    #
    #         adv_custom_globals = dill.dumps(adv_custom_net.__globals__)
    #         adv_custom_globals = base64.b64encode(adv_custom_globals).decode('ascii')
    #         adv_custom_net = marshal.dumps(adv_custom_net.__code__)
    #         adv_custom_net = base64.b64encode(adv_custom_net).decode('ascii')

    agent_att = pickle.dumps(agent_att)
    agent_att = base64.b64encode(agent_att).decode('ascii')

    data = {
        'agent': agent_att,
        'custom_net': custom_net,
        'custom_globals': custom_globals,
        # 'actor_custom_net': actor_custom_net,
        # 'actor_custom_globals': actor_custom_globals,
        # 'critic_custom_net': critic_custom_net,
        # 'critic_custom_globals': critic_custom_globals,
        'common_custom_net': common_custom_net,
        'common_custom_globals': common_custom_globals,
        'value_custom_net': value_custom_net,
        'value_custom_globals': value_custom_globals,
        'adv_custom_net': adv_custom_net,
        'adv_custom_globals': adv_custom_globals,
    }

    with open(os.path.join(path, 'agent_data.json'), 'w') as f:
        json.dump(data, f)

    shutil.rmtree(tmp_path)

def save_legacy(agent, path):
    assert isinstance(path, str)

    folder = os.path.dirname(path)
    # file_name = os.path.basename(path)
    tmp_path = 'capoirl_tmp_saving_folder/'

    # agent_aux = copy.deepcopy(agent)
    if not os.path.exists(folder) and folder != '':
        os.makedirs(folder)

    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    agent._save_network(tmp_path + 'tmp_model')
    agent_att, custom_net, actor_custom_net, critic_custom_net, common_custom_net, value_custom_net, adv_custom_net \
        = extract_agent_attributes(agent)

    custom_globals = actor_custom_globals = critic_custom_globals = common_custom_globals = value_custom_globals = adv_custom_globals = None

    if agent_att['net_architecture'] is not None and 'use_custom_network' in agent_att['net_architecture'].keys() \
            and agent_att['net_architecture']['use_custom_network']:
        if custom_net is not None:
            custom_globals = dill.dumps(custom_net.__globals__)
            custom_globals = base64.b64encode(custom_globals).decode('ascii')
            custom_net = marshal.dumps(custom_net.__code__)
            custom_net = base64.b64encode(custom_net).decode('ascii')


        elif actor_custom_net is not None and critic_custom_net is not None:
            actor_custom_globals = dill.dumps(actor_custom_net.__globals__)
            actor_custom_globals = base64.b64encode(actor_custom_globals).decode('ascii')
            actor_custom_net = marshal.dumps(actor_custom_net.__code__)
            actor_custom_net = base64.b64encode(actor_custom_net).decode('ascii')

            critic_custom_globals = dill.dumps(critic_custom_net.__globals__)
            critic_custom_globals = base64.b64encode(critic_custom_globals).decode('ascii')
            critic_custom_net = marshal.dumps(critic_custom_net.__code__)
            critic_custom_net = base64.b64encode(critic_custom_net).decode('ascii')

        elif common_custom_net is not None and value_custom_net is not None and adv_custom_net is not None:
            common_custom_globals = dill.dumps(common_custom_net.__globals__)
            common_custom_globals = base64.b64encode(common_custom_globals).decode('ascii')
            common_custom_net = marshal.dumps(common_custom_net.__code__)
            common_custom_net = base64.b64encode(common_custom_net).decode('ascii')

            value_custom_globals = dill.dumps(value_custom_net.__globals__)
            value_custom_globals = base64.b64encode(value_custom_globals).decode('ascii')
            value_custom_net = marshal.dumps(value_custom_net.__code__)
            value_custom_net = base64.b64encode(value_custom_net).decode('ascii')

            adv_custom_globals = dill.dumps(adv_custom_net.__globals__)
            adv_custom_globals = base64.b64encode(adv_custom_globals).decode('ascii')
            adv_custom_net = marshal.dumps(adv_custom_net.__code__)
            adv_custom_net = base64.b64encode(adv_custom_net).decode('ascii')

    agent_att = pickle.dumps(agent_att)
    agent_att = base64.b64encode(agent_att).decode('ascii')

    try:
        f_json = tmp_path + 'tmp_model.json'
        with open(f_json, 'rb') as fp:
            json_data = fp.read()
        json_data = base64.b64encode(json_data).decode('ascii')
    except:
        json_data = None

    try:
        f_h5 = tmp_path + 'tmp_model.h5'
        with open(f_h5, 'rb') as fp:
            h5_data = fp.read()
        h5_data = base64.b64encode(h5_data).decode('ascii')
    except:
        h5_data = None

    try:
        f_h5 = tmp_path + 'tmp_modelactor.h5'
        with open(f_h5, 'rb') as fp:
            h5_actor = fp.read()
        h5_actor_data = base64.b64encode(h5_actor).decode('ascii')
        f_h5 = tmp_path + 'tmp_modelcritic.h5'
        with open(f_h5, 'rb') as fp:
            h5_critic = fp.read()
        h5_critic_data = base64.b64encode(h5_critic).decode('ascii')
    except:
        h5_actor_data = None
        h5_critic_data = None

    try:
        f_check = tmp_path + 'checkpoint'
        with open(f_check, 'rb') as fp:
            checkpoint = fp.read()
        checkpoint = base64.b64encode(checkpoint).decode('ascii')
        f_check = tmp_path + 'tmp_model.index'
        with open(f_check, 'rb') as fp:
            checkpoint_index = fp.read()
        checkpoint_index = base64.b64encode(checkpoint_index).decode('ascii')
        f_check = tmp_path + 'tmp_model.meta'
        with open(f_check, 'rb') as fp:
            checkpoint_meta = fp.read()
        checkpoint_meta = base64.b64encode(checkpoint_meta).decode('ascii')

        for file in os.listdir(tmp_path):
            if 'tmp_model.data' in file:
                f_check = tmp_path + file
        with open(f_check, 'rb') as fp:
            checkpoint_data = fp.read()
        checkpoint_data = base64.b64encode(checkpoint_data).decode('ascii')

    except:
        checkpoint = None
        checkpoint_index = None
        checkpoint_meta = None
        checkpoint_data = None

    data = {
        'agent': agent_att,
        'model_json': json_data,
        'model_h5': h5_data,
        'model_ckpt': checkpoint,
        'actor_h5': h5_actor_data,
        'critic_h5': h5_critic_data,
        'model_index': checkpoint_index,
        'model_meta': checkpoint_meta,
        'model_data': checkpoint_data,
        'custom_net': custom_net,
        'custom_globals': custom_globals,
        'actor_custom_net': actor_custom_net,
        'actor_custom_globals': actor_custom_globals,
        'critic_custom_net': critic_custom_net,
        'critic_custom_globals': critic_custom_globals,
        'common_custom_net': common_custom_net,
        'common_custom_globals': common_custom_globals,
        'value_custom_net': value_custom_net,
        'value_custom_globals': value_custom_globals,
        'adv_custom_net': adv_custom_net,
        'adv_custom_globals': adv_custom_globals
    }

    with open(path, 'w') as f:
        json.dump(data, f)

    shutil.rmtree(tmp_path)

def load(path, agent, overwrite_attrib=False):
    """
    Load an agent from file.
    :param path: (string) path to saved agent folder
    :param agent: (RL_Agent) agent entity to load.
    :param overwrite_attrib: (bool) If False the agent's attributes will be loaded from file. If True the new defined
                                agent's attributes will be used
    """
    with open(os.path.join(path, 'agent_data.json'), 'r') as f:
        data = json.load(f)

    agent_att = base64.b64decode(data['agent'])
    agent_att = pickle.loads(agent_att)

    try:
        custom_net = base64.b64decode(data['custom_net'])
        custom_globals = base64.b64decode(data['custom_globals'])
    except:
        custom_net = None
        custom_globals = None

    try:
        actor_custom_net = base64.b64decode(data['actor_custom_net'])
        actor_custom_globals = base64.b64decode(data['actor_custom_globals'])
        critic_custom_net = base64.b64decode(data['critic_custom_net'])
        critic_custom_globals = base64.b64decode(data['critic_custom_globals'])
    except:
        actor_custom_net = None
        actor_custom_globals = None
        critic_custom_net = None
        critic_custom_globals = None

    try:
        common_custom_net = base64.b64decode(data['common_custom_net'])
        common_custom_globals = base64.b64decode(data['common_custom_globals'])
        value_custom_net = base64.b64decode(data['value_custom_net'])
        value_custom_globals = base64.b64decode(data['value_custom_globals'])
        adv_custom_net = base64.b64decode(data['adv_custom_net'])
        adv_custom_globals = base64.b64decode(data['adv_custom_globals'])
    except:
        common_custom_net = None
        common_custom_globals = None
        value_custom_net = None
        value_custom_globals = None
        adv_custom_net = None
        adv_custom_globals = None

    # custom_nets = pickle.loads(custom_nets)

    if custom_net is not None:
        custom_globals = dill.loads(custom_globals)
        custom_globals = process_globals(custom_globals)
        # Quizas sea mejor definir las dependencias dentro de la propia función para luego cargarla bien
        code = marshal.loads(custom_net)
        custom_net = types.FunctionType(code, custom_globals, "custom_net_func")
        agent_att['net_architecture']['custom_network'] = custom_net

    elif actor_custom_net is not None and critic_custom_net is not None:
        actor_custom_globals = dill.loads(actor_custom_globals)
        actor_custom_globals = process_globals(actor_custom_globals)
        code = marshal.loads(actor_custom_net)
        actor_custom_net = types.FunctionType(code, actor_custom_globals, "actor_custom_net_func")
        agent_att['net_architecture']['actor_custom_network'] = actor_custom_net

        critic_custom_globals = dill.loads(critic_custom_globals)
        critic_custom_globals = process_globals(critic_custom_globals)
        code = marshal.loads(critic_custom_net)
        critic_custom_net = types.FunctionType(code, critic_custom_globals, "critic_custom_net_func")
        agent_att['net_architecture']['critic_custom_network'] = critic_custom_net

    elif common_custom_net is not None and value_custom_net is not None and adv_custom_net is not None:
        common_custom_globals = dill.loads(common_custom_globals)
        common_custom_globals = process_globals(common_custom_globals)
        code = marshal.loads(common_custom_net)
        common_custom_net = types.FunctionType(code, common_custom_globals, "common_custom_net_func")
        agent_att['net_architecture']['common_custom_network'] = common_custom_net

        value_custom_globals = dill.loads(value_custom_globals)
        value_custom_globals = process_globals(value_custom_globals)
        code = marshal.loads(value_custom_net)
        value_custom_net = types.FunctionType(code, value_custom_globals, "value_custom_net_func")
        agent_att['net_architecture']['value_custom_network'] = value_custom_net

        adv_custom_globals = dill.loads(adv_custom_globals)
        adv_custom_globals = process_globals(adv_custom_globals)
        code = marshal.loads(adv_custom_net)
        adv_custom_net = types.FunctionType(code, adv_custom_globals, "adv_custom_net_func")
        agent_att['net_architecture']['action_custom_network'] = adv_custom_net

    # tmp_path = 'capoirl_tmp_loading_folder/'
    #
    # if not os.path.exists(tmp_path):
    #     os.makedirs(tmp_path)
    #
    # try:
    #     model_json_bytes = base64.b64decode(data['model_json'])
    #     with open(tmp_path + 'tmp_model.json', 'wb') as fp:
    #         fp.write(model_json_bytes)
    # except:
    #     pass
    # try:
    #     model_h5_bytes = base64.b64decode(data['model_h5'])
    #     with open(tmp_path + 'tmp_model.h5', 'wb') as fp:
    #         fp.write(model_h5_bytes)
    # except:
    #     pass
    #
    # try:
    #     model_actor_bytes = base64.b64decode(data['actor_h5'])
    #     with open(tmp_path + 'tmp_modelactor.h5', 'wb') as fp:
    #         fp.write(model_actor_bytes)
    #     model_critic_bytes = base64.b64decode(data['critic_h5'])
    #     with open(tmp_path + 'tmp_modelcritic.h5', 'wb') as fp:
    #         fp.write(model_critic_bytes)
    # except:
    #     pass
    #
    # try:
    #     checkpoint_bytes = base64.b64decode(data['model_ckpt'])
    #     with open(tmp_path + 'checkpoint', 'wb') as fp:
    #         fp.write(checkpoint_bytes)
    #     index_ckpt_bytes = base64.b64decode(data['model_index'])
    #     with open(tmp_path + 'tmp_model.index', 'wb') as fp:
    #         fp.write(index_ckpt_bytes)
    #     meta_ckpt_bytes = base64.b64decode(data['model_meta'])
    #     with open(tmp_path + 'tmp_model.meta', 'wb') as fp:
    #         fp.write(meta_ckpt_bytes)
    #     data_ckpt_bytes = base64.b64decode(data['model_data'])
    #     with open(tmp_path + 'tmp_model.data-00000-of-00001', 'wb') as fp:
    #         fp.write(data_ckpt_bytes)
    # except:
    #     pass

    # if agent is None:
    #     # TODO: ¿Eliminar esta opción?
    #     # If there is not input agent create a raw new agent
    #     agent = create_new_agent(agent_att)
    #     set_agent_attributes(agent_att, agent, overwrite=True)
    if not agent.agent_builded:
        # If the agent is not built, build it.
        # if agent.loads_saved_params:
        set_agent_attributes(agent_att, agent, not overwrite_attrib, set_nones=True)
        if agent_att['action_low_bound'] is None and agent_att['action_high_bound'] is None:
            agent.build_agent(state_size=agent.state_size,
                              n_actions=agent.n_actions,
                              stack=agent.stack)
        else:
            # agent.build_agent(state_size=agent_att["state_size"],
            #                   n_actions=agent_att["n_actions"],
            #                   stack=agent_att["stack"],
            #                   action_bound=[agent_att['action_low_bound'], agent_att['action_high_bound']])
            agent.build_agent(state_size=agent.state_size,
                              n_actions=agent.n_actions,
                              stack=agent.stack,
                              action_bound=[agent_att['action_low_bound'], agent_att['action_high_bound']])
    else:
        # If the agent is built, load its attributes
        # TODO: No tiene que sobreescribir todo
        set_agent_attributes(agent_att, agent, not overwrite_attrib)

    agent._load(path)

    return agent

def load_from_protobuf(path, agent=None):
    """
     Load an agent from protobuffer format for deploying.
     This do not allows to retrain the agent once is loaded.
     :param path: str. Folder for saving the agent
     :param agent: RL_Agent. Initialized agent to load.
     """
    with open(os.path.join(path, 'agent_data.json'), 'r') as f:
        data = json.load(f)

    agent_att = base64.b64decode(data['agent'])
    agent_att = pickle.loads(agent_att)

    if agent is None:
        # TODO: ¿Eliminar esta opción?
        # If there is not input agent create a raw new agent
        agent = create_new_agent(agent_att)
        set_agent_attributes(agent_att, agent, overwrite=True)
    elif not agent.agent_builded:
        # If the agent is not built, build it.
        set_agent_attributes(agent_att, agent)
        if agent_att['action_low_bound'] is None and agent_att['action_high_bound'] is None:
            agent.build_agent(state_size=agent_att["state_size"],
                              n_actions=agent_att["n_actions"],
                              stack=agent_att["stack"])
        else:
            agent.build_agent(state_size=agent_att["state_size"],
                              n_actions=agent_att["n_actions"],
                              stack=agent_att["stack"],
                              action_bound=[agent_att['action_low_bound'], agent_att['action_high_bound']])
    else:
        # If the agent is built, load its attributes
        # TODO: No tiene que sobreescribir todo
        set_agent_attributes(agent_att, agent)
    agent._load_protobuf(path)

    return agent

def load_legacy(path, agent=None):
    with open(path, 'r') as f:
        data = json.load(f)

    agent_att = base64.b64decode(data['agent'])
    agent_att = pickle.loads(agent_att)

    try:
        custom_net = base64.b64decode(data['custom_net'])
        custom_globals = base64.b64decode(data['custom_globals'])
    except:
        custom_net = None
        custom_globals = None

    try:
        actor_custom_net = base64.b64decode(data['actor_custom_net'])
        actor_custom_globals = base64.b64decode(data['actor_custom_globals'])
        critic_custom_net = base64.b64decode(data['critic_custom_net'])
        critic_custom_globals = base64.b64decode(data['critic_custom_globals'])
    except:
        actor_custom_net = None
        actor_custom_globals = None
        critic_custom_net = None
        critic_custom_globals = None

    try:
        common_custom_net = base64.b64decode(data['common_custom_net'])
        common_custom_globals = base64.b64decode(data['common_custom_globals'])
        value_custom_net = base64.b64decode(data['value_custom_net'])
        value_custom_globals = base64.b64decode(data['value_custom_globals'])
        adv_custom_net = base64.b64decode(data['adv_custom_net'])
        adv_custom_globals = base64.b64decode(data['adv_custom_globals'])
    except:
        common_custom_net = None
        common_custom_globals = None
        value_custom_net = None
        value_custom_globals = None
        adv_custom_net = None
        adv_custom_globals = None

    # custom_nets = pickle.loads(custom_nets)

    if custom_net is not None:
        custom_globals = dill.loads(custom_globals)
        custom_globals = process_globals(custom_globals)
        # Quizas sea mejor definir las dependencias dentro de la propia función para luego cargarla bien
        code = marshal.loads(custom_net)
        custom_net = types.FunctionType(code, custom_globals, "custom_net_func")
        agent_att['net_architecture']['custom_network'] = custom_net

    elif actor_custom_net is not None and critic_custom_net is not None:
        actor_custom_globals = dill.loads(actor_custom_globals)
        actor_custom_globals = process_globals(actor_custom_globals)
        code = marshal.loads(actor_custom_net)
        actor_custom_net = types.FunctionType(code, actor_custom_globals, "actor_custom_net_func")
        agent_att['net_architecture']['actor_custom_network'] = actor_custom_net

        critic_custom_globals = dill.loads(critic_custom_globals)
        critic_custom_globals = process_globals(critic_custom_globals)
        code = marshal.loads(critic_custom_net)
        critic_custom_net = types.FunctionType(code, critic_custom_globals, "critic_custom_net_func")
        agent_att['net_architecture']['critic_custom_network'] = critic_custom_net

    elif common_custom_net is not None and value_custom_net is not None and adv_custom_net is not None:
        common_custom_globals = dill.loads(common_custom_globals)
        common_custom_globals = process_globals(common_custom_globals)
        code = marshal.loads(common_custom_net)
        common_custom_net = types.FunctionType(code, common_custom_globals, "common_custom_net_func")
        agent_att['net_architecture']['common_custom_network'] = common_custom_net

        value_custom_globals = dill.loads(value_custom_globals)
        value_custom_globals = process_globals(value_custom_globals)
        code = marshal.loads(value_custom_net)
        value_custom_net = types.FunctionType(code, value_custom_globals, "value_custom_net_func")
        agent_att['net_architecture']['value_custom_network'] = value_custom_net

        adv_custom_globals = dill.loads(adv_custom_globals)
        adv_custom_globals = process_globals(adv_custom_globals)
        code = marshal.loads(adv_custom_net)
        adv_custom_net = types.FunctionType(code, adv_custom_globals, "adv_custom_net_func")
        agent_att['net_architecture']['action_custom_network'] = adv_custom_net



    tmp_path = 'capoirl_tmp_loading_folder/'

    if not os.path.exists(tmp_path):
        os.makedirs(tmp_path)

    try:
        model_json_bytes = base64.b64decode(data['model_json'])
        with open(tmp_path + 'tmp_model.json', 'wb') as fp:
            fp.write(model_json_bytes)
    except:
        pass
    try:
        model_h5_bytes = base64.b64decode(data['model_h5'])
        with open(tmp_path + 'tmp_model.h5', 'wb') as fp:
            fp.write(model_h5_bytes)
    except:
        pass

    try:
        model_actor_bytes = base64.b64decode(data['actor_h5'])
        with open(tmp_path + 'tmp_modelactor.h5', 'wb') as fp:
            fp.write(model_actor_bytes)
        model_critic_bytes = base64.b64decode(data['critic_h5'])
        with open(tmp_path + 'tmp_modelcritic.h5', 'wb') as fp:
            fp.write(model_critic_bytes)
    except:
        pass

    try:
        checkpoint_bytes = base64.b64decode(data['model_ckpt'])
        with open(tmp_path + 'checkpoint', 'wb') as fp:
            fp.write(checkpoint_bytes)
        index_ckpt_bytes = base64.b64decode(data['model_index'])
        with open(tmp_path + 'tmp_model.index', 'wb') as fp:
            fp.write(index_ckpt_bytes)
        meta_ckpt_bytes = base64.b64decode(data['model_meta'])
        with open(tmp_path + 'tmp_model.meta', 'wb') as fp:
            fp.write(meta_ckpt_bytes)
        data_ckpt_bytes = base64.b64decode(data['model_data'])
        with open(tmp_path + 'tmp_model.data-00000-of-00001', 'wb') as fp:
            fp.write(data_ckpt_bytes)
    except:
        pass

    if agent is None:
        agent = create_new_agent(agent_att)

    set_agent_attributes(agent_att, agent)
    agent._load(tmp_path + 'tmp_model')

    # # Se pone el tamaño del estado a none por que ya se ha utilizado para cargar la red neuronal
    # agent.state_size = None
    shutil.rmtree(tmp_path)

    return agent


def extract_agent_attributes(agent):
    # TODO: save train_action_selection_options and action_selection_options
    try:
        action_low_bound = agent.action_bound[0]
        action_high_bound = agent.action_bound[1]
    except:
        action_low_bound = None
        action_high_bound = None

    custom_net = None
    actor_custom_net = None
    critic_custom_net = None
    common_custom_net = None
    value_custom_net = None
    adv_custom_net = None

    if agent.net_architecture is not None and 'use_custom_network' in agent.net_architecture.keys() \
            and agent.net_architecture['use_custom_network']:
        try:
            custom_net = agent.net_architecture['custom_network']
            agent.net_architecture['custom_network'] = None
        except:
            custom_net = None
        try:
            actor_custom_net = agent.net_architecture['actor_custom_network']
            critic_custom_net = agent.net_architecture['critic_custom_network']
            agent.net_architecture['actor_custom_network'] = None
            agent.net_architecture['critic_custom_network'] = None
        except:
            actor_custom_net = None
            critic_custom_net = None
        try:
            common_custom_net = agent.net_architecture['common_custom_network']
            value_custom_net = agent.net_architecture['value_custom_network']
            adv_custom_net = agent.net_architecture['action_custom_network']
            agent.net_architecture['common_custom_network'] = None
            agent.net_architecture['value_custom_network'] = None
            agent.net_architecture['action_custom_network'] = None
        except:
            value_custom_net = None
            adv_custom_net = None
            common_custom_net = None

        if custom_net is None and actor_custom_net is None and critic_custom_net is None and common_custom_net is None and \
                value_custom_net is None and adv_custom_net is None:
            raise Exception('There are some errors when trying to save the custom network defined by the user')

    dict = {
        'state_size': agent.state_size,
        'env_state_size': agent.env_state_size,
        'n_actions': agent.n_actions,
        'stack': agent.stack,
        'learning_rate': agent.learning_rate,
        'actor_lr': agent.actor_lr,
        'critic_lr': agent.critic_lr,
        'batch_size': agent.batch_size,
        'epsilon': agent.epsilon,
        'epsilon_decay': agent.epsilon_decay,
        'epsilon_min': agent.epsilon_min,
        'gamma': agent.gamma,
        'tau': agent.tau,
        'memory_size': agent.memory_size,
        'loss_clipping': agent.loss_clipping,
        'critic_discount': agent.critic_discount,
        'entropy_beta': agent.entropy_beta,
        'lmbda': agent.lmbda,
        'train_epochs': agent.train_epochs,
        'exploration_noise': agent.exploration_noise,
        'n_step_return': agent.n_step_return,
        'n_stack': agent.n_stack,
        'img_input': agent.img_input,
        'n_threads': agent.n_threads,
        'net_architecture': agent.net_architecture,
        'action_low_bound': action_low_bound,
        'action_high_bound': action_high_bound,
        # 'save_base': agent.save_base,
        # 'save_name': agent.save_name,
        # 'save_each': agent.save_each,
        # 'save_if_better': agent.save_if_better,
        'agent_compiled': agent.agent_builded,
        'agent_name': agent.agent_name,
        'tensorboard_dir': agent.tensorboard_dir
    }
    return dict, custom_net, actor_custom_net, critic_custom_net, common_custom_net, value_custom_net, adv_custom_net


def set_agent_attributes(att, agent, set_all, set_nones=False):
    """
    Asigna los atributos del agente cargados desde fichero.
    :param att: (dict) atributos
    :param agent: (RL_Agent) agente
    :param set_all: (bool) If True, all the attributes will be overwrited with values from att dictionary. If False, set_nones will be taked into account.
    :param set_nones: (bool) If True, onlly the attributes with None value will be setted. If False, everey attribute of the agent will be setted.
    """
    if (set_nones and agent.state_size is None) or set_all:
        agent.state_size = att['state_size']
    if (set_nones and agent.env_state_size is None) or set_all:
        agent.env_state_size = att['env_state_size']
    if (set_nones and agent.n_actions is None) or set_all:
        agent.n_actions = att['n_actions']
    if (set_nones and agent.n_stack is None) or set_all:
        agent.n_stack = att['n_stack']
    if (set_nones and agent.stack is None):
        agent.stack = agent.n_stack > 1
    if set_all:
        agent.stack = att['stack']
    if (set_nones and agent.learning_rate is None) or set_all:
        agent.learning_rate = att['learning_rate']
    if (set_nones and agent.actor_lr is None) or set_all:
        agent.actor_lr = att['actor_lr']
    if (set_nones and agent.critic_lr is None) or set_all:
        agent.critic_lr = att['critic_lr']
    if (set_nones and agent.batch_size is None) or set_all:
        agent.batch_size = att['batch_size']
    if (set_nones and agent.epsilon is None) or set_all:
        agent.epsilon = att['epsilon']
    if (set_nones and agent.epsilon_decay is None) or set_all:
        agent.epsilon_decay = att['epsilon_decay']
    if (set_nones and agent.epsilon_min is None) or set_all:
        agent.epsilon_min = att['epsilon_min']
    if (set_nones and agent.gamma is None) or set_all:
        agent.gamma = att['gamma']
    if (set_nones and agent.tau is None) or set_all:
        agent.tau = att['tau']
    if (set_nones and agent.memory_size is None) or set_all:
        agent.memory_size = att['memory_size']
    if (set_nones and agent.loss_clipping is None) or set_all:
        agent.loss_clipping = att['loss_clipping']
    if (set_nones and agent.critic_discount is None) or set_all:
        agent.critic_discount = att['critic_discount']
    if (set_nones and agent.entropy_beta is None) or set_all:
        agent.entropy_beta = att['entropy_beta']
    if (set_nones and agent.lmbda is None) or set_all:
        agent.lmbda = att['lmbda']
    if (set_nones and agent.train_epochs is None) or set_all:
        agent.train_epochs = att['train_epochs']
    if (set_nones and agent.exploration_noise is None) or set_all:
        agent.exploration_noise = att['exploration_noise']
    if (set_nones and agent.n_step_return is None) or set_all:
        agent.n_step_return = att['n_step_return']
    if (set_nones and agent.img_input is None) or set_all:
        agent.img_input = att['img_input']
    if (set_nones and agent.n_threads is None) or set_all:
        agent.n_threads = att['n_threads']
    if (set_nones and agent.net_architecture is None) or set_all:
        agent.net_architecture = att['net_architecture']
    if (set_nones and hasattr(agent, 'action_bound') and agent.action_bound is None) or set_all:
        agent.action_bound = [att['action_low_bound'], att['action_high_bound']]
    # TODO: estos atributos no son necesarios en la nueva versión
    # agent.save_base = att['save_base']
    # agent.save_name = att['save_name']
    # agent.save_each = att['save_each']
    # agent.save_if_better = att['save_if_better']
    # TODO: agent_builded no lo cargo por que solo será true una vez que se haya contruido el modelo. Al cargarlo no esta todavía construido.
    # if (set_nones and agent.agent_builded is None) or set_all:
    #     agent.agent_builded = att['agent_compiled']
    # TODO: agent_name no lo cargo por que siempre lo tiene que tener el agente
    # agent.agent_name = att['agent_name']
    if (set_nones and agent.tensorboard_dir is None) or set_all:
        agent.tensorboard_dir = att['tensorboard_dir']
    return agent


def create_new_agent(att):
    if att["agent_name"] == agent_globals.names["dqn"]:
        from RL_Agent.legacy_agents import dqn_agent
        agent = dqn_agent.Agent()
    elif att["agent_name"] == agent_globals.names["ddqn"]:
        from RL_Agent.legacy_agents import ddqn_agent
        agent = ddqn_agent.Agent()
    elif att["agent_name"] == agent_globals.names["dddqn"]:
        from RL_Agent.legacy_agents import dddqn_agent
        agent = dddqn_agent.Agent()
    elif att["agent_name"] == agent_globals.names["dpg"]:
        from RL_Agent.legacy_agents import dpg_agent
        agent = dpg_agent.Agent()
    elif att["agent_name"] == agent_globals.names["ddpg"]:
        from RL_Agent.legacy_agents import ddpg_agent
        agent = ddpg_agent.Agent()
    elif att["agent_name"] == agent_globals.names["a2c_discrete"]:
        from RL_Agent.legacy_agents import a2c_agent_discrete
        agent = a2c_agent_discrete.Agent()
    elif att["agent_name"] == agent_globals.names["a2c_continuous"]:
        from RL_Agent.legacy_agents import a2c_agent_continuous
        agent = a2c_agent_continuous.Agent()
    elif att["agent_name"] == agent_globals.names["a2c_discrete_queue"]:
        from RL_Agent.legacy_agents import a2c_agent_discrete_queue
        agent = a2c_agent_discrete_queue.Agent()
    elif att["agent_name"] == agent_globals.names["a2c_continuous_queue"]:
        from RL_Agent.legacy_agents import a2c_agent_continuous_queue
        agent = a2c_agent_continuous_queue.Agent()
    elif att["agent_name"] == agent_globals.names["a3c_discrete"]:
        from RL_Agent.legacy_agents import a3c_agent_discrete
        agent = a3c_agent_discrete.Agent()
    elif att["agent_name"] == agent_globals.names["a3c_continuous"]:
        from RL_Agent.legacy_agents import a3c_agent_continuous
        agent = a3c_agent_continuous.Agent()
    elif att["agent_name"] == agent_globals.names["ppo_discrete"]:
        from RL_Agent.legacy_agents import ppo_agent_discrete
        agent = ppo_agent_discrete.Agent()
    elif att["agent_name"] == agent_globals.names["ppo_discrete_multithread"]:
        from RL_Agent.legacy_agents import ppo_agent_discrete_parallel
        agent = ppo_agent_discrete_parallel.Agent()
    elif att["agent_name"] == agent_globals.names["ppo_continuous"]:
        from RL_Agent.legacy_agents import ppo_agent_continuous
        agent = ppo_agent_continuous.Agent()
    elif att["agent_name"] == agent_globals.names["ppo_continuous_multithread"]:
        from RL_Agent.legacy_agents import ppo_agent_continuous_parallel
        agent = ppo_agent_continuous_parallel.Agent()
    set_agent_attributes(att, agent)

    if att['action_low_bound'] is None and att['action_high_bound'] is None:
        agent.build_agent(state_size=att["state_size"], n_actions=att["n_actions"], stack=att["stack"])
    else:
        agent.build_agent(state_size=att["state_size"], n_actions=att["n_actions"], stack=att["stack"],
                          action_bound=[att['action_low_bound'], att['action_high_bound']])
    return agent

def process_globals(custom_globals):
    globs = globals()
    for key in globs:
        for cust_key in custom_globals:
            if key == cust_key:
                custom_globals[cust_key] = globs[key]
                break
    return custom_globals