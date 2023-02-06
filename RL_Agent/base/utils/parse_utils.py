def parse_model_params(model_params):
    batch_size = model_params.get("batch_size")
    epsilon = model_params.get("epsilon")
    epsilon_min = model_params.get("epsilon_min")
    epsilon_decay = model_params.get("epsilon_decay")
    learning_rate = model_params.get("learning_rate")
    n_step_return = model_params.get("n_step_return")
    game = model_params.get("a3c_game")

    if game is None:
        return batch_size, epsilon, epsilon_min, epsilon_decay, learning_rate, n_step_return
    else:  # This case is for A3C
        return batch_size, epsilon, epsilon_min, epsilon_decay, learning_rate, n_step_return, game


def parse_saving_model_params(params):
    base = params.get("base")
    name = params.get("name")
    save_each = params.get("save_each")
    save_if_better = params.get("save_if_better")

    if base is None:
        base = "/home/shernandez/PycharmProjects/CAPORL_full_project/saved_models/"
    if save_each is None:
        save_each = 500
    if save_if_better is None:
        save_if_better = False

    return base, name, save_each, save_if_better

def parse_discriminator_params(irl_params):
    lr_disc = irl_params.get("lr_disc")
    batch_size_disc = irl_params.get("batch_size_disc")
    epochs_disc = irl_params.get("epochs_disc")
    val_split_disc = irl_params.get("val_split_disc")
    agent_collect_iter = irl_params.get("agent_collect_iter")
    agent_train_iter = irl_params.get("agent_train_iter")

    return lr_disc, batch_size_disc, epochs_disc, val_split_disc, agent_collect_iter, agent_train_iter

