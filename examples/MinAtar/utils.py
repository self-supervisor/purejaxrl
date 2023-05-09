from typing import NamedTuple, Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper


class HActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, transformed_x):
        # x /= jnp.array([10, 1, 1, 1])
        # transformed_x /= jnp.array([10, 1, 1, 1])
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(transformed_x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


def build_forwards_backwards_model():
    import torch

    forward_model = TransitionModel()
    forward_model_params = forward_model.init(jax.random.PRNGKey(0), jnp.ones((1, 425)))

    forward_weights = torch.load("forward.pt")

    from flax.core.frozen_dict import freeze

    forward_model_params = forward_model_params.unfreeze()
    assert (
        forward_model_params["params"]["Dense_0"]["kernel"].shape
        == forward_weights["fc1.weight"].T.shape
    )
    forward_model_params["params"]["Dense_0"]["kernel"] = (
        forward_weights["fc1.weight"].T.cpu().detach().numpy()
    )
    assert (
        forward_model_params["params"]["Dense_1"]["kernel"].shape
        == forward_weights["fc2.weight"].T.shape
    )
    forward_model_params["params"]["Dense_1"]["kernel"] = (
        forward_weights["fc2.weight"].T.cpu().detach().numpy()
    )
    assert (
        forward_model_params["params"]["Dense_2"]["kernel"].shape
        == forward_weights["fc4.weight"].T.shape
    )
    forward_model_params["params"]["Dense_2"]["kernel"] = (
        forward_weights["fc4.weight"].T.cpu().detach().numpy()
    )

    assert (
        forward_model_params["params"]["Dense_0"]["bias"].shape
        == forward_weights["fc1.bias"].shape
    )
    forward_model_params["params"]["Dense_0"]["bias"] = (
        forward_weights["fc1.bias"].cpu().detach().numpy()
    )
    assert (
        forward_model_params["params"]["Dense_1"]["bias"].shape
        == forward_weights["fc2.bias"].shape
    )
    forward_model_params["params"]["Dense_1"]["bias"] = (
        forward_weights["fc2.bias"].cpu().detach().numpy()
    )
    assert (
        forward_model_params["params"]["Dense_2"]["bias"].shape
        == forward_weights["fc4.bias"].shape
    )
    forward_model_params["params"]["Dense_2"]["bias"] = (
        forward_weights["fc4.bias"].cpu().detach().numpy()
    )
    forward_model_params = freeze(forward_model_params)

    backward_model = TransitionModel()
    # init flax model
    backward_model_params = backward_model.init(
        jax.random.PRNGKey(0), jnp.ones((1, 425))
    )

    backward_weights = torch.load("backward.pt")

    from flax.core.frozen_dict import freeze

    backward_model_params = backward_model_params.unfreeze()
    assert (
        backward_model_params["params"]["Dense_0"]["kernel"].shape
        == backward_weights["fc1.weight"].T.shape
    )
    backward_model_params["params"]["Dense_0"]["kernel"] = (
        backward_weights["fc1.weight"].T.cpu().detach().numpy()
    )
    assert (
        backward_model_params["params"]["Dense_1"]["kernel"].shape
        == backward_weights["fc2.weight"].T.shape
    )
    backward_model_params["params"]["Dense_1"]["kernel"] = (
        backward_weights["fc2.weight"].T.cpu().detach().numpy()
    )
    assert (
        backward_model_params["params"]["Dense_2"]["kernel"].shape
        == backward_weights["fc4.weight"].T.shape
    )
    backward_model_params["params"]["Dense_2"]["kernel"] = (
        backward_weights["fc4.weight"].T.cpu().detach().numpy()
    )

    assert (
        backward_model_params["params"]["Dense_0"]["bias"].shape
        == backward_weights["fc1.bias"].shape
    )
    backward_model_params["params"]["Dense_0"]["bias"] = (
        backward_weights["fc1.bias"].cpu().detach().numpy()
    )
    assert (
        backward_model_params["params"]["Dense_1"]["bias"].shape
        == backward_weights["fc2.bias"].shape
    )
    backward_model_params["params"]["Dense_1"]["bias"] = (
        backward_weights["fc2.bias"].cpu().detach().numpy()
    )
    assert (
        backward_model_params["params"]["Dense_2"]["bias"].shape
        == backward_weights["fc4.bias"].shape
    )
    backward_model_params["params"]["Dense_2"]["bias"] = (
        backward_weights["fc4.bias"].cpu().detach().numpy()
    )
    backward_model_params = freeze(backward_model_params)
    return forward_model, forward_model_params, backward_model, backward_model_params


def equivalent_state_with_model(
    forward_model,
    forward_model_params,
    backward_model,
    backward_model_params,
    obs,
    action,
):
    # inputs = jnp.concatenate(
    #     [
    #         obs,
    #         jnp.repeat(jax.nn.one_hot(action, num_classes=5), repeats=5).reshape(
    #             -1, 25
    #         ),
    #     ],
    #     axis=1,
    # )
    # pred_next_state = forward_model.apply(forward_model_params, inputs)
    # pred_next_state = jnp.concatenate(
    #     [
    #         pred_next_state,
    #         jnp.repeat(
    #             jnp.repeat(jnp.array([1, 0, 0, 0, 0],), repeats=5).reshape(-1, 25),
    #             repeats=pred_next_state.shape[0],
    #             axis=0,
    #         ),
    #     ],
    #     axis=1,
    # )
    # equivalent_state = backward_model.apply(backward_model_params, pred_next_state)
    # return equivalent_state

    # import jax.numpy as jnp

    # jax.debug.breakpoint()
    # obs = inputs[:, :400]
    # action = jnp.argmax(inputs[:, 400:405], axis=1)
    # jax.debug.print("action {action}", action=action)
    obs = obs.reshape(-1, 4, 10, 10)
    equivalent_obs_0 = obs.copy()
    equivalent_obs_1 = obs.copy()
    equivalent_obs_1.at[:, 0, :, :].set(
        jnp.concatenate(
            (
                (equivalent_obs_1[:, 0, :, :][:, :, 1:]),
                (equivalent_obs_1[:, 0, :, :][:, :, :1]),
            ),
            axis=2,
        )
    )
    equivalent_obs_2 = obs.copy()
    equivalent_obs_2.at[:, 0, :, :].set(
        jnp.concatenate(
            (
                (equivalent_obs_2[:, 0, :, :][:, 1:, :]),
                (equivalent_obs_2[:, 0, :, :][:, :1, :]),
            ),
            axis=1,
        )
    )
    equivalent_obs_3 = obs.copy()
    equivalent_obs_3.at[:, 0, :, :].set(
        jnp.concatenate(
            (
                (equivalent_obs_3[:, 0, :, :][:, :, -1:]),
                (equivalent_obs_3[:, 0, :, :][:, :, :-1]),
            ),
            axis=2,
        )
    )
    equivalent_obs_4 = obs.copy()
    equivalent_obs_4.at[:, 0, :, :].set(
        jnp.concatenate(
            (
                (equivalent_obs_4[:, 0, :, :][:, -1:, :]),
                (equivalent_obs_4[:, 0, :, :][:, :-1, :]),
            ),
            axis=1,
        )
    )
    equivalent_obs = (
        equivalent_obs_0 * jax.nn.one_hot(action, 5)[:, 0].reshape(-1, 1, 1, 1)
        + equivalent_obs_1 * jax.nn.one_hot(action, 5)[:, 1].reshape(-1, 1, 1, 1)
        + equivalent_obs_2 * jax.nn.one_hot(action, 5)[:, 2].reshape(-1, 1, 1, 1)
        + equivalent_obs_3 * jax.nn.one_hot(action, 5)[:, 3].reshape(-1, 1, 1, 1)
        + equivalent_obs_4 * jax.nn.one_hot(action, 5)[:, 4].reshape(-1, 1, 1, 1)
    )
    equivalent_obs = equivalent_obs.reshape(equivalent_obs.shape[0], -1)
    return equivalent_obs
    # action = jax.nn.one_hot(action, 5)
    # equivalent_obs = (
    #     action[:, 0] * equivalent_obs_action_0
    #     + action[:, 1] * equivalent_obs_1
    #     # + action[:, 2] * equivalent_obs_2
    #     # bs
    #     # + action[:, 3] * equivalent_obs_3
    #     # + action[:, 4] * equivalent_obs_4
    # )
    # import jax.numpy as jnp
    # obs = obs.reshape(-1, 4, 10, 10)
    # jax.debug.print("obs {obs}", obs=obs[0].reshape(4, 10, 10)[0])
    # jax.debug.breakpoint()
    #
    # _, player_ys, player_xs = obs.nonzero()
    # jax.debug.print("player_ys {player_ys}", player_ys=player_ys)
    # jnp.where(player_ys == 0, 9, player_ys - 1)
    # gymnax code
    # player_xs = (
    #     jnp.maximum(0, player_xs - 1) * (action == 1)  # l
    #     + jnp.minimum(9, player_ys + 1) * (action == 3)  # r
    #     + player_xs * jnp.logical_and(action != 1, action != 3)
    # )  # others

    # player_ys = (
    #     jnp.maximum(1, player_ys - 1) * (action == 2)  # u
    #     + jnp.minimum(8, player_ys + 1) * (action == 4)  # d
    #     + player_ys * jnp.logical_and(action != 2, action != 4)
    # )  # others
    # obs = obs.reshape(obs.shape[0], -1)

    # # jax.debug.breakpoint()
    # # jax.debug.print("action shape {action}", action=action.shape)
    # return obs


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        # if len(x.shape) == 1:
        #     x = jnp.reshape(x, (-1,))
        # else:
        #     x = jnp.reshape(x, (x.shape[0], -1))
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        pi = distrax.Categorical(logits=actor_mean)

        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


# class TransitionModel(nn.Module):
#     state_dim: Sequence[int]

#     @nn.compact
#     def __call__(self, inputs):
#         # if len(inputs.shape) == 1:
#         #     inputs /= jnp.array([10, 1, 1, 1, 1])
#         #     # inputs /= jnp.array([10, 1, 1, 1, 1])
#         #     x_0 = nn.Dense(self.state_dim)(inputs[: -1])
#         #     x_1 = nn.Dense(self.state_dim)(inputs[: -1])
#         #     x_1 *= inputs[-1:] # if zero == 0, if one == 1
#         #     x_0 *= (inputs[-1:] - 1) ** 2 # if zero == 1, if one == 0
#         #     pred = x_0 + x_1
#         # else:
#         #     inputs /= jnp.array([10, 1, 1, 1, 1])
#         #     # inputs /= jnp.array([10, 1, 1, 1, 1])
#         #     x_0 = nn.Dense(self.state_dim)(inputs[:, : -1])
#         #     x_1 = nn.Dense(self.state_dim)(inputs[:, : -1])
#         #     x_1 *= inputs[:, -1:] # if zero == 0, if one == 1
#         #     x_0 *= (inputs[:, -1:] - 1) ** 2 # if zero == 1, if one == 0
#         #     pred = x_0 + x_1
#         # inputs /= jnp.array([10, 1, 1, 1, 1])
#         x = nn.relu(nn.Dense(256)(inputs))
#         x = nn.relu(nn.Dense(256)(x))
#         x = nn.relu(nn.Dense(256)(x))
#         pred = nn.Dense(self.state_dim)(x)
#         return pred


# class TransitionModel(nn.Module):
#     @nn.compact
#     def __call__(self, inputs):
#         if len(inputs.shape) == 1:
#             # inputs /= jnp.array([10, 1, 1, 1, 1])
#             # inputs /= jnp.array([10, 1, 1, 1, 1])
#             x_0 = nn.Dense(4)(inputs[:-1])
#             x_1 = nn.Dense(4)(inputs[:-1])
#             x_1 *= inputs[-1:]  # if zero == 0, if one == 1
#             x_0 *= (inputs[-1:] - 1) ** 2  # if zero == 1, if one == 0
#             pred = x_0 + x_1
#         else:
#             # inputs /= jnp.array([10, 1, 1, 1, 1])
#             # inputs /= jnp.array([10, 1, 1, 1, 1])
#             x_0 = nn.Dense(4)(inputs[:, :-1])
#             x_1 = nn.Dense(4)(inputs[:, :-1])
#             x_1 *= inputs[:, -1:]  # if zero == 0, if one == 1
#             x_0 *= (inputs[:, -1:] - 1) ** 2  # if zero == 1, if one == 0
#             pred = x_0 + x_1
#         # inputs /= jnp.array([10, 1, 1, 1, 1])
#         # x = nn.relu(nn.Dense(256)(inputs))
#         # x = nn.relu(nn.Dense(256)(x))
#         # x = nn.relu(nn.Dense(256)(x))
#         # pred = nn.Dense(4)(x)
#         return pred


class TransitionModel(nn.Module):
    @nn.compact
    def __call__(self, inputs):
        x = nn.relu(nn.Dense(4096)(inputs))
        x = nn.relu(nn.Dense(4096)(x))
        x = nn.Dense(400)(x)
        x = nn.sigmoid(x)
        return x


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: jnp.ndarray


def make_env(env_name):
    env, env_params = gymnax.make("Asterix-MinAtar")
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)
    return env, env_params


def make_initialised_model(env, env_params, rng: jax.random.PRNGKey):
    network = ActorCritic(env.action_space(env_params).n, activation="relu")
    init_x = jnp.zeros(env.observation_space(env_params).shape)
    network_params = network.init(rng, init_x)
    return network, network_params


def replace_params_with_trained_params(network, random_params, trained_params):
    from flax.core import freeze, unfreeze

    random_params = unfreeze(random_params)
    for first_key in random_params["params"].keys():
        for second_key in random_params["params"][first_key].keys():
            assert (
                random_params["params"][first_key][second_key].shape
                == trained_params["params"][first_key][second_key][0][0].shape
            ), f"Shape mismatch for {first_key}/{second_key}: {random_params['params'][first_key][second_key].shape} vs {trained_params['params'][first_key][second_key][0][0].shape}"
            random_params["params"][first_key][second_key] = trained_params["params"][
                first_key
            ][second_key][0][0]

    random_params = freeze(random_params)
    return random_params


def rollout(rng_input, network, policy_params, env, env_params, steps_in_episode=1000):
    """from gymnax examples: Rollout a jitted gymnax episode with lax.scan."""
    # Reset the environment
    rng_reset, rng_episode = jax.random.split(rng_input)
    obs, state = env.reset(rng_reset, env_params)

    def policy_step(state_input, tmp):
        """lax.scan compatible step transition in jax env."""
        obs, state, policy_params, rng = state_input
        rng, rng_step, rng_net = jax.random.split(rng, 3)
        dist, value = network.apply(policy_params, obs)
        action = dist.sample(seed=rng_net)
        next_obs, next_state, reward, done, _ = env.step(
            rng_step, state, action, env_params
        )
        carry = [next_obs, next_state, policy_params, rng]
        return carry, [obs, action, reward, next_obs, done]

    # Scan over episode step loop
    _, scan_out = jax.lax.scan(
        policy_step, [obs, state, policy_params, rng_episode], (), steps_in_episode
    )
    # Return masked sum of rewards accumulated by agent in episode
    obs, action, reward, next_obs, done = scan_out
    return obs, action, reward, next_obs, done


def collect_episodes(network, params, num_episodes, rng, env, env_params):
    rng_batch = jax.random.split(rng, num_episodes)
    batch_rollout = jax.vmap(rollout, in_axes=(0, None, None, None, None))
    outputs = batch_rollout(rng_batch, network, params, env, env_params)
    all_actions = np.array(outputs[1].reshape(-1))
    all_states = np.array(outputs[0].reshape(-1, 400))
    all_next_states = np.array(outputs[3].reshape(-1, 400))
    return all_states, all_actions, all_next_states


def make_train_data(
    rng: jax.random.PRNGKey,
    num_episodes: int,
    params_path: str = "params.pkl",
    env_name: str = "Asterix-MinAtar",
):
    import datetime

    env, env_params = make_env(env_name=env_name)
    trained_params = load_pkl_object(params_path)["params"]
    network, random_params = make_initialised_model(env, env_params, rng)
    params = replace_params_with_trained_params(network, random_params, trained_params)
    all_states, all_actions, all_next_states = collect_episodes(
        network, params, num_episodes, rng, env, env_params
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    np.savez(
        f"model_data/train_data_{timestamp}.npz",
        states=all_states,
        actions=all_actions,
        next_states=all_next_states,
    )


# from gymnax code
def save_pkl_object(obj, filename):
    """Helper to store pickle objects."""
    import pickle
    from pathlib import Path

    output_file = Path(filename)
    output_file.parent.mkdir(exist_ok=True, parents=True)

    with open(filename, "wb") as output:
        # Overwrites any existing file.
        pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)

    print(f"Stored data at {filename}.")


def load_pkl_object(filename: str):
    """Helper to reload pickle objects."""
    import pickle

    with open(filename, "rb") as input:
        obj = pickle.load(input)
    print(f"Loaded data from {filename}.")
    return obj
