import jax

print("jax devices", jax.devices())
import jax.numpy as jnp
import time
from h_train import make_train
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
import wandb


def generate_combinations(
    lr,
    ent_coef,
    transition_model_lr,
    max_grad_norm,
    schedule_accelerator,
    num_envs,
    num_minibatches,
    clip_eps,
):
    """
    from chatGPT
    """
    # Generate all possible combinations of lr, ent_coef, and vf_coef
    combinations = np.array(
        list(
            product(
                lr,
                ent_coef,
                transition_model_lr,
                max_grad_norm,
                schedule_accelerator,
                num_envs,
                num_minibatches,
                clip_eps,
            )
        )
    )

    # Split the combinations into separate arrays
    lr_combinations = combinations[:, 0]
    ent_coef_combinations = combinations[:, 1]
    transition_model_lr_combinations = combinations[:, 2]
    max_grad_norm_combinations = combinations[:, 3]
    schedule_accelerator_combinations = combinations[:, 4]
    num_envs_combinations = combinations[:, 5]
    num_minibatches_combinations = combinations[:, 6]
    clips_eps_combinations = combinations[:, 7]

    # Return a tuple of the resulting arrays
    return (
        lr_combinations,
        ent_coef_combinations,
        transition_model_lr_combinations,
        max_grad_norm_combinations,
        schedule_accelerator_combinations,
        num_envs_combinations,
        num_minibatches_combinations,
        clips_eps_combinations,
    )


def main(config):
    group = wandb.util.generate_id()

    print("config", config)
    ent_coef_search = [0.001]
    # ent_coef_search = [config["ENT_COEF"]]
    lr_search = [0.0025]  # , 0.00025, 0.000025]
    # lr_search = [config["LR"]]
    transition_model_lr_search = [1e-4]
    max_grad_norm_search = [0.5]  # [5, 0.5, 0.05]
    # max_grad_norm_search = [config["MAX_GRAD_NORM"]]
    schedule_accelerator_search = [1.0]
    num_envs = [config["NUM_ENVS"]]
    num_minibatches = [config["NUM_MINIBATCHES"]]
    clip_eps = [0.02]  # [0.02, 0.2, 2]
    # clip_eps = [config["CLIP_EPS"]]
    (
        lr_combinations,
        ent_coef_combinations,
        transition_model_lr_combinations,
        max_grad_norm_combinations,
        schedule_accelerator_combinations,
        num_envs_combinations,
        num_minibatches_combinations,
        clips_eps_combinations,
    ) = generate_combinations(
        lr_search,
        ent_coef_search,
        transition_model_lr_search,
        max_grad_norm_search,
        schedule_accelerator_search,
        num_envs,
        num_minibatches,
        clip_eps,
    )
    lr_combinations = jnp.array(lr_combinations)
    ent_coef_combinations = jnp.array(ent_coef_combinations)
    transition_model_lr_combinations = jnp.array(transition_model_lr_combinations)
    max_grad_norm_combinations = jnp.array(max_grad_norm_combinations)
    schedule_accelerator_combinations = jnp.array(schedule_accelerator_combinations)
    num_envs_combinations = jnp.array(num_envs_combinations)
    num_minibatches_combinations = jnp.array(num_minibatches_combinations)
    clips_eps_combinations = jnp.array(clips_eps_combinations)
    combinations = [
        lr_combinations,
        ent_coef_combinations,
        transition_model_lr_combinations,
        max_grad_norm_combinations,
        schedule_accelerator_combinations,
        num_envs_combinations.astype(int),
        num_minibatches_combinations.astype(int),
        clips_eps_combinations,
    ]

    NUMBER_OF_SEEDS = 25
    # num_minibatches_combinations = jnp.ones([81,], dtype=jnp.int32) * 2

    rng = jax.random.PRNGKey(NUMBER_OF_SEEDS * len(combinations))
    rngs = jax.random.split(rng, NUMBER_OF_SEEDS)

    train_vvjit = jax.jit(
        jax.vmap(jax.vmap(make_train(config), in_axes=(None, 0)), in_axes=(0, None))
    )
    t0 = time.time()
    outs = jax.block_until_ready(train_vvjit(combinations, rngs))
    print(f"time: {time.time() - t0:.2f} s")

    dict_outs = {}
    combinations = jnp.stack(combinations, axis=1)
    for i in range(len(combinations)):
        (
            lr,
            ent_coef,
            transition_lr,
            max_grad_norm,
            schedule_accelerator,
            num_envs,
            num_minibatches,
            clip_eps,
        ) = combinations[i]
        to_plot_ent_coef = str(round(ent_coef, 4))
        to_plot_lr = str(round(lr, 4))
        to_plot_transition_model_lr = str(round(transition_lr, 4))
        to_plot_max_grad_norm = str(round(max_grad_norm, 4))
        to_plot_schedule_accelerator = str(round(schedule_accelerator, 4))
        to_plot_num_envs = str(round(num_envs, 4))
        to_plot_num_minibatches = str(round(num_minibatches, 4))
        to_plot_clip_eps = str(round(clip_eps, 4))
        new_config = {
            "LR": to_plot_lr,
            "TRANSITION_MODEL_LR": to_plot_transition_model_lr,
            "ENT_COEF": to_plot_ent_coef,
            "MAX_GRAD_NORM": to_plot_max_grad_norm,
            "SCHEDULE_ACCELERATOR": to_plot_schedule_accelerator,
            "NUM_ENVS": to_plot_num_envs,
            "NUM_MINIBATCHES": to_plot_num_minibatches,
            "CLIP_EPS": to_plot_clip_eps,
        }
        config.update(new_config)

        # wandb.init(
        #     project="purejaxrl", entity="self-supervisor", config=config, group=group
        # )
        # list_to_log = [
        #     j.item()
        #     for j in outs["metrics"]["returned_episode_returns"][i]
        #     .mean(0)
        #     .mean(-1)
        #     .reshape(-1)
        # ]
        # for a_val_to_log in list_to_log:
        #     wandb.log({"episode_returns": a_val_to_log})
        # wandb.finish()

        plt.plot(
            outs["metrics"]["returned_episode_returns"][i].mean(0).mean(-1).reshape(-1),
        )
        dict_outs[
            f"ent_coef={to_plot_ent_coef}, lr={to_plot_lr}, transition_lr={to_plot_transition_model_lr}, max_grad_norm={to_plot_max_grad_norm}, schedule_accelerator={to_plot_schedule_accelerator}, num_envs={to_plot_num_envs}, num_minibatches={to_plot_num_minibatches}, clip_eps={to_plot_clip_eps}"
        ] = round(
            outs["metrics"]["returned_episode_returns"][i]
            .mean(0)
            .mean(-1)
            .reshape(-1)[:-1000]
            .mean()
            .item(),
            1,
        )
    plt.savefig("hyperparam_search.png")
    plt.close()
    dict_outs = {
        k: v
        for k, v in sorted(
            dict_outs.items(), key=lambda item: np.mean(item[1]), reverse=True
        )
    }
    headers = [
        "return",
        "ent_coef",
        "lr",
        "transition_lr",
        "max_grad_norm",
        "schedule_accelerator",
        "num_envs",
        "num_minibatches",
        "clip_eps",
    ]
    print("|".join(headers))
    print("-" * (len(headers) * 12))
    for key, value in dict_outs.items():
        (
            ent_coef,
            lr,
            transition_lr,
            max_grad_norm,
            schedule_accelerator,
            num_envs,
            num_minibatches,
            clip_eps,
        ) = key.split(", ")
        row_values = [
            "{:.2f}".format(value),
            ent_coef.split("=")[1],
            lr.split("=")[1],
            transition_lr.split("=")[1],
            max_grad_norm.split("=")[1],
            schedule_accelerator.split("=")[1],
            num_envs.split("=")[1],
            clip_eps.split("=")[1],
        ]
        print("|".join(row_values))


if __name__ == "__main__":
    import argparse

    from config import config

    parser = argparse.ArgumentParser()
    parser.add_argument("--LR", type=float, default=config["LR"])
    parser.add_argument("--NUM_ENVS", type=int, default=config["NUM_ENVS"])
    parser.add_argument("--NUM_STEPS", type=int, default=config["NUM_STEPS"])
    parser.add_argument(
        "--TOTAL_TIMESTEPS", type=int, default=config["TOTAL_TIMESTEPS"]
    )
    parser.add_argument("--UPDATE_EPOCHS", type=int, default=config["UPDATE_EPOCHS"])
    parser.add_argument(
        "--NUM_MINIBATCHES", type=int, default=config["NUM_MINIBATCHES"]
    )
    parser.add_argument("--GAMMA", type=float, default=config["GAMMA"])
    parser.add_argument("--GAE_LAMBDA", type=float, default=config["GAE_LAMBDA"])
    parser.add_argument("--CLIP_EPS", type=float, default=config["CLIP_EPS"])
    parser.add_argument("--ENT_COEF", type=float, default=config["ENT_COEF"])
    parser.add_argument("--VF_COEF", type=float, default=config["VF_COEF"])
    parser.add_argument("--MAX_GRAD_NORM", type=float, default=config["MAX_GRAD_NORM"])
    parser.add_argument("--ACTIVATION", type=str, default=config["ACTIVATION"])
    parser.add_argument("--ENV_NAME", type=str, default=config["ENV_NAME"])
    parser.add_argument("--ANNEAL_LR", type=bool, default=config["ANNEAL_LR"])
    parser.add_argument(
        "--TRANSITION_MODEL_LR", type=float, default=config["TRANSITION_MODEL_LR"]
    )
    args = parser.parse_args()
    config.update(vars(args))

    main(config=config)
