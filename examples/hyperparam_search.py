import jax

print("jax devices", jax.devices())
import jax.numpy as jnp
import time
from train import make_train
import matplotlib.pyplot as plt
import numpy as np
from itertools import product


def generate_combinations(
    lr, ent_coef, transition_model_lr, max_grad_norm, schedule_accelerator
):
    """
    from chatGPT
    """
    # Generate all possible combinations of lr, ent_coef, and vf_coef
    combinations = np.array(
        list(
            product(
                lr, ent_coef, transition_model_lr, max_grad_norm, schedule_accelerator
            )
        )
    )

    # Split the combinations into separate arrays
    lr_combinations = combinations[:, 0]
    ent_coef_combinations = combinations[:, 1]
    transition_model_lr_combinations = combinations[:, 2]
    max_grad_norm_combinations = combinations[:, 3]
    schedule_accelerator_combinations = combinations[:, 4]

    # Return a tuple of the resulting arrays
    return (
        lr_combinations,
        ent_coef_combinations,
        transition_model_lr_combinations,
        max_grad_norm_combinations,
        schedule_accelerator_combinations,
    )


def main(config):
    ent_coef_search = [0.01]
    lr_search = [0.00025]
    transition_model_lr_search = [0.0001]
    max_grad_norm_search = [0.5]
    schedule_accelerator_search = [1.0]
    (
        lr_combinations,
        ent_coef_combinations,
        transition_model_lr_combinations,
        max_grad_norm_combinations,
        schedule_accelerator_combinations,
    ) = generate_combinations(
        lr_search,
        ent_coef_search,
        transition_model_lr_search,
        max_grad_norm_search,
        schedule_accelerator_search,
    )
    lr_combinations = jnp.array(lr_combinations)
    ent_coef_combinations = jnp.array(ent_coef_combinations)
    transition_model_lr_combinations = jnp.array(transition_model_lr_combinations)
    max_grad_norm_combinations = jnp.array(max_grad_norm_combinations)
    schedule_accelerator_combinations = jnp.array(schedule_accelerator_combinations)
    combinations = jnp.stack(
        [
            lr_combinations,
            ent_coef_combinations,
            transition_model_lr_combinations,
            max_grad_norm_combinations,
            schedule_accelerator_combinations,
        ],
        axis=1,
    )
    NUMBER_OF_SEEDS = 10

    rng = jax.random.PRNGKey(NUMBER_OF_SEEDS * len(combinations))
    rngs = jax.random.split(rng, NUMBER_OF_SEEDS)

    train_vvjit = jax.jit(
        jax.vmap(jax.vmap(make_train(config), in_axes=(None, 0)), in_axes=(0, None))
    )
    t0 = time.time()
    outs = jax.block_until_ready(train_vvjit(combinations, rngs))
    print(f"time: {time.time() - t0:.2f} s")

    dict_outs = {}
    for i, value in enumerate(combinations):
        lr, ent_coef, transition_lr, max_grad_norm, schedule_accelerator = value
        to_plot_ent_coef = str(round(ent_coef, 4))
        to_plot_lr = str(round(lr, 4))
        to_plot_transition_model_lr = str(round(transition_lr, 4))
        to_plot_max_grad_norm = str(round(max_grad_norm, 4))
        to_plot_schedule_accelerator = str(round(schedule_accelerator, 4))
        plt.plot(
            outs["metrics"]["returned_episode_returns"][i].mean(0).mean(-1).reshape(-1),
        )
        dict_outs[
            f"ent_coef={to_plot_ent_coef}, lr={to_plot_lr}, transition_lr={to_plot_transition_model_lr}, max_grad_norm={to_plot_max_grad_norm}, schedule_accelerator={to_plot_schedule_accelerator}"
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
    ]
    print("|".join(headers))
    print("-" * (len(headers) * 12))
    for key, value in dict_outs.items():
        ent_coef, lr, transition_lr, max_grad_norm, schedule_accelerator = key.split(
            ", "
        )
        row_values = [
            "{:.2f}".format(value),
            ent_coef.split("=")[1],
            lr.split("=")[1],
            transition_lr.split("=")[1],
            max_grad_norm.split("=")[1],
            schedule_accelerator.split("=")[1],
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
