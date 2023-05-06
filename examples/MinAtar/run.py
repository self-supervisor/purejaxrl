import jax

from train import make_train


def main(config):
    rng = jax.random.PRNGKey(42)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
    import time

    import matplotlib.pyplot as plt

    rng = jax.random.PRNGKey(42)
    t0 = time.time()
    out = jax.block_until_ready(train_jit(rng))
    print(f"time: {time.time() - t0:.2f} s")
    plt.plot(out["metrics"]["returned_episode_returns"].mean(-1).reshape(-1))
    plt.savefig("return.png")


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
    args = parser.parse_args()
    config.update(vars(args))

    main(config=config)
