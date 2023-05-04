import jax

print("jax devices", jax.devices())
import gymnax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper

from utils import ActorCritic, Transition, TransitionModel


def make_train(config):
    env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(env)
    env = LogWrapper(env)

    def train(combinations=None, rng=None):
        if combinations is None:
            lr = config["LR"]
            ent_coef = config["ENT_COEF"]
            transition_model_lr = config["TRANSITION_MODEL_LR"]
            max_grad_norm = config["MAX_GRAD_NORM"]
            schedule_accelerator = config["SCHEDULE_ACCELERATOR"]
            num_minibatches = config["NUM_MINIBATCHES"]
        else:
            (
                lr,
                ent_coef,
                transition_model_lr,
                max_grad_norm,
                schedule_accelerator,
                _,
                _,
                _,
            ) = (
                combinations[0],
                combinations[1],
                combinations[2],
                combinations[3],
                combinations[4],
                combinations[5],
                combinations[6],
                combinations[7],
            )
            num_minibatches = config["NUM_MINIBATCHES"]

        config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
        )
        config["MINIBATCH_SIZE"] = (
            config["NUM_ENVS"] * config["NUM_STEPS"] // num_minibatches
        )

        def linear_schedule(count):
            frac = 1.0 - (count // (num_minibatches * config["UPDATE_EPOCHS"])) / (
                config["NUM_UPDATES"] * schedule_accelerator
            )
            return lr * frac

        # INIT NETWORK
        network = ActorCritic(
            env.action_space(env_params).n, activation=config["ACTIVATION"]
        )
        forward_transition_model = TransitionModel(
            state_dim=env.observation_space(env_params).shape[0]
        )
        backward_transition_model = TransitionModel(
            state_dim=env.observation_space(env_params).shape[0]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x, init_x)
        init_transition = jnp.zeros((5,))
        forward_transition_model_params = forward_transition_model.init(
            _rng, init_transition
        )
        backward_transition_model_params = backward_transition_model.init(
            _rng, init_transition
        )
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(lr, eps=1e-5),
            )
        tx_forward_transition_model = optax.adam(learning_rate=transition_model_lr)
        tx_backward_transition_model = optax.adam(learning_rate=transition_model_lr)
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )
        forward_transition_model_train_state = TrainState.create(
            apply_fn=forward_transition_model.apply,
            params=forward_transition_model_params,
            tx=tx_forward_transition_model,
        )
        backward_transition_model_train_state = TrainState.create(
            apply_fn=backward_transition_model.apply,
            params=backward_transition_model_params,
            tx=tx_backward_transition_model,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_state,
                    forward_transition_model_state,
                    backward_transition_model_state,
                    env_state,
                    last_obs,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs, last_obs)
                action_test = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action_test)

                # GET VALUE
                equivalent_obs = jnp.concatenate(
                    [last_obs, jnp.expand_dims(action_test, axis=1)], axis=1
                )
                equivalent_obs = forward_transition_model.apply(
                    forward_transition_model_state.params, equivalent_obs
                )
                equivalent_obs = jnp.concatenate(
                    [
                        equivalent_obs,
                        jnp.expand_dims(jnp.zeros_like(action_test), axis=1),
                    ],
                    axis=1,
                )
                equivalent_obs = backward_transition_model.apply(
                    backward_transition_model_state.params, equivalent_obs
                )
                equivalent_obs = jax.lax.stop_gradient(equivalent_obs)

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs, equivalent_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)
                obsv /= jnp.array([10, 1, 1, 1])
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, obsv, info
                )
                runner_state = (
                    train_state,
                    forward_transition_model_state,
                    backward_transition_model_state,
                    env_state,
                    obsv,
                    rng,
                )
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            (
                train_state,
                forward_transition_model_state,
                backward_transition_model_state,
                env_state,
                last_obs,
                rng,
            ) = runner_state
            _, last_val = network.apply(train_state.params, last_obs, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch_transition_model(update_state, unused):
                def _update_minbatch_forward_transition_model(
                    forward_transition_model_state,
                    batch_info,
                ):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn_forward_transition_model(forward_params, traj_batch):
                        inputs = jnp.concatenate(
                            [
                                traj_batch.obs,
                                jnp.expand_dims(traj_batch.action, axis=1) - 0.5,
                            ],
                            axis=-1,
                        )
                        preds = forward_transition_model.apply(forward_params, inputs)
                        targets = traj_batch.next_obs
                        loss = jnp.square(preds - targets).mean()
                        return loss

                    grad_fn_forward_transition_model = jax.value_and_grad(
                        _loss_fn_forward_transition_model, has_aux=False
                    )
                    (
                        total_loss_forward_transition_model,
                        grads_forward_transition_model,
                    ) = grad_fn_forward_transition_model(
                        forward_transition_model_state.params, traj_batch
                    )
                    forward_transition_model_state = (
                        forward_transition_model_state.apply_gradients(
                            grads=grads_forward_transition_model
                        )
                    )
                    return (
                        forward_transition_model_state,
                        total_loss_forward_transition_model,
                    )

                def _update_minbatch_backward_transition_model(
                    backward_transition_model_state,
                    batch_info,
                ):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn_backward_transition_model(backward_params, traj_batch):
                        inputs = jnp.concatenate(
                            [
                                traj_batch.next_obs,
                                jnp.expand_dims(traj_batch.action, axis=1) - 0.5,
                            ],
                            axis=-1,
                        )
                        preds = forward_transition_model.apply(backward_params, inputs)
                        targets = traj_batch.obs
                        loss = jnp.square(preds - targets).mean()
                        return loss

                    grad_fn_backward_transition_model = jax.value_and_grad(
                        _loss_fn_backward_transition_model, has_aux=False
                    )
                    (
                        total_loss_backward_transition_model,
                        grads_backward_transition_model,
                    ) = grad_fn_backward_transition_model(
                        backward_transition_model_state.params, traj_batch
                    )
                    backward_transition_model_state = (
                        backward_transition_model_state.apply_gradients(
                            grads=grads_backward_transition_model
                        )
                    )
                    return (
                        backward_transition_model_state,
                        total_loss_backward_transition_model,
                    )

                (
                    train_state,
                    forward_transition_model_state,
                    backward_transition_model_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)
                #### transition model batches
                batch_size_transition_model = config["MINIBATCH_SIZE"] * num_minibatches
                assert (
                    batch_size_transition_model
                    == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation_transition_model = jax.random.permutation(
                    _rng, batch_size_transition_model
                )
                batch_transition_model = (traj_batch, advantages, targets)
                batch_transition_model = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size_transition_model,) + x.shape[2:]),
                    batch_transition_model,
                )
                shuffled_batch_transition_model = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation_transition_model, axis=0),
                    batch_transition_model,
                )
                minibatches_transition_model = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch_transition_model,
                )
                forward_transition_model_state, total_loss = jax.lax.scan(
                    _update_minbatch_forward_transition_model,
                    forward_transition_model_state,
                    minibatches_transition_model,
                )
                # jax.debug.print("forward_transition_model_loss: {forward_transition_model_loss}", forward_transition_model_loss=total_loss)

                backward_transition_model_state, total_loss = jax.lax.scan(
                    _update_minbatch_backward_transition_model,
                    backward_transition_model_state,
                    minibatches_transition_model,
                )
                update_state = (
                    train_state,
                    forward_transition_model_state,
                    backward_transition_model_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, total_loss

            # UPDATE NETWORK
            def _update_epoch_actor_critic(update_state, unused):
                def _update_minbatch_network(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        action_test = traj_batch.action

                        equivalent_obs = jnp.concatenate(
                            [traj_batch.obs, jnp.expand_dims(action_test, axis=1)],
                            axis=1,
                        )
                        equivalent_obs = forward_transition_model.apply(
                            forward_transition_model_state.params, equivalent_obs
                        )
                        equivalent_obs = jnp.concatenate(
                            [
                                equivalent_obs,
                                jnp.expand_dims(jnp.zeros_like(action_test), axis=1),
                            ],
                            axis=1,
                        )
                        equivalent_obs = backward_transition_model.apply(
                            backward_transition_model_state.params, equivalent_obs
                        )
                        equivalent_obs = jax.lax.stop_gradient(equivalent_obs)

                        pi, value = network.apply(
                            params, traj_batch.obs, equivalent_obs
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # calculate value loss
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # calculate actor loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - ent_coef * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    forward_transition_model_state,
                    backward_transition_model_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                #### RL batches
                batch_size = config["MINIBATCH_SIZE"] * num_minibatches
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [num_minibatches, -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch_network,
                    train_state,
                    minibatches,
                )
                update_state = (
                    train_state,
                    forward_transition_model_state,
                    backward_transition_model_state,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                # jax.debug.print("total_loss {x}", x=total_loss)
                return update_state, total_loss

            update_state = (
                train_state,
                forward_transition_model_state,
                backward_transition_model_state,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, _ = jax.lax.scan(
                _update_epoch_actor_critic, update_state, None, config["UPDATE_EPOCHS"]
            )
            update_state, _ = jax.lax.scan(
                _update_epoch_transition_model,
                update_state,
                None,
                config["UPDATE_EPOCHS"] * 32,
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]

            runner_state = (
                train_state,
                forward_transition_model_state,
                backward_transition_model_state,
                env_state,
                last_obs,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            forward_transition_model_train_state,
            backward_transition_model_train_state,
            env_state,
            obsv,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train
