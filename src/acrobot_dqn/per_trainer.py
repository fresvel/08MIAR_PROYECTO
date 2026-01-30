from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from .model import DQNModelBuilder
from .memory import PrioritizedReplayMemory
from .utils import set_seeds, make_run_dir, save_config, copy_config_source
from .plots import plot_training_logs


def _linear_schedule(value_start, value_end, step, total_steps):
    if total_steps <= 0:
        return value_end
    fraction = min(float(step) / float(total_steps), 1.0)
    return value_start + fraction * (value_end - value_start)


class PrioritizedDQNTrainer:
    """Custom training loop to enable full Prioritized Replay (TD-error updates)."""

    def __init__(self, cfg, config_path=None):
        self.cfg = cfg
        self.config_path = config_path

    def train(self):
        env = gym.make(self.cfg["env_name"])
        set_seeds(self.cfg["seed"], env)

        nb_actions = env.action_space.n
        model = DQNModelBuilder(
            hidden_units=self.cfg["model"]["hidden_units"],
            activation=self.cfg["model"]["activation"],
        ).build(env.observation_space.shape, nb_actions)

        target_model = tf.keras.models.clone_model(model)
        target_model.set_weights(model.get_weights())

        model.compile(
            optimizer=Adam(learning_rate=self.cfg["training"]["learning_rate"]),
            loss="mse",
        )

        memory_cfg = self.cfg.get("memory", {})
        memory = PrioritizedReplayMemory(
            limit=self.cfg["memory_limit"],
            window_length=self.cfg["window_length"],
            alpha=memory_cfg.get("alpha", 0.6),
            eps=memory_cfg.get("eps", 1e-6),
        )

        output_dir = self.cfg["logging"]["output_dir"]
        run_dir = make_run_dir(
            output_dir,
            run_name=self.cfg.get("run_name"),
            run_id=self.cfg.get("run_id"),
        )
        save_config(self.cfg, run_dir / "config.yaml")
        copy_config_source(self.config_path, run_dir)

        weights_path = run_dir / "checkpoints" / f"dqn_{self.cfg['env_name']}_weights.h5"
        log_path = run_dir / "logs" / f"dqn_{self.cfg['env_name']}_log.json"

        nb_steps = self.cfg["training"]["nb_steps"]
        warmup_steps = self.cfg["training"]["warmup_steps"]
        gamma = self.cfg["training"]["gamma"]
        target_update = self.cfg["training"]["target_model_update"]
        batch_size = self.cfg["training"].get("batch_size", 32)
        logging_cfg = self.cfg.get("logging", {})
        checkpoint_interval = logging_cfg["checkpoint_interval"]
        log_interval = logging_cfg["log_interval"]
        console_interval = logging_cfg.get(
            "console_interval",
            logging_cfg.get("print_interval", log_interval),
        )

        eps_max = self.cfg["policy"]["eps_max"]
        eps_min = self.cfg["policy"]["eps_min"]
        anneal_steps = self.cfg["policy"]["anneal_steps"]

        beta_start = memory_cfg.get("beta_start", 0.4)
        beta_steps = memory_cfg.get("beta_steps", nb_steps)

        double_dqn = self.cfg["variants"]["double_dqn"]

        logs = []
        episode = 0
        episode_reward = 0.0
        episode_qs = []
        episode_actions = []
        episode_steps = 0
        total_steps = 0
        episode_rewards_window = deque(maxlen=100)
        episode_start_time = time.time()
        last_console_step = 0

        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs

        while total_steps < nb_steps:
            eps = _linear_schedule(eps_max, eps_min, total_steps, anneal_steps)
            state_batch = np.array([memory.get_recent_state(state)])
            q_values = model.predict_on_batch(state_batch)[0]

            if np.random.rand() < eps:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(q_values))

            step_out = env.step(action)
            if len(step_out) == 5:
                next_state, reward, done, truncated, _ = step_out
                done = done or truncated
            else:
                next_state, reward, done, _ = step_out

            memory.append(state, action, reward, done, training=True)

            episode_reward += reward
            episode_steps += 1
            episode_qs.append(float(q_values[action]))
            episode_actions.append(int(action))
            total_steps += 1

            if total_steps > warmup_steps and memory.nb_entries >= batch_size:
                beta = _linear_schedule(beta_start, 1.0, total_steps, beta_steps)
                experiences, indices, weights = memory.sample(
                    batch_size, return_indices=True, beta=beta
                )
                state0_batch = np.array([e.state0 for e in experiences])
                state1_batch = np.array([e.state1 for e in experiences])
                action_batch = np.array([e.action for e in experiences], dtype=np.int64)
                reward_batch = np.array([e.reward for e in experiences], dtype=np.float32)
                terminal_batch = np.array([float(e.terminal1) for e in experiences], dtype=np.float32)

                if double_dqn:
                    q_next_online = model.predict_on_batch(state1_batch)
                    best_actions = np.argmax(q_next_online, axis=1)
                    q_next_target = target_model.predict_on_batch(state1_batch)
                    next_q = q_next_target[np.arange(batch_size), best_actions]
                else:
                    q_next_target = target_model.predict_on_batch(state1_batch)
                    next_q = np.max(q_next_target, axis=1)

                targets = reward_batch + (1.0 - terminal_batch) * gamma * next_q
                q_targets = model.predict_on_batch(state0_batch)
                td_errors = targets - q_targets[np.arange(batch_size), action_batch]
                q_targets[np.arange(batch_size), action_batch] = targets

                sample_weights = None
                if weights is not None:
                    sample_weights = weights.astype(np.float32)

                model.train_on_batch(state0_batch, q_targets, sample_weight=sample_weights)
                memory.update_priorities(indices, np.abs(td_errors) + memory.eps)

            if target_update and total_steps % target_update == 0:
                target_model.set_weights(model.get_weights())

            if checkpoint_interval and total_steps % checkpoint_interval == 0:
                model.save_weights(str(weights_path))

            if done:
                episode += 1
                mean_q = float(np.mean(episode_qs)) if episode_qs else float("nan")
                duration = max(time.time() - episode_start_time, 1e-6)
                steps_per_second = episode_steps / duration
                episode_rewards_window.append(episode_reward)
                window_mean = float(np.mean(episode_rewards_window))
                window_min = float(np.min(episode_rewards_window))
                window_max = float(np.max(episode_rewards_window))
                mean_action = float(np.mean(episode_actions)) if episode_actions else 0.0
                min_action = int(np.min(episode_actions)) if episode_actions else 0
                max_action = int(np.max(episode_actions)) if episode_actions else 0
                logs.append(
                    {
                        "episode": episode,
                        "episode_reward": float(episode_reward),
                        "mean_q": mean_q,
                        "iteration": total_steps,
                        "episode_steps": episode_steps,
                        "mean_eps": float(eps),
                    }
                )
                if console_interval and total_steps - last_console_step >= console_interval:
                    print(
                        f"{total_steps}/{nb_steps}: episode: {episode}, duration: {duration:.3f}s, "
                        f"episode steps: {episode_steps}, steps per second: {steps_per_second:.0f}, "
                        f"episode reward: {episode_reward:.3f}, mean reward: {window_mean:.3f} "
                        f"[{window_min:.3f}, {window_max:.3f}], mean action: {mean_action:.3f} "
                        f"[{min_action}, {max_action}], mean_q: {mean_q:.6f}, mean_eps: {eps:.6f}",
                        flush=True,
                    )
                    last_console_step = total_steps
                episode_reward = 0.0
                episode_qs = []
                episode_actions = []
                episode_steps = 0
                episode_start_time = time.time()
                obs = env.reset()
                state = obs[0] if isinstance(obs, tuple) else obs
            else:
                state = next_state

        Path(log_path).write_text(
            json.dumps(logs, indent=2), encoding="utf-8"
        )

        model.save_weights(str(weights_path))
        weights_export_dir = run_dir.parent.parent / "weights"
        weights_export_dir.mkdir(parents=True, exist_ok=True)
        final_weights_path = weights_export_dir / f"{run_dir.name}_weights.h5"
        model.save_weights(str(final_weights_path))

        final_model_path = weights_export_dir / f"{run_dir.name}_model.keras"
        try:
            model.save(str(final_model_path))
        except Exception:
            final_model_path = None

        plot_training_logs(log_path, run_dir / "figures")

        return str(final_weights_path), str(run_dir)
