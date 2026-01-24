from pathlib import Path
import gym
import numpy as np

from .model import build_model
from .agent import build_agent
from .utils import set_seeds


def evaluate(cfg, weights_path=None, nb_episodes=100):
    env = gym.make(cfg['env_name'])
    set_seeds(cfg['seed'], env)

    nb_actions = env.action_space.n
    model = build_model(
        env.observation_space.shape,
        nb_actions,
        hidden_units=cfg['model']['hidden_units'],
        activation=cfg['model']['activation'],
    )
    agent = build_agent(model, nb_actions, cfg)

    if weights_path:
        agent.load_weights(str(weights_path))

    successes = 0
    rewards = []

    for _ in range(nb_episodes):
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        agent.reset_states()
        done = False
        total_reward = 0
        steps = 0
        while not done:
            action = agent.forward(state)
            step_out = env.step(action)
            if len(step_out) == 5:
                next_state, reward, done, truncated, _ = step_out
                done = done or truncated
            else:
                next_state, reward, done, _ = step_out
            state = next_state
            total_reward += reward
            steps += 1
        rewards.append(total_reward)

        max_steps = getattr(env, '_max_episode_steps', None)
        elapsed = getattr(env, '_elapsed_steps', steps)
        if max_steps is not None and elapsed < max_steps:
            successes += 1

    return {
        'successes': successes,
        'success_rate': successes / nb_episodes,
        'mean_reward': float(np.mean(rewards)),
        'rewards': rewards,
    }
