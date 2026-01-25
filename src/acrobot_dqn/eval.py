import json
from pathlib import Path
import gym
import numpy as np

from .model import DQNModelBuilder
from .agent import DQNAgentFactory
from .utils import set_seeds
from .plots import plot_eval_results


class DQNEvaluator:
    def __init__(self, cfg):
        self.cfg = cfg

    def evaluate(self, weights_path=None, nb_episodes=100, save_path=None, fig_dir=None):
        env = gym.make(self.cfg['env_name'])
        set_seeds(self.cfg['seed'], env)

        nb_actions = env.action_space.n
        model = DQNModelBuilder(
            hidden_units=self.cfg['model']['hidden_units'],
            activation=self.cfg['model']['activation'],
        ).build(env.observation_space.shape, nb_actions)
        agent = DQNAgentFactory(self.cfg).build(model, nb_actions)

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

        results = {
            'successes': successes,
            'success_rate': successes / nb_episodes,
            'mean_reward': float(np.mean(rewards)),
            'rewards': rewards,
        }
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)

        if fig_dir:
            plot_eval_results(results, fig_dir)
        elif save_path:
            run_dir = save_path.parent.parent if save_path.parent.name == 'metrics' else save_path.parent
            plot_eval_results(results, run_dir / 'figures')
        return results


def evaluate(cfg, weights_path=None, nb_episodes=100, save_path=None, fig_dir=None):
    return DQNEvaluator(cfg).evaluate(
        weights_path=weights_path,
        nb_episodes=nb_episodes,
        save_path=save_path,
        fig_dir=fig_dir,
    )
