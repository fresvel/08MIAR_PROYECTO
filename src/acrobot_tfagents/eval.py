import json
from pathlib import Path
import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment

from acrobot_dqn.plots import plot_eval_results
from acrobot_dqn.utils import set_seeds

from .model import TFAgentsNetworkBuilder
from .agent import TFAgentsDQNFactory


def evaluate(cfg, run_dir, nb_episodes=100, save_path=None, fig_dir=None):
    env_name = cfg['env_name']
    py_env = suite_gym.load(env_name)
    tf_env = tf_py_environment.TFPyEnvironment(py_env)

    set_seeds(cfg['seed'], py_env)

    network = TFAgentsNetworkBuilder(
        fc_layer_params=cfg['network']['fc_layer_params'],
        network_type=cfg['network']['type'],
    ).build(tf_env.observation_spec(), tf_env.action_spec())

    agent = TFAgentsDQNFactory(cfg).build(
        tf_env.time_step_spec(),
        tf_env.action_spec(),
        network,
    )

    # Restore checkpoint
    ckpt_dir = Path(run_dir) / 'checkpoints'
    checkpointer = tf.train.Checkpoint(agent=agent)
    latest = tf.train.latest_checkpoint(str(ckpt_dir))
    if latest:
        checkpointer.restore(latest).expect_partial()

    rewards = []
    successes = 0
    max_steps = getattr(py_env, '_max_episode_steps', None)

    for _ in range(nb_episodes):
        time_step = tf_env.reset()
        total_reward = 0.0
        steps = 0
        while not time_step.is_last():
            action_step = agent.policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            total_reward += time_step.reward.numpy()[0]
            steps += 1
        rewards.append(float(total_reward))
        if max_steps is not None and steps < max_steps:
            successes += 1

    results = {
        'successes': successes,
        'success_rate': successes / nb_episodes,
        'mean_reward': float(sum(rewards) / len(rewards)),
        'rewards': rewards,
    }

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(json.dumps(results, indent=2), encoding='utf-8')

    if fig_dir:
        plot_eval_results(results, fig_dir)
    elif save_path:
        run_dir = save_path.parent.parent if save_path.parent.name == 'metrics' else save_path.parent
        plot_eval_results(results, run_dir / 'figures')

    return results
