import json
from pathlib import Path
import tensorflow as tf
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.utils import common

from acrobot_dqn.utils import set_seeds, make_run_dir, save_config, copy_config_source
from acrobot_dqn.plots import plot_training_logs

from .model import TFAgentsNetworkBuilder
from .agent import TFAgentsDQNFactory


def compute_avg_return(tf_env, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = tf_env.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = tf_env.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    return (total_return / num_episodes).numpy()[0]


class TFAgentsTrainer:
    def __init__(self, cfg, config_path=None):
        self.cfg = cfg
        self.config_path = config_path

    def train(self):
        env_name = self.cfg['env_name']
        py_env = suite_gym.load(env_name)
        eval_py_env = suite_gym.load(env_name)
        tf_env = tf_py_environment.TFPyEnvironment(py_env)
        eval_tf_env = tf_py_environment.TFPyEnvironment(eval_py_env)

        set_seeds(self.cfg['seed'], py_env)

        network = TFAgentsNetworkBuilder(
            fc_layer_params=self.cfg['network']['fc_layer_params'],
            network_type=self.cfg['network']['type'],
        ).build(tf_env.observation_spec(), tf_env.action_spec())

        agent = TFAgentsDQNFactory(self.cfg).build(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
            network,
        )

        output_dir = self.cfg['logging']['output_dir']
        run_dir = make_run_dir(
            output_dir,
            run_name=self.cfg.get('run_name'),
            run_id=self.cfg.get('run_id'),
        )
        save_config(self.cfg, run_dir / 'config.yaml')
        copy_config_source(self.config_path, run_dir)

        log_path = run_dir / 'logs' / 'train_log.json'
        figures_dir = run_dir / 'figures'

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=agent.collect_data_spec,
            batch_size=tf_env.batch_size,
            max_length=self.cfg['training']['replay_buffer_capacity'],
        )

        random_policy = random_tf_policy.RandomTFPolicy(
            tf_env.time_step_spec(),
            tf_env.action_spec(),
        )
        initial_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            random_policy,
            observers=[replay_buffer.add_batch],
            num_steps=self.cfg['training']['initial_collect_steps'],
        )
        initial_driver.run()

        dataset = replay_buffer.as_dataset(
            num_parallel_calls=3,
            sample_batch_size=self.cfg['training']['batch_size'],
            num_steps=2,
        ).prefetch(3)
        iterator = iter(dataset)

        collect_driver = dynamic_step_driver.DynamicStepDriver(
            tf_env,
            agent.collect_policy,
            observers=[replay_buffer.add_batch],
            num_steps=self.cfg['training']['collect_steps_per_iteration'],
        )

        checkpointer = common.Checkpointer(
            ckpt_dir=str(run_dir / 'checkpoints'),
            max_to_keep=3,
            agent=agent,
            policy=agent.policy,
            replay_buffer=replay_buffer,
            global_step=agent.train_step_counter,
        )

        logs = []
        for iteration in range(self.cfg['training']['num_iterations']):
            collect_driver.run()
            experience, _ = next(iterator)
            train_loss = agent.train(experience)

            if iteration % self.cfg['training']['log_interval'] == 0:
                avg_return = compute_avg_return(
                    eval_tf_env,
                    agent.policy,
                    num_episodes=self.cfg['training']['num_eval_episodes'],
                )
                row = {
                    'iteration': int(iteration),
                    'loss': float(train_loss.loss.numpy()),
                    'average_return': float(avg_return),
                }
                logs.append(row)
                log_path.write_text(json.dumps(logs, indent=2), encoding='utf-8')

            if iteration % self.cfg['training']['checkpoint_interval'] == 0:
                checkpointer.save(agent.train_step_counter)

        checkpointer.save(agent.train_step_counter)

        weights_export_dir = Path(output_dir) / 'weights'
        weights_export_dir.mkdir(parents=True, exist_ok=True)
        policy_dir = weights_export_dir / f"{run_dir.name}_policy"
        try:
            tf.saved_model.save(agent.policy, str(policy_dir))
        except Exception:
            pass

        plot_training_logs(log_path, figures_dir)

        return str(run_dir)


def train(cfg, config_path=None):
    return TFAgentsTrainer(cfg, config_path=config_path).train()
