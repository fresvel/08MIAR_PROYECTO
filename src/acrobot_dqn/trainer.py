from pathlib import Path
import gym
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from .model import build_model
from .agent import build_agent
from .utils import set_seeds, ensure_dir


def train(cfg):
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

    output_dir = Path(cfg['logging']['output_dir'])
    checkpoints_dir = output_dir / 'checkpoints'
    logs_dir = output_dir / 'logs'
    ensure_dir(checkpoints_dir)
    ensure_dir(logs_dir)

    weights_path = checkpoints_dir / f"dqn_{cfg['env_name']}_weights.h5f"
    log_path = logs_dir / f"dqn_{cfg['env_name']}_log.json"

    callbacks = [
        ModelIntervalCheckpoint(str(weights_path), interval=cfg['logging']['checkpoint_interval']),
        FileLogger(str(log_path), interval=cfg['logging']['log_interval']),
    ]

    agent.fit(
        env,
        nb_steps=cfg['training']['nb_steps'],
        visualize=False,
        verbose=2,
        callbacks=callbacks,
    )
    agent.save_weights(str(weights_path), overwrite=True)
    return str(weights_path)
