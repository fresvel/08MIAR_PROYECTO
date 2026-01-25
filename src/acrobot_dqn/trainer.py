import gym
from rl.callbacks import FileLogger, ModelIntervalCheckpoint

from .model import DQNModelBuilder
from .agent import DQNAgentFactory
from .utils import set_seeds, make_run_dir, save_config, copy_config_source
from .plots import plot_training_logs


class DQNTrainer:
    def __init__(self, cfg, config_path=None):
        self.cfg = cfg
        self.config_path = config_path

    def train(self):
        env = gym.make(self.cfg['env_name'])
        set_seeds(self.cfg['seed'], env)

        nb_actions = env.action_space.n
        model = DQNModelBuilder(
            hidden_units=self.cfg['model']['hidden_units'],
            activation=self.cfg['model']['activation'],
        ).build(env.observation_space.shape, nb_actions)
        agent = DQNAgentFactory(self.cfg).build(model, nb_actions)

        output_dir = self.cfg['logging']['output_dir']
        run_dir = make_run_dir(
            output_dir,
            run_name=self.cfg.get('run_name'),
            run_id=self.cfg.get('run_id'),
        )
        save_config(self.cfg, run_dir / 'config.yaml')
        copy_config_source(self.config_path, run_dir)

        weights_path = run_dir / 'checkpoints' / f"dqn_{self.cfg['env_name']}_weights.h5f"
        log_path = run_dir / 'logs' / f"dqn_{self.cfg['env_name']}_log.json"

        callbacks = [
            ModelIntervalCheckpoint(str(weights_path), interval=self.cfg['logging']['checkpoint_interval']),
            FileLogger(str(log_path), interval=self.cfg['logging']['log_interval']),
        ]

        agent.fit(
            env,
            nb_steps=self.cfg['training']['nb_steps'],
            visualize=False,
            verbose=2,
            callbacks=callbacks,
        )
        agent.save_weights(str(weights_path), overwrite=True)

        weights_export_dir = run_dir.parent.parent / 'weights'
        weights_export_dir.mkdir(parents=True, exist_ok=True)
        final_weights_path = weights_export_dir / f"{run_dir.name}_weights.h5f"
        agent.save_weights(str(final_weights_path), overwrite=True)

        # Exportar el modelo completo en formato .keras (Keras 3)
        final_model_path = weights_export_dir / f"{run_dir.name}_model.keras"
        try:
            agent.model.save(str(final_model_path))
        except Exception:
            final_model_path = None

        plot_training_logs(log_path, run_dir / 'figures')

        return str(final_weights_path), str(run_dir)


def train(cfg, config_path=None):
    return DQNTrainer(cfg, config_path=config_path).train()
