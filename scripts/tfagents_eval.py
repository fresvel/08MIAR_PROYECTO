import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))

from acrobot_dqn.utils import load_config
from acrobot_tfagents.eval import evaluate


def main():
    parser = argparse.ArgumentParser(description='Evaluar DQN TF-Agents para Acrobot.')
    parser.add_argument('--config', required=True, help='Ruta al archivo de configuracion (yaml/json).')
    parser.add_argument('--run-dir', required=True, help='Directorio del run (outputs/runs/<run_id>).')
    parser.add_argument('--episodes', type=int, default=100, help='Numero de episodios de test.')
    parser.add_argument('--save', default=None, help='Ruta para guardar métricas en JSON.')
    parser.add_argument('--fig-dir', default=None, help='Directorio para guardar gráficas.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    results = evaluate(
        cfg,
        run_dir=args.run_dir,
        nb_episodes=args.episodes,
        save_path=args.save,
        fig_dir=args.fig_dir,
    )
    print(f"Exitos: {results['successes']}/{args.episodes} ({results['success_rate']:.1%})")
    print(f"Recompensa media: {results['mean_reward']:.2f}")


if __name__ == '__main__':
    main()
