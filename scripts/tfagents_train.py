import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))

from acrobot_dqn.utils import load_config
from acrobot_tfagents.trainer import train


def main():
    parser = argparse.ArgumentParser(description='Entrenar DQN con TF-Agents para Acrobot.')
    parser.add_argument('--config', required=True, help='Ruta al archivo de configuracion (yaml/json).')
    parser.add_argument('--run-name', default=None, help='Etiqueta opcional para el run (usa timestamp).')
    parser.add_argument('--run-id', default=None, help='Identificador exacto del run (sin timestamp).')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.run_name:
        cfg['run_name'] = args.run_name
    if args.run_id:
        cfg['run_id'] = args.run_id

    run_dir = train(cfg, config_path=args.config)
    print(f'Run dir: {run_dir}')


if __name__ == '__main__':
    main()
