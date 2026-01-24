import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / 'src'))

from acrobot_dqn.utils import load_config
from acrobot_dqn.trainer import train


def main():
    parser = argparse.ArgumentParser(description='Entrenar DQN para Acrobot.')
    parser.add_argument('--config', required=True, help='Ruta al archivo de configuracion (yaml/json).')
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights_path = train(cfg)
    print(f'Pesos guardados en: {weights_path}')


if __name__ == '__main__':
    main()
