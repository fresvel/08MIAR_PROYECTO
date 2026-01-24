from pathlib import Path
from datetime import datetime
import json
import random
import shutil
import numpy as np
import tensorflow as tf

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def set_seeds(seed, env=None):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    if env is not None:
        try:
            env.seed(seed)
        except Exception:
            pass


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def load_config(path):
    path = Path(path)
    if path.suffix.lower() == '.json':
        return json.loads(path.read_text(encoding='utf-8'))
    if path.suffix.lower() in {'.yml', '.yaml'}:
        if yaml is None:
            raise RuntimeError('PyYAML no esta instalado. Instala con: pip install pyyaml')
        return yaml.safe_load(path.read_text(encoding='utf-8'))
    raise ValueError(f'Formato de config no soportado: {path.suffix}')


def save_config(cfg, path):
    path = Path(path)
    if path.suffix.lower() in {'.yml', '.yaml'}:
        if yaml is None:
            raise RuntimeError('PyYAML no esta instalado. Instala con: pip install pyyaml')
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding='utf-8')
        return
    if path.suffix.lower() == '.json':
        path.write_text(json.dumps(cfg, indent=2), encoding='utf-8')
        return
    raise ValueError(f'Formato de config no soportado: {path.suffix}')


def make_run_dir(output_dir, run_name=None, run_id=None):
    output_dir = Path(output_dir)
    if run_id:
        final_id = run_id
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        final_id = f'{run_name}_{ts}' if run_name else ts
    run_dir = output_dir / 'runs' / final_id
    ensure_dir(run_dir / 'checkpoints')
    ensure_dir(run_dir / 'logs')
    ensure_dir(run_dir / 'metrics')
    return run_dir


def copy_config_source(config_path, run_dir):
    if not config_path:
        return
    config_path = Path(config_path)
    if config_path.exists():
        shutil.copy2(config_path, run_dir / f'config_source{config_path.suffix}')
