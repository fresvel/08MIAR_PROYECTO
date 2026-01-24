from pathlib import Path
import json
import random
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
