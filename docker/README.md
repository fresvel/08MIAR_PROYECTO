# Docker (Python 3.8) para entrenamiento

Este contenedor instala las dependencias de `requirements.txt` en Python 3.8.

## Build

```bash
docker build -f docker/Dockerfile -t miar-rl:py38 .
```

## Entrenamiento (base)

```bash
docker run --rm -it \
  -v "$PWD:/workspace:Z" \
  -w /workspace \
  miar-rl:py38 \
  python scripts/train.py --config configs/dqn_base.yaml --run-id dqn_base_01
```

## Evaluación

```bash
docker run --rm -it \
  -v "$PWD:/workspace:Z" \
  -w /workspace \
  miar-rl:py38 \
  python scripts/eval.py --config configs/dqn_base.yaml \
    --weights outputs/weights/dqn_base_01_weights.h5 \
    --episodes 100 \
    --save outputs/runs/dqn_base_01/metrics/eval.json
```

Nota (RHEL/SELinux): se usa `:Z` en el volumen para evitar problemas de permisos.
