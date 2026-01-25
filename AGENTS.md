# Plan de trabajo (día siguiente)

## Objetivo
Mejorar la estabilidad del modelo base de Acrobot (reducir outliers en evaluación) manteniendo el 100% de éxito.

## Estado actual
- Run base: `dqn_base_00`
- `success_rate = 1.0`, `mean_reward = -88.23`
- Persisten outliers en evaluación (recompensas hasta ~-200).
- Gráficos generados en `outputs/runs/dqn_base_00/figures/`

## Tareas del día siguiente

1) **Nuevo experimento base (estabilidad)**
   - Crear `configs/dqn_base_stable.yaml` con:
     - `learning_rate: 0.0005`
     - `target_model_update: 5000`
   - Entrenar:
     ```
     ./docker_train_eval.sh configs/dqn_base_stable.yaml dqn_base_01 100
     ```
   - Guardar resultados en `outputs/runs/dqn_base_01/`

2) **Comparación base vs estable**
   - Comparar:
     - `mean_reward`
     - histograma de recompensas
     - outliers en `eval_rewards.png`
   - Si `dqn_base_01` reduce outliers y mantiene 100%, elegirlo como **base final**.

3) **(Opcional) Mejora de exploración**
   - Solo si el punto anterior no mejora:
     - `anneal_steps: 100000`
     - `eps_min: 0.01`
   - Ejecutar otro run (e.g., `dqn_base_02`) y comparar.

4) **Documentación**
   - Actualizar `main.ipynb` con:
     - tabla de hiperparámetros del run ganador
     - gráficos de entrenamiento y evaluación
     - comentarios justificando el cambio

## Criterio de aceptación
Un run que mantenga:
- `success_rate = 1.0` (100/100)
- `mean_reward` mejor o igual que `-88`
- Menos outliers en evaluación
