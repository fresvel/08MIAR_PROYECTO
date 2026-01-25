## Análisis de resultados (dqn_base_00)

Revisé todas las salidas del run `dqn_base_00` (logs y las 4 imágenes). Este es el análisis y las mejoras sugeridas:

**Resumen numérico (eval.json)**
- `success_rate = 1.0` (100/100)
- `mean_reward = -88.23`
- Recompensas oscilan aprox. entre **-70 y -200** (hay outliers).

**Lectura de los gráficos**

1) **training_reward.png**  
   - Sube rápido de ~–500 a ~–100 en los primeros ~300–500 episodios.  
   - Luego se estabiliza alrededor de –90/–110 con variabilidad moderada.  
   - **Interpretación:** aprende correctamente el swing‑up; la política converge.

2) **training_mean_q.png**  
   - Q medio baja de ~0 a ~–35 y luego se recupera y estabiliza cerca de ~–31.  
   - **Interpretación:** entrenamiento estable, sin explosión de Q‑values.

3) **eval_rewards.png**  
   - La mayoría de episodios está en ~–80 a –100, pero hay **picos negativos** (–150 a –200) en episodios aislados.  
   - **Interpretación:** la política es buena pero **no totalmente robusta**; aún hay episodios que tardan más pasos.

4) **eval_rewards_hist.png**  
   - Distribución concentrada en ~–80/–95, con cola hacia –140/–200.  
   - **Interpretación:** éxito consistente, pero con **variabilidad**.

**Conclusión actual**
El modelo **cumple el requisito** (100/100). Sin embargo, la variabilidad indica margen de mejora: no es inestable, pero aún hay episodios con demasiados pasos.

**Recomendación de mejora (técnica y mínima)**

Opción A (más estabilidad):
- `learning_rate: 0.0005`
- `target_model_update: 5000`

Opción B (más exploración controlada):
- `anneal_steps: 100000`
- `eps_min: 0.01`

Opción C (más capacidad):
- `hidden_units: [256, 256]`

**Sugerencia concreta:** empezar con **Opción A** para reducir outliers en evaluación.

## Nota técnica: Jupyter desde Docker (Python 3.8)

Si no se puede instalar Python 3.8 en su máquina, se puede se ejecutar Jupyter dentro del contenedor y abrirlo en su navegador:

```bash
docker compose run --rm -p 8888:8888 rl \
  jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

Luego se abre:

```
http://localhost:8888/?token=TU_TOKEN
```

Así podrás ver `main.ipynb` y se ejecutar `model.summary()` con el entorno correcto.
