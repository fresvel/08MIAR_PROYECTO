from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

try:
    import yaml
except Exception:  # pragma: no cover - optional dependency
    yaml = None


def _deep_get(cfg: Dict, *keys, default=None):
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


@dataclass
class RunArtifacts:
    run_id: str
    run_dir: Path
    cfg: Dict
    metrics: Dict


class ExperimentComparator:
    """Load, summarize and compare experiment runs saved under outputs/runs."""

    def __init__(self, output_dir: str | Path = "outputs"):
        self.output_dir = Path(output_dir)
        self.runs_dir = self.output_dir / "runs"

    def _load_yaml(self, path: Path) -> Dict:
        if yaml is None:
            raise RuntimeError("PyYAML no esta instalado. Instala con: pip install pyyaml")
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    def load_run(self, run_id: str) -> RunArtifacts:
        run_dir = self.runs_dir / run_id
        cfg_path = run_dir / "config.yaml"
        metrics_path = run_dir / "metrics" / "eval.json"

        if not cfg_path.exists():
            raise FileNotFoundError(cfg_path)

        cfg = self._load_yaml(cfg_path)
        metrics = (
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.exists()
            else {}
        )
        return RunArtifacts(run_id=run_id, run_dir=run_dir, cfg=cfg, metrics=metrics)

    def summarize_run(self, run_id: str) -> Dict:
        art = self.load_run(run_id)
        cfg = art.cfg
        metrics = art.metrics
        row = {
            "run_id": run_id,
            "success_rate": metrics.get("success_rate"),
            "mean_reward": metrics.get("mean_reward"),
            "learning_rate": _deep_get(cfg, "training", "learning_rate"),
            "target_model_update": _deep_get(cfg, "training", "target_model_update"),
            "eps_min": _deep_get(cfg, "policy", "eps_min"),
            "anneal_steps": _deep_get(cfg, "policy", "anneal_steps"),
            "hidden_units": _deep_get(cfg, "model", "hidden_units"),
            "double_dqn": _deep_get(cfg, "variants", "double_dqn"),
            "dueling_dqn": _deep_get(cfg, "variants", "dueling_dqn"),
            "run_dir": art.run_dir.as_posix(),
        }
        # Avoid pandas treating lists as array-like columns.
        row["hidden_units"] = repr(row["hidden_units"])
        return row

    def compare(
        self,
        run_ids: Iterable[str],
        *,
        sequential: bool = True,
    ) -> pd.DataFrame:
        rows: List[Dict] = []
        for run_id in run_ids:
            try:
                rows.append(self.summarize_run(run_id))
            except FileNotFoundError as exc:
                rows.append(
                    {
                        "run_id": run_id,
                        "error": f"Run incompleto o inexistente: {exc}",
                    }
                )

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        if sequential:
            for col in ("success_rate", "mean_reward"):
                if col in df.columns:
                    df[f"delta_{col}"] = df[col].diff()
        return df

    def diff_params(
        self, run_id_a: str, run_id_b: str, keys: Optional[Iterable[str]] = None
    ) -> pd.DataFrame:
        art_a = self.load_run(run_id_a)
        art_b = self.load_run(run_id_b)

        if keys is None:
            keys = [
                "model.hidden_units",
                "model.activation",
                "training.learning_rate",
                "training.target_model_update",
                "training.gamma",
                "policy.eps_min",
                "policy.anneal_steps",
                "variants.double_dqn",
                "variants.dueling_dqn",
            ]

        rows = []
        for key in keys:
            parts = key.split(".")
            val_a = _deep_get(art_a.cfg, *parts)
            val_b = _deep_get(art_b.cfg, *parts)
            if val_a != val_b:
                rows.append({"param": key, run_id_a: repr(val_a), run_id_b: repr(val_b)})

        return pd.DataFrame(rows)

    def figure_paths(self, run_id: str) -> List[Path]:
        fig_dir = self.runs_dir / run_id / "figures"
        return [
            fig_dir / "training_reward.png",
            fig_dir / "training_mean_q.png",
            fig_dir / "eval_rewards.png",
            fig_dir / "eval_rewards_hist.png",
        ]
