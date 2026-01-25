import json
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def _load_json_logs(path):
    path = Path(path)
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Fallback: JSON lines
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows


def _set_plot_style():
    sns.set_theme(style="whitegrid")
    sns.set_palette("Set2")


def _moving_average(values, window=10):
    if window <= 1 or len(values) < window:
        return values
    weights = [1.0 / window] * window
    return list(
        __import__("numpy").convolve(values, weights, mode="valid")
    )


def plot_training_logs(log_path, fig_dir):
    rows = _load_json_logs(log_path)
    if not rows:
        return

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    _set_plot_style()

    # Try to find reward/episode fields
    episodes = []
    rewards = []
    mean_qs = []
    iterations = []
    avg_returns = []

    for row in rows:
        if isinstance(row, dict):
            ep = row.get("episode") or row.get("nb_episode")
            if ep is not None:
                episodes.append(ep)
            reward = row.get("episode_reward") or row.get("reward")
            if reward is not None:
                rewards.append(reward)
            mean_q = row.get("mean_q")
            if mean_q is not None:
                mean_qs.append(mean_q)
            it = row.get("iteration")
            if it is not None:
                iterations.append(it)
            avg_ret = row.get("average_return")
            if avg_ret is not None:
                avg_returns.append(avg_ret)

    if episodes and rewards and len(episodes) == len(rewards):
        plt.figure(figsize=(7, 4))
        sns.lineplot(x=episodes, y=rewards, label="Recompensa")
        smoothed = _moving_average(rewards, window=10)
        if smoothed and len(smoothed) < len(episodes):
            sns.lineplot(x=episodes[-len(smoothed):], y=smoothed, label="Media móvil (10)")
        plt.xlabel("Episodio")
        plt.ylabel("Recompensa")
        plt.title("Recompensa por episodio")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "training_reward.png", dpi=150)
        plt.close()

    if episodes and mean_qs and len(episodes) == len(mean_qs):
        plt.figure(figsize=(7, 4))
        sns.lineplot(x=episodes, y=mean_qs, label="Q medio")
        smoothed = _moving_average(mean_qs, window=10)
        if smoothed and len(smoothed) < len(episodes):
            sns.lineplot(x=episodes[-len(smoothed):], y=smoothed, label="Media móvil (10)")
        plt.xlabel("Episodio")
        plt.ylabel("Q medio")
        plt.title("Q medio por episodio")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "training_mean_q.png", dpi=150)
        plt.close()

    if iterations and avg_returns and len(iterations) == len(avg_returns):
        plt.figure(figsize=(7, 4))
        sns.lineplot(x=iterations, y=avg_returns, label="Average return")
        smoothed = _moving_average(avg_returns, window=10)
        if smoothed and len(smoothed) < len(iterations):
            sns.lineplot(x=iterations[-len(smoothed):], y=smoothed, label="Media móvil (10)")
        plt.xlabel("Iteración")
        plt.ylabel("Average return")
        plt.title("Average return por iteración")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "training_average_return.png", dpi=150)
        plt.close()


def plot_eval_results(results, fig_dir):
    if not results:
        return
    rewards = results.get("rewards")
    if not rewards:
        return

    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    _set_plot_style()

    plt.figure(figsize=(7, 4))
    xs = list(range(1, len(rewards) + 1))
    sns.lineplot(x=xs, y=rewards, label="Recompensa")
    smoothed = _moving_average(rewards, window=5)
    if smoothed and len(smoothed) < len(xs):
        sns.lineplot(x=xs[-len(smoothed):], y=smoothed, label="Media móvil (5)")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa")
    plt.title("Recompensa en evaluación (100 episodios)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "eval_rewards.png", dpi=150)
    plt.close()

    plt.figure(figsize=(7, 4))
    sns.histplot(rewards, bins=20, kde=True)
    plt.xlabel("Recompensa")
    plt.ylabel("Frecuencia")
    plt.title("Distribución de recompensas (evaluación)")
    plt.tight_layout()
    plt.savefig(fig_dir / "eval_rewards_hist.png", dpi=150)
    plt.close()
