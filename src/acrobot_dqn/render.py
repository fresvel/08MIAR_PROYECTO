import os
import random
from pathlib import Path

import numpy as np

# Ensure headless rendering before importing gym/pyglet.
os.environ.setdefault('PYGLET_HEADLESS', 'true')
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
# If running headless in Docker, ensure DISPLAY is set for Xvfb.
os.environ.setdefault('DISPLAY', ':99')

import gym
import imageio.v2 as imageio


class EpisodeVideoRecorder:
    """Record a single episode to MP4 using rgb_array rendering."""

    def __init__(self, fps=30, max_steps=1000, seed=42):
        self.fps = fps
        self.max_steps = max_steps
        self.seed = seed

    def _set_seeds(self, env):
        random.seed(self.seed)
        np.random.seed(self.seed)
        try:
            env.seed(self.seed)
        except Exception:
            pass

    def record(self, env_name, policy_fn, out_path, reset_fn=None):
        env = gym.make(env_name)
        self._set_seeds(env)
        if reset_fn:
            reset_fn()
        obs = env.reset()
        state = obs[0] if isinstance(obs, tuple) else obs
        frames = []
        frame = env.render(mode="rgb_array")
        if frame is not None:
            frames.append(frame)
        done = False
        steps = 0
        while not done and steps < self.max_steps:
            action = policy_fn(state, env)
            step_out = env.step(action)
            if len(step_out) == 5:
                next_state, reward, terminated, truncated, _ = step_out
                done = terminated or truncated
            else:
                next_state, reward, done, _ = step_out
            state = next_state
            frame = env.render(mode="rgb_array")
            if frame is not None:
                frames.append(frame)
            steps += 1
        env.close()

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not frames:
            raise RuntimeError("No se obtuvieron frames del entorno.")
        imageio.mimsave(out_path, frames, fps=self.fps)
        return out_path
