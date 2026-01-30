from collections import deque, namedtuple
from typing import List

import numpy as np

Experience = namedtuple("Experience", ["state0", "action", "reward", "state1", "terminal1"])


def _zero_observation(obs):
    return np.zeros_like(obs)


class PrioritizedReplayMemory:
    def __init__(
        self,
        limit: int,
        window_length: int = 1,
        alpha: float = 0.6,
        eps: float = 1e-6,
        ignore_episode_boundaries: bool = False,
    ):
        self.limit = limit
        self.window_length = window_length
        self.alpha = alpha
        self.eps = eps
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.observations = deque(maxlen=limit)
        self.actions = deque(maxlen=limit)
        self.rewards = deque(maxlen=limit)
        self.terminals = deque(maxlen=limit)
        self.priorities = deque(maxlen=limit)

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)

    @property
    def nb_entries(self) -> int:
        return len(self.observations)

    def append(self, observation, action, reward, terminal, training=True):
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)
        if not training:
            return

        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminals.append(terminal)
        max_prio = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_prio)

    def get_recent_state(self, current_observation):
        state = list(self.recent_observations)
        state.append(current_observation)
        state = state[-self.window_length :]
        while len(state) < self.window_length:
            state.insert(0, _zero_observation(current_observation))
        return state

    def _get_state(self, idx: int, observations: List, terminals: List):
        state = []
        if not observations:
            return state
        zero_obs = _zero_observation(observations[0])
        for offset in range(self.window_length):
            cur_idx = idx - (self.window_length - 1 - offset)
            if cur_idx < 0:
                state.append(zero_obs)
                continue
            state.append(observations[cur_idx])
        return state

    def sample(self, batch_size: int, return_indices: bool = False, beta: float = None):
        if self.nb_entries < 2:
            raise ValueError("Not enough entries to sample from.")

        max_index = self.nb_entries - 2
        indices = np.arange(0, max_index + 1, dtype=np.int64)
        prios = np.array([self.priorities[i] for i in indices], dtype=np.float64) + self.eps
        if self.alpha <= 0 or prios.sum() == 0:
            probs = np.ones_like(prios) / len(prios)
        else:
            probs = prios ** self.alpha
            probs = probs / probs.sum()

        replace = len(indices) < batch_size
        batch_idxs = np.random.choice(indices, size=batch_size, replace=replace, p=probs)

        observations = list(self.observations)
        actions = list(self.actions)
        rewards = list(self.rewards)
        terminals = list(self.terminals)

        experiences = []
        for idx in batch_idxs:
            state0 = self._get_state(idx, observations, terminals)
            state1 = self._get_state(idx + 1, observations, terminals)
            experiences.append(
                Experience(
                    state0=state0,
                    action=actions[idx],
                    reward=rewards[idx],
                    state1=state1,
                    terminal1=terminals[idx],
                )
            )
        if return_indices:
            weights = None
            if beta is not None:
                # Importance-sampling weights to reduce bias.
                sample_probs = probs[batch_idxs]
                weights = (len(indices) * sample_probs) ** (-beta)
                weights = weights / (weights.max() + self.eps)
            return experiences, batch_idxs, weights
        return experiences

    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            if 0 <= idx < self.nb_entries:
                self.priorities[idx] = float(prio)

    def get_config(self):
        return {
            "limit": self.limit,
            "window_length": self.window_length,
            "alpha": self.alpha,
            "eps": self.eps,
            "ignore_episode_boundaries": self.ignore_episode_boundaries,
        }
