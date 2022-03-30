import numpy as np


class AgentHistory:
    @property
    def length(self):
        return self._length

    def __init__(self, max_length):
        # Experience replay buffers
        self._action_history = []
        self._observation_history = []
        self._new_observation_history = []
        self._rewards_history = []
        self._done_history = []
        # Current buffer length
        self._length = 0
        # Maximum replay length
        self._max_length = max_length

    def update(self, action, state, state_next, done, reward):
        # Save actions and states in replay buffer
        self._action_history.append(action)
        self._observation_history.append(state)
        self._new_observation_history.append(state_next)
        self._done_history.append(float(done))
        self._rewards_history.append(reward)
        self._length += 1
        # Trim buffers when over capacity
        if self._length >= self._max_length:
            del self._rewards_history[:1]
            del self._observation_history[:1]
            del self._new_observation_history[:1]
            del self._action_history[:1]
            del self._done_history[:1]
            del self._done_history[:1]

    def get_sample(self, size):
        # Get indices of samples for replay buffers
        indices = np.random.choice(range(self._length), size=size)

        # Using list comprehension to sample from replay buffer
        state_sample = np.array([self._observation_history[i] for i in indices])
        state_next_sample = np.array([self._new_observation_history[i] for i in indices])
        rewards_sample = np.array([self._rewards_history[i] for i in indices])
        action_sample = np.array([self._action_history[i] for i in indices])
        done_sample = np.array([self._done_history[i] for i in indices])
        return state_sample, state_next_sample, rewards_sample, action_sample, done_sample

