import numpy as np

from .memory import Memory

class UniformMemory(Memory):
    """A uniform memory replay. It samples a batch randomly from memory."""

    def __init__(self, max_size, obs_size):
        super().__init__(max_size, obs_size)
        self._rng = np.random.default_rng()

    def sample_batch(self, batch_size):
        indices_batch = self._rng.choice(self.size, batch_size, False)
        return self._sample_batch_idxs(indices_batch)