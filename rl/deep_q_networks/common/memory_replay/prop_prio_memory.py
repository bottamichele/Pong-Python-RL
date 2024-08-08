import numpy as np

from c_python.sum_tree import SumTree

from .memory import Memory

class ProportionalPrioritizedMemory(Memory):
    """A proportional prioritized memory replay. It samples a batch memory in order to transiction's priority."""

    def __init__(self, max_size, obs_size, alpha=0.6, beta=0.4, eps=10**-5):
        """Create new memory replay.
        
        Parameters
        --------------------
        max_size: int
            maximum size of memory replay
            
        obs_size: int
            observation size
            
        alpha: float, optional
            priority factor
            
        beta: float, optional
            importance sample factor
            
        eps: float, optional
            small value for avoid division by zero"""
        
        super().__init__(max_size, obs_size)

        self._priorities = np.zeros(max_size, dtype=np.float32)             #Priority for each transiction.
        self._cum_prios = SumTree(max_size)                                 #Cumulative priorities.
        self.alpha = alpha
        self.beta = beta
        self._epsilon = eps                                                 #Small value epsilon.
        self._idxs_sampled = None                                           #Last sample of batch indices

    def __reduce__(self):
        reduce_state = super().__reduce__()

        if "_cum_prios" in reduce_state[2].keys():
            dict_state = reduce_state[2].copy()
            dict_state.pop("_cum_prios")
            return reduce_state[0], reduce_state[1], dict_state
        else:
            return reduce_state

    def __setstate__(self, state):
        self.__dict__.update(state)

        self._cum_prios = SumTree(self._max_size)
        for i in range(self._current_size):
            self._cum_prios.set_priority(i, self._priorities[i] ** self.alpha)

    def store_transiction(self, obs, action, reward, next_obs, next_obs_done):
        #Set priority for current transiction.
        prio = np.max(self._priorities) if self._current_size > 0 else 1.0
        
        self._priorities[self._current_idx] = prio
        self._cum_prios.set_priority(self._current_idx, prio**self.alpha)

        #Store transiction on memory replay.
        super().store_transiction(obs, action, reward, next_obs, next_obs_done)

    def sample_batch(self, batch_size):
        """Sample a batch from memory replay.
        
        Parameter
        --------------------
        batch_size: int
            batch size to sample
            
        Returns
        --------------------
        obs_batch: ndarray
            observation batch sampled from memory replay
            
        action_batch: ndarray
            action batch sampled from memory replay
            
        reward_batch: ndarray
            reward batch sampled from memory replay
            
        next_batch: ndarray
            next observations batch sampled from memory replay
            
        next_obs_done_batch: ndarray
            next observations done batch sampled from memory replay
            
        weights_batch: ndarray
            weights batch sampled from memory replay"""
        
        #Sample index of transictions and probabilties
        idxs = self._cum_prios.sample_batch(batch_size)
        probs = self._cum_prios.get_probability_of_batch(idxs)

        #Compute weights.
        weights = (self._max_size * probs)**(-self.beta)
        weights /= np.max(weights)

        self._idxs_sampled = idxs
        obs_b, action_b, reward_b, next_obs_b, next_obs_done_b = self._sample_batch_idxs(idxs)
        return obs_b, action_b, reward_b, next_obs_b, next_obs_done_b, weights
    
    def update_priorities(self, td_errors):
        """Update priorities of current batch sampled.
        
        Parameter
        --------------------
        td_errors: ndarray
            temporal difference errors"""
        
        assert len(self._idxs_sampled) == len(td_errors)

        for i in range(self._idxs_sampled.size):
            idx = self._idxs_sampled[i]
            self._priorities[idx] = abs(td_errors[i]) + self._epsilon