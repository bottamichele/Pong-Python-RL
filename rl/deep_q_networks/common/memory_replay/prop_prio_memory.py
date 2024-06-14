import numpy as np

from math import log2

from .memory import Memory

# ==================================================
# =============== CUMULATIVE PRIORITY ==============
# ==================================================

class CumulativePriority:
    """A complete binary tree that contains cumulative priorities (similar Prioritzed Experience Replay papar's Sum Tree)."""

    def __init__(self, num_leaves, alpha):
        """Create cumulative priority tree.
        
        Parameter
        --------------------
        num_leaves: int
            number of leaves of tree
            
        alpha: float
            alpha value of Proportional Prioritized Memory"""
        
        self._n_leaves = num_leaves                                                                                 #Number of leaves of binary tree.
        self._depth = int(log2(num_leaves)) + 1 if isinstance(log2(num_leaves), float) else int(log2(num_leaves))   #Depth of binary tree.
        self._tree = np.zeros(2**(self._depth + 1) - 1, dtype=np.float32)                                           #Binary tree.
        self._alpha = alpha

    def _get_parent_index(self, idx_node):
        """Return parent's index of a node.
        
        Parameter
        --------------------
        idx_node: int
            index of a node
            
        Return
        --------------------
        idx_parent: int
            index of parent or -1 if idx_node is root node"""
        
        assert 0 <= idx_node and idx_node <= self._tree.size - 1
        
        #Is idx_node root node?
        if idx_node == 0:
            return -1
        
        return (idx_node + 1) // 2 - 1
    
    def _get_children_indices(self, idx_node):
        """Return indices of left child and right child of a node.
        
        Parameter
        --------------------
        idx_node: int
            index of a node
            
        Returns
        --------------------
        idx_left_child: int
            index of left child or -1 if idx_node does't have any left child
            
        idx_right_child: int
            idex of right child or -1 if idx_node doesn't have any right child"""

        assert 0 <= idx_node and idx_node <= self._tree.size - 1

        idx_left_c = 2 * idx_node + 1       # <--- 2 * (idx_node + 1) - 1
        idx_right_c = 2 * idx_node + 2      # <--- 2 * (idx_node + 1) + 1 - 1

        #Is idx_node a leaf node?
        if 2**self._depth - 1 <= idx_node and idx_node <= self._tree.size - 1:
            idx_left_c = -1
            idx_right_c = -1

        return idx_left_c, idx_right_c

    def set_priority(self, idx, prio):
        """Set a transiction's priority on tree.
        
        Parameters
        --------------------
        idx: int
            index of transiction
            
        prio: float
            priority of transiction"""
        
        assert 0 <= idx and idx <= self._n_leaves - 1
        
        #Set priority of a transiction.
        self._tree[2**self._depth - 1 + idx] = prio ** self._alpha

        #Update cumulative priorities that have current transiction as leaf node.
        idx_parent = self._get_parent_index(2**self._depth - 1 + idx)

        while idx_parent != -1:
            idx_left_child, idx_right_child = self._get_children_indices(idx_parent)
            self._tree[idx_parent] = self._tree[idx_left_child] + self._tree[idx_right_child]

            idx_parent = self._get_parent_index(idx_parent)

    def get_random_transiction(self):
        """Return a transiction randomly.
        
        Return
        --------------------
        idx_trans: int
            index of transiction"""
        
        p = np.random.default_rng().uniform(0, self._tree[0])
        up = self._tree[0]
        
        idx_node = 0
        idx_lc, idx_rc = self._get_children_indices(idx_node)

        while idx_lc != -1 and idx_rc != -1:
            if p <= up - self._tree[idx_rc]:
                up -= self._tree[idx_rc]
                idx_node = idx_lc
            else:
                idx_node = idx_rc

            idx_lc, idx_rc = self._get_children_indices(idx_node)

        return idx_node - (2**self._depth - 1)
    
    def get_probability_of_transiction(self, idx):
        """Return probability of a transiction.
        
        Parameter
        --------------------
        idx: int
            index of a transiction

        Return
        --------------------
        prob: float
            probability of transiction"""
        
        assert 0 <= idx and idx <= self._n_leaves - 1

        return self._tree[2**self._depth - 1 + idx] / self._tree[0]
            
# ==================================================
# ========= PROPORTIONAL PRIORITIZED MEMORY ========
# ==================================================

class ProportionalPrioritizedMemory(Memory):
    """A proportional prioritized memory replay. It samples a batch memory in order to transiction's priority."""

    def __init__(self, max_size, obs_size, alpha=0.6, beta=0.4, eps=10**-4):
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
        self._cum_prios = CumulativePriority(max_size, alpha)               #Cumulative priorities.
        self.alpha = alpha
        self.beta = beta
        self._epsilon = eps                                                 #Small value epsilon.
        self._idxs_sampled = None                                           #Last sample of batch indices

    def store_transiction(self, obs, action, reward, next_obs, next_obs_done):
        #Set priority for current transiction.
        prio = np.max(self._priorities) if self._current_size > 0 else 1.0
        
        self._priorities[self._current_idx] = prio
        self._cum_prios.set_priority(self._current_idx, prio)

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
        idxs = np.zeros(batch_size, dtype=np.int32)
        probs = np.zeros(batch_size, dtype=np.float32)

        for i in range(batch_size):
            #Sample until a index of transiction is not a duplicate
            idx_trans_sampled = self._cum_prios.get_random_transiction()
            while (idx_trans_sampled + 1) in (idxs + 1):
                idx_trans_sampled = self._cum_prios.get_random_transiction()

            #Store transiction sampled.
            idxs[i] = idx_trans_sampled
            probs[i] = self._cum_prios.get_probability_of_transiction(idx_trans_sampled)

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