import torch as tc
from torch.nn import Module, Linear
from torch.nn.functional import relu

class DuelingDDQN(Module):
    """A Dueling Double Deep Q-Networks (Dueling DDQN)."""

    def __init__(self, obs_size, use_cuda=True):
        """Create new Dueling DDQN.
        
        Parameters
        --------------------
        obs_size: int
            observation size
            
        use_cuda: bool, optional
            True if cuda is used, False otherwise"""

        super(DuelingDDQN, self).__init__()

        #Neural networks's architecture.
        self._fc1 = Linear(obs_size, 256)
        self._fc2 = Linear(256, 256)
        self._value = Linear(256, 1)
        self._advantage = Linear(256, 3)
        self._out = Linear(256, 3)

        # --------------------
        self.device = tc.device("cuda:0" if tc.cuda.is_available() and use_cuda else "cpu")
        self.to(self.device)

    def forward(self, x):
        """Evalue x.
        
        Parameter
        --------------------
        x: Tensor
            a tensor
            
        Return
        --------------------
        y: Tensor
            x evalueted by Dueling DDQN"""
        
        v = relu(self._fc1(x))
        v = relu(self._fc2(v))
        value = self._value(v)
        advantage = self._advantage(v)
        y = value + advantage - advantage.mean()

        return y
    
    def copy_from(self, other):
        """
        Copy parameters from another Dueling DDQN.
        
        Parameter
        --------------------
        other: DuelingDDQN
            a Dueling DDQN.
        """
        
        self.load_state_dict(other.state_dict())