import torch as tc
from torch.nn import Module, Linear
from torch.nn.functional import relu

class DDQN(Module):
    """A Double Deep Q-Networks (DDQN)."""

    def __init__(self, obs_size, use_cuda=True):
        """Create new DDQN.
        
        Parameters
        --------------------
        obs_size: int
            observation size
            
        use_cuda: bool, optional
            True if cuda is used, False otherwise"""

        super(DDQN, self).__init__()

        #Neural networks's architecture.
        self._fc1 = Linear(obs_size, 256)
        self._fc2 = Linear(256, 256)
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
            x evalueted by DDQN"""
        
        v = relu(self._fc1(x))
        v = relu(self._fc2(v))
        y = self._out(v)

        return y
    
    def copy_from(self, other):
        """
        Copy parameters from another DDQN.
        
        Parameter
        --------------------
        other: DDQN
            a DDQN.
        """
        
        self.load_state_dict(other.state_dict())