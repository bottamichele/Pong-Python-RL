import torch as tc
import pickle
import os

from rl.common.sp.training_sp_session import TrainingSPASession
from rl.common.utils import FULL_OBSERVATION_SIZE
from rl.deep_q_networks.common.memory_replay import UniformMemory

from collections import deque

from torch.optim import Adam
from torch.nn import MSELoss

from .costants import TRAINING_SESSION_PATH, MODEL_PATH, MODEL_NAME
from .dueling_ddqn import DuelingDDQN


# ==================================================
# ===================== POLICY =====================
# ==================================================
class Policy:
    """An agent's policy to be trained."""

    def __init__(self, lr):
        """Create new policy to train.
        
        Parameter
        --------------------
        lr: float
            learning rate"""
        
        self.model = DuelingDDQN(FULL_OBSERVATION_SIZE)                 
        self.target = DuelingDDQN(FULL_OBSERVATION_SIZE)                
        self.optimizer = Adam(self.model.parameters(), lr=lr)           
        self.total_states = 0                                           #Number of total states done with this policy.
        self.count_episodes = 0                                         #Number of episodes done with this policy.

        self.target.copy_from(self.model)

# ==================================================
# ========= DUELING DDQN SELF PLAY SESSION =========
# ==================================================

class DuelingDDQNTrainingSPASession(TrainingSPASession):
    """A session for traning of an agent thats uses Dueling DDQN with self-play method."""
    
    def __init__(self, n_episodes, mem_size, batch_size, update_rate_target, lr=10**-4, gamma=0.99, eps_init=1.0, eps_min=0.01, eps_decay=9.9*10**-6, stack_pol_size=8):
        """Create new Dueling DDQN training session with self-play method.
        
        Parameters
        --------------------
        n_episodes: int
            number of episodes to train bot.

        mem_size: int
            memory replay size

        batch_size: int
            batch size

        update_rate_target: int
            how many steps to wait to update target net.
            
        lr: float, optional
            learning rate
            
        gamma: float, optional
            discount factor
            
        eps_init: float, optional
            initial epsilon value
            
        eps_min: float, optional
            minimun epsilon value allowed
            
        eps_decay: float, optional
            epsilon decay value
            
        stack_pol_size: int, optional
            stacl policies size"""
        
        super().__init__()
        self.n_episodes = n_episodes
        self.memory = UniformMemory(mem_size, FULL_OBSERVATION_SIZE)
        self.batch_size = batch_size
        self.update_rate_target = update_rate_target
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_init
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.states_done = 0                                                #Total states done on current episode.
        self.total_states_done = 0                                          #Total states done.

        #Training models.
        self.stack_policies = deque(maxlen=stack_pol_size)
        self.current_policy = Policy(lr)
        self.model_p1 = DuelingDDQN(FULL_OBSERVATION_SIZE)
        self.target_p1 = DuelingDDQN(FULL_OBSERVATION_SIZE)
        self.optimizer_p1 = Adam(self.model_p1.parameters(), lr=lr)

        #Training model for right controller.
        self.model_p2 = DuelingDDQN(FULL_OBSERVATION_SIZE)
        self.target_p2 = DuelingDDQN(FULL_OBSERVATION_SIZE)
        self.optimizer_p2 = Adam(self.model_p2.parameters(), lr=lr)

        self.loss_function = MSELoss()

    def is_ended(self):
        return self.episode > self.n_episodes

    def save_model(self):
        os.makedirs(MODEL_PATH, exist_ok=True)
        tc.save(self.model_p1.state_dict(), MODEL_PATH + MODEL_NAME + ".pth")
    
    def save_current_training_session(self):
        os.makedirs(TRAINING_SESSION_PATH, exist_ok=True)

        #Save neural networks on disk.
        tc.save(self.model_p1.state_dict(), TRAINING_SESSION_PATH + "model_p1.pth")
        tc.save(self.target_p1.state_dict(), TRAINING_SESSION_PATH + "target_p1.pth")
        tc.save(self.model_p2.state_dict(), TRAINING_SESSION_PATH + "model_p2.pth")
        tc.save(self.target_p2.state_dict(), TRAINING_SESSION_PATH + "target_p2.pth")
        
        #Save training session infos on disk.
        current_infos = {"episode": self.episode,
                         "n_episodes": self.n_episodes,
                         "total_states_done": self.total_states_done,
                         "history_scores_p1": self.history_scores_p1, 
                         "history_rewards_p1": self.history_rewards_p1,
                         "history_q_p1": self.history_q_p1,
                         "history_scores_p2": self.history_scores_p2, 
                         "history_rewards_p2": self.history_rewards_p2,
                         "history_q_p2": self.history_q_p2,
                         "gamma": self.gamma,
                         "batch_size": self.batch_size,
                         "lr": self.lr,
                         "update_rate_target": self.update_rate_target,
                         "epsilon": self.epsilon,
                         "epsilon_min": self.epsilon_min,
                         "epsilon_decay": self.epsilon_decay}
        
        training_session_file = open(TRAINING_SESSION_PATH + "training_session_infos.pkl", "wb")
        pickle.dump(current_infos, training_session_file)
        training_session_file.close()

        #Save memory replay on disk.
        memory_replay_p1_file = open(TRAINING_SESSION_PATH + "memory_replay_p1.pkl", "wb")
        pickle.dump(self.memory_p1, memory_replay_p1_file)
        memory_replay_p1_file.close()

        memory_replay_p2_file = open(TRAINING_SESSION_PATH + "memory_replay_p2.pkl", "wb")
        pickle.dump(self.memory_p2, memory_replay_p2_file)
        memory_replay_p2_file.close() 

    def load_last_training_session(self):
        #Load neural networks from disk.
        self.model_p1.load_state_dict( tc.load(TRAINING_SESSION_PATH + "model_p1.pth") )
        self.target_p1.load_state_dict( tc.load(TRAINING_SESSION_PATH + "target_p1.pth") )
        self.model_p2.load_state_dict( tc.load(TRAINING_SESSION_PATH + "model_p2.pth") )
        self.target_p2.load_state_dict( tc.load(TRAINING_SESSION_PATH + "target_p2.pth") )

        #Load last training session saved on disk.
        training_session_file = open(TRAINING_SESSION_PATH + "training_session_infos.pkl", "rb")
        last_infos = pickle.load(training_session_file)
        training_session_file.close()

        self.episode            = last_infos["episode"]
        self.n_episodes         = last_infos["n_episodes"]
        self.total_states_done  = last_infos["total_states_done"]
        self.history_scores_p1  = last_infos["history_scores_p1"]
        self.history_rewards_p1 = last_infos["history_rewards_p1"]
        self.history_q_p1       = last_infos["history_q_p1"]
        self.history_scores_p2  = last_infos["history_scores_p2"]
        self.history_rewards_p2 = last_infos["history_rewards_p2"]
        self.history_q_p2       = last_infos["history_q_p2"]
        self.gamma              = last_infos["gamma"]
        self.batch_size         = last_infos["batch_size"]
        learning_rate           = last_infos["lr"]
        self.update_rate_target = last_infos["update_rate_target"]
        self.epsilon            = last_infos["epsilon"]
        self.epsilon_min        = last_infos["epsilon_min"]
        self.epsilon_decay      = last_infos["epsilon_decay"]

        self.optimizer_p1 = Adam(self.model_p1.parameters(), lr=learning_rate)
        self.optimizer_p2 = Adam(self.model_p1.parameters(), lr=learning_rate)

        #Load replay memory from disk.
        memory_replay_p1_file = open(TRAINING_SESSION_PATH + "memory_replay_p1.pkl", "rb")
        self.memory_p1 = pickle.load(memory_replay_p1_file)
        memory_replay_p1_file.close()

        memory_replay_p2_file = open(TRAINING_SESSION_PATH + "memory_replay_p2.pkl", "rb")
        self.memory_p2 = pickle.load(memory_replay_p2_file)
        memory_replay_p2_file.close()