import torch as tc
import pickle
import os

from rl.common.sa.training_sa_session import TrainingSASession
from rl.common.utils import FULL_OBSERVATION_SIZE
from rl.deep_q_networks.common.memory_replay import UniformMemory

from collections import deque

from torch.optim import Adam
from torch.nn import MSELoss

from .costants import TRAINING_SESSION_PATH, MODEL_PATH, MODEL_NAME
from .ddqn import DDQN

class DDQNTrainingSASession(TrainingSASession):
    """A session for traning of a single agent thats uses DDQN."""
    
    def __init__(self, n_episodes, opponent_type, mem_size, batch_size, update_rate_target, lr=10**-4, gamma=0.99, eps_init=1.0, eps_min=0.01, eps_decay=9.9*10**-6):
        """Create new DDQN training session.
        
        Parameters
        --------------------
        n_episodes: int
            number of episodes to train bot.

        opponent_type: OpponentType
            a opponent type against to train.

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
            
        eps_decay: float, optional"""
        
        super().__init__(opponent_type)
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
        self.history_q = deque(maxlen=25000)

        #Training model.
        self.model = DDQN(FULL_OBSERVATION_SIZE)
        self.target = DDQN(FULL_OBSERVATION_SIZE)
        self.loss_function = MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        self.target.copy_from(self.model)

    def is_ended(self):
        return self.episode > self.n_episodes

    def save_model(self):
        os.makedirs(MODEL_PATH, exist_ok=True)
        tc.save(self.model.state_dict(), MODEL_PATH + MODEL_NAME + "_vs_" + self.opponent_type.name + ".pth")
    
    def save_current_training_session(self):
        os.makedirs(TRAINING_SESSION_PATH, exist_ok=True)

        #Save neural networks on disk.
        tc.save(self.model.state_dict(), TRAINING_SESSION_PATH + "model.pth")
        tc.save(self.target.state_dict(), TRAINING_SESSION_PATH + "target.pth")
        
        #Save training session infos on disk.
        current_infos = {"episode": self.episode,
                         "n_episodes": self.n_episodes,
                         "opponent_type": self.opponent_type,
                         "total_states_done": self.total_states_done,
                         "history_scores": self.history_scores, 
                         "history_rewards": self.history_rewards,
                         "history_q": self.history_q,
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
        memory_replay_file = open(TRAINING_SESSION_PATH + "memory_replay.pkl", "wb")
        pickle.dump(self.memory, memory_replay_file)
        memory_replay_file.close() 

    def load_last_training_session(self):
        #Load neural networks from disk.
        self.model.load_state_dict( tc.load(TRAINING_SESSION_PATH + "model.pth") )
        self.target.load_state_dict( tc.load(TRAINING_SESSION_PATH + "target.pth") )

        #Load last training session saved on disk.
        training_session_file = open(TRAINING_SESSION_PATH + "training_session_infos.pkl", "rb")
        last_infos = pickle.load(training_session_file)
        training_session_file.close()

        self.episode            = last_infos["episode"]
        self.n_episodes         = last_infos["n_episodes"]
        self.opponent_type      = last_infos["opponent_type"]
        self.total_states_done  = last_infos["total_states_done"]
        self.history_scores     = last_infos["history_scores"]
        self.history_rewards    = last_infos["history_rewards"]
        self.history_q          = last_infos["history_q"]
        self.gamma              = last_infos["gamma"]
        self.batch_size         = last_infos["batch_size"]
        learning_rate           = last_infos["lr"]
        self.update_rate_target = last_infos["update_rate_target"]
        self.epsilon            = last_infos["epsilon"]
        self.epsilon_min        = last_infos["epsilon_min"]
        self.epsilon_decay      = last_infos["epsilon_decay"]

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        #Load replay memory from disk.
        memory_replay_file = open(TRAINING_SESSION_PATH + "memory_replay.pkl", "rb")
        self.memory = pickle.load(memory_replay_file)
        memory_replay_file.close()