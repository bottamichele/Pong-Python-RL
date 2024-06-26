import torch as tc
import pickle
import os

from rl.common.sp.training_sp_session import Policy, TrainingSPSession, MODEL_PATH
from rl.common.utils import FULL_OBSERVATION_SIZE
from rl.deep_q_networks.common.memory_replay.prop_prio_memory import ProportionalPrioritizedMemory

from torch.optim import Adam
from torch.nn import MSELoss

from .costants import TRAINING_SESSION_PATH, MODEL_NAME
from .dueling_ddqn import DuelingDDQN


# ==================================================
# =============== DUELING DDQN POLICY ==============
# ==================================================

class DuelingDDQNPolicy(Policy):
    """A policy copied from training Dueling DDQN agent."""

    def __init__(self, model):
        """Create agent's policy.
        
        Parameter
        --------------------
        model: DuelingDDQN
            a model."""
        
        self.model = DuelingDDQN(FULL_OBSERVATION_SIZE)
        self.model.load_state_dict(model.state_dict())

# ==================================================
# ======= DUELING DDQN PER SELF-PLAY SESSION =======
# ==================================================

class DuelingDDQNTraining_PER_SPSession(TrainingSPSession):
    """A session for traning of an agent thats uses Dueling DDQN with self-play method and prioritized memory replay."""
    
    def __init__(self, n_episodes, mem_size, batch_size, update_rate_target, lr=10**-4, gamma=0.99, eps_init=1.0, eps_min=0.01, eps_decay=9.9*10**-6, n_policies=8, copy_policy_games=20, change_opp_policy_games=10, play_last_policy_prob=0.5):
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
            
        n_policies: int, optional
            number of agent policies copied

        copy_policy_games: int, optional
            how many games a training agent policy is copied
            
        change_opp_policy_games: int, optional
            how many games a opponent policy is changed
            
        play_last_policy_prob: float, optional
            probability to play against last policy copied"""
        
        super().__init__(n_policies, copy_policy_games, change_opp_policy_games, play_last_policy_prob)

        self.n_episodes = n_episodes
        self.memory = ProportionalPrioritizedMemory(mem_size, FULL_OBSERVATION_SIZE)
        self.batch_size = batch_size
        self.update_rate_target = update_rate_target
        self.lr = lr
        self.gamma = gamma
        self.epsilon = eps_init
        self.epsilon_min = eps_min
        self.epsilon_decay = eps_decay
        self.states_done = 0                                                #Total states done on current episode.
        self.total_states_done = 0                                          #Total states done.

        #Training model.
        self.model = DuelingDDQN(FULL_OBSERVATION_SIZE)
        self.target = DuelingDDQN(FULL_OBSERVATION_SIZE)
        self.optimizer = Adam(self.model.parameters(), lr=lr)

        self.initialize_opponent_policy()

    def is_ended(self):
        return self.episode > self.n_episodes
    
    def get_ta_policy(self):
        """Return training agent's policy.
        
        Return
        --------------------
        ta_policy: DuelingDDQNPolicy
            current training agent's policy"""
        
        return DuelingDDQNPolicy(self.model)

    def save_model(self):
        os.makedirs(MODEL_PATH, exist_ok=True)
        tc.save(self.model.state_dict(), MODEL_PATH + MODEL_NAME + ".pth")
    
    def save_current_training_session(self):
        os.makedirs(TRAINING_SESSION_PATH, exist_ok=True)

        #Save neural networks on disk.
        tc.save(self.model.state_dict(), TRAINING_SESSION_PATH + "model.pth")
        tc.save(self.target.state_dict(), TRAINING_SESSION_PATH + "target.pth")
        
        #Save training session infos on disk.
        current_infos = {"episode": self.episode,
                         "n_episodes": self.n_episodes,
                         "total_states_done": self.total_states_done,
                         "gamma": self.gamma,
                         "batch_size": self.batch_size,
                         "lr": self.lr,
                         "update_rate_target": self.update_rate_target,
                         "epsilon": self.epsilon,
                         "epsilon_min": self.epsilon_min,
                         "epsilon_decay": self.epsilon_decay,
                         "n_policies": self.n_policies,
                         "copy_policy_games": self.copy_policy_games,
                         "change_opp_policy_games": self.change_opp_policy_games,
                         "play_last_policy_prob": self.play_last_policy_prob,
                         "policies_copied": self.policies_copied,
                         "current_opp_policy": self._current_opp_policy}
        
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

        self.episode                    = last_infos["episode"] + 1
        self.n_episodes                 = last_infos["n_episodes"]
        self.total_states_done          = last_infos["total_states_done"]
        self.gamma                      = last_infos["gamma"]
        self.batch_size                 = last_infos["batch_size"]
        learning_rate                   = last_infos["lr"]
        self.update_rate_target         = last_infos["update_rate_target"]
        self.epsilon                    = last_infos["epsilon"]
        self.epsilon_min                = last_infos["epsilon_min"]
        self.epsilon_decay              = last_infos["epsilon_decay"]
        self.n_policies                 = last_infos["n_policies"]
        self.copy_policy_games          = last_infos["copy_policy_games"]
        self.change_opp_policy_games    = last_infos["change_opp_policy_games"]
        self.play_last_policy_prob      = last_infos["play_last_policy_prob"] 
        self.policies_copied            = last_infos["policies_copied"]
        self._current_opp_policy        = last_infos["current_opp_policy"]

        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)

        #Load replay memory from disk.
        memory_replay_file = open(TRAINING_SESSION_PATH + "memory_replay.pkl", "rb")
        self.memory = pickle.load(memory_replay_file)
        memory_replay_file.close()