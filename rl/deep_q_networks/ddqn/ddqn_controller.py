import torch as tc
import numpy as np
import os
import pickle

from torch.optim import Adam
from torch.nn import MSELoss
from rl.common.train_single_agent import TrainingSession, TrainingBotController
from rl.common.utils import FULL_OBSERVATION_SIZE, get_full_observation_normalized, get_full_inverse_observation_normalized
from pong.controller.controller import Controller, PaddlePosition, MovingType

from ..common.memory_replay import UniformMemory
from .ddqn import DDQN

# ==================================================
# ================ GLOBAL VARIABLES ================
# ==================================================

TRAINING_PATH = "./rl/deep_q_networks/ddqn/models/"
MODEL_PATH = "./rl/models/v.s. Bot/"
TRAINING_SESSION_PATH = TRAINING_PATH + "training_session/"
MODEL_NAME = "pong_ddqn.pth"

# ==================================================
# ============== DDQN Bot Controller ===============
# ==================================================

class DDQNBotController(Controller):
    """A bot controller that uses Double Deep Q-Networks (DDQN) to control a paddle."""

    def __init__(self, a_paddle, position, current_game):
        """Create DDQN controller.
        
        Parameters
        --------------------
        a_paddle: Paddle
            paddle to control with this controller
            
        position: PaddlePosition
            paddle position this controller controlls
            
        current_game: Game
            current game session"""
        
        super().__init__(a_paddle, position)
        self._opponent_paddle = current_game.paddle_2 if position == PaddlePosition.LEFT else current_game.paddle_1
        self._ball = current_game.ball
        self._current_game = current_game
        self._model = DDQN(FULL_OBSERVATION_SIZE)                       #Neural network used for playing.
        self._get_obs_fun = get_full_observation_normalized if position == PaddlePosition.LEFT else get_full_inverse_observation_normalized

        #Load model trained.
        self._model.load_state_dict(tc.load(MODEL_PATH + MODEL_NAME))
            
    def update(self, delta_time):
        current_observation = self._get_obs_fun(self._current_game)
        
        #Choose action to perform.
        x = tc.Tensor(np.array([current_observation])).to(self._model.device)
        q = self._model.forward(x)
        action = tc.argmax(q).item()

        #Perform action choosen.
        self._move_paddle(MovingType(action))


# ==================================================
# ============== DDQN TRAINING SESSION =============
# ==================================================

class DDQNTrainingSession(TrainingSession):
    """Training session for training a DDQN bot."""
    
    def __init__(self, n_episodes, mem_size, batch_size, update_rate_target, lr=10**-4, gamma=0.99, eps_init=1.0, eps_min=0.01, eps_decay=10**-5):
        """Create new DDQN training session.
        
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
            
        eps_decay: float, optional"""
        
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
        #self.q_values = deque(maxlen=10000)

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
        tc.save(self.model.state_dict(), MODEL_PATH + MODEL_NAME)
        tc.save(self.model.state_dict(), TRAINING_PATH + MODEL_NAME)
    
    def save_current_training_session(self):
        os.makedirs(TRAINING_SESSION_PATH, exist_ok=True)

        #Save neural networks on disk.
        tc.save(self.model.state_dict(), TRAINING_SESSION_PATH + "model.pth")
        tc.save(self.target.state_dict(), TRAINING_SESSION_PATH + "target.pth")
        
        #Save training session infos on disk.
        current_infos = {"episode": self.episode,
                         "n_episodes": self.n_episodes,
                         "total_states_done": self.total_states_done,
                         "history_scores": self.history_scores, 
                         "history_rewards": self.history_rewards,
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
        self.total_states_done  = last_infos["total_states_done"]
        self.history_scores     = last_infos["history_scores"]
        self.history_rewards    = last_infos["history_rewards"]
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


# ==================================================
# ========== DDQN TRAINING BOT CONTROLLER ==========
# ==================================================

class DDQNTraininingBotController(TrainingBotController):
    """A bot controller that uses Double Deep Q-Networks (DDQN) to be trainined playing Pong."""

    def __init__(self, a_paddle, position, current_game, training_session):
        """Create new DDQN bot controller to be tranined.
        
        Parameters
        --------------------
        a_paddle: Paddle
            paddle to control
            
        position: PaddlePosition
            paddle position that this controller takes control
            
        current_game: Game
            current game session
            
        training_session: DDQNTrainingSession
            training session"""
        
        super().__init__(a_paddle, position, current_game, training_session)
        self._training_session.states_done = 0
        self._rng = np.random.default_rng()

    def _choose_action(self):
        x = tc.Tensor(np.array([self._current_obs])).to(self._training_session.model.device)
        q = self._training_session.model.forward(x)

        if self._rng.uniform() <= self._training_session.epsilon:
            #A random action is choosen.
            action = self._rng.integers(0, 3)
        else:
            #Best action is choosen.
            action = tc.argmax(q).item()

        self._current_action = action
        #return action, (q.cpu())[0, action].item()

    def _train_step(self):
        #Store transiction on memory replay.
        self._training_session.memory.store_transiction(self._current_obs, self._current_action, self._current_reward, self._current_next_obs, self._is_terminated)
        
        if self._training_session.memory.size < self._training_session.batch_size:
            return
        
        #A minibatch is built.
        obs_b, action_b, reward_b, next_obs_b, next_obs_done_b = self._training_session.memory.sample_batch(self._training_session.batch_size)
        obs_b = tc.Tensor(obs_b).to(self._training_session.model.device)
        next_obs_b = tc.Tensor(next_obs_b).to(self._training_session.model.device)
        reward_b = tc.Tensor(reward_b).to(self._training_session.model.device)

        #Compute q-values.
        idxs = np.arange(0, self._training_session.batch_size, 1, dtype=np.int32)
        
        q = self._training_session.model.forward(obs_b)[idxs, action_b]
        
        best_action_batch = np.array( tc.argmax(self._training_session.model.forward(next_obs_b), dim=1).cpu() )
        q_next = self._training_session.target.forward(next_obs_b)[idxs, best_action_batch]
        q_next[next_obs_done_b] = 0.0

        q_target = reward_b + self._training_session.gamma * q_next

        #Do training step.
        self._training_session.optimizer.zero_grad()
        loss = self._training_session.loss_function(q, q_target).to(self._training_session.model.device)
        loss.backward()
        self._training_session.optimizer.step()

        #Update epsilon
        self._training_session.epsilon = self._training_session.epsilon - self._training_session.epsilon_decay if self._training_session.epsilon > self._training_session.epsilon_min else self._training_session.epsilon_min

    def _on_post_train_step(self):
        self._training_session.states_done += 1
        self._training_session.total_states_done += 1

        #Update target net.
        if self._training_session.total_states_done % self._training_session.update_rate_target == 0:
            self._training_session.target.copy_from(self._training_session.model)