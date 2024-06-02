import numpy as np
import torch as tc

from pong.controller.controller import PaddlePosition
from rl.common.train_bot_controller import TrainingBotController

class DuelingDDQNTraininingSPBotController(TrainingBotController):
    """A bot controller that uses Dueling Double Deep Q-Networks (Dueling DDQN) to be trained on Pong and self-play technique."""

    def __init__(self, current_game, training_session):
        """Create new Dueling DDQN bot controller to be trained.
        
        Parameters
        --------------------
        current_game: Game
            current game session
            
        training_session: DuelingDDQNTrainingSPSession
            training session"""

        self._rng = np.random.default_rng()
           
        super().__init__(current_game.paddle_1, PaddlePosition.LEFT, current_game, training_session)
        self._training_session.states_done = 0

    def _chose_action(self):
        x = tc.Tensor( np.array([self._current_obs]) ).to(self._training_session.model.device)
        q = self._training_session.model.forward(x)

        if self._rng.uniform() <= self._training_session.epsilon:
            #A random action is choosen.
            action = self._rng.integers(0, 3)
        else:
            #Best action is choosen.
            action = tc.argmax(q).item()

        self._current_action = action

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