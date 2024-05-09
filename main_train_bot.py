from rl.deep_q_networks.ddqn.ddqn_controller import DDQNTrainingSession
from rl.deep_q_networks.ddqn.train_ddqn_app import DDQNTrainingBotApp

from rl.deep_q_networks.dueling_ddqn.dueling_ddqn_controller import DuelingDDQNTrainingSession
from rl.deep_q_networks.dueling_ddqn.train_dueling_ddqn_app import DuelingDDQNTrainingBotApp

TRAIN_DDQN = False
TRAIN_DUELING_DDQN = True

if __name__ == "__main__":
    if TRAIN_DDQN:
        training_session = DDQNTrainingSession(300, 500000, 64, 5000, eps_decay=4.95*10**-6)
        application = DDQNTrainingBotApp(training_session)
        application.train()
    elif TRAIN_DUELING_DDQN:
        training_session = DuelingDDQNTrainingSession(500, 500000, 64, 5000, eps_decay=4.95*10**-6)
        application = DuelingDDQNTrainingBotApp(training_session)
        application.train()