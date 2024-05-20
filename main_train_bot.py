from rl.common.sa.opponent_type import OpponentType

from rl.deep_q_networks.dueling_ddqn.sa.dddqn_training_sa_session import DDDQNTrainingSASession
from rl.deep_q_networks.dueling_ddqn.sa.train_dddqn_sa_app import DDDQNTrainingSAApp

TRAIN_DDQN = False
TRAIN_DUELING_DDQN = True

if __name__ == "__main__":
    training_session = DDDQNTrainingSASession(500, OpponentType.BASIC_BOT, 500000, 64, 5000, eps_decay=4.95*10**-6)
    training_session.load_last_training_session()
    application = DDDQNTrainingSAApp(training_session)
    application.train()