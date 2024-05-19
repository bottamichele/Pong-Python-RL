from rl.deep_q_networks.dueling_ddqn.dddqn_training_sa_session import DDDQNTrainingSASession
from rl.deep_q_networks.dueling_ddqn.train_dddqn_sa_app import DDDQNTrainingSAApp

TRAIN_DDQN = False
TRAIN_DUELING_DDQN = True

if __name__ == "__main__":
    training_session = DDDQNTrainingSASession(500, 500000, 64, 5000, eps_decay=4.95*10**-6)
    application = DDDQNTrainingSAApp(training_session)
    application.train()