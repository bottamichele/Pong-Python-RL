from rl.common.sa.opponent_type import OpponentType

from rl.deep_q_networks.ddqn.sa.ddqn_training_sa_session import DDQNTrainingSASession
from rl.deep_q_networks.ddqn.sa.train_ddqn_sa_app import DDQNTrainingSAApp

from rl.deep_q_networks.ddqn.sp.ddqn_training_sp_session import DDQNTrainingSPSession
from rl.deep_q_networks.ddqn.sp.train_ddqn_sp_app import DDQNTrainingSPApp

USE_SELF_PLAY = True

if __name__ == "__main__":
    if not USE_SELF_PLAY:
        training_session = DDQNTrainingSASession(500, OpponentType.BASIC_BOT, 500000, 64, 5000, eps_decay=4.95*10**-6)
        #training_session.load_last_training_session()
        application = DDQNTrainingSAApp(training_session)
        application.train()
    else:
        training_session = DDQNTrainingSPSession(500, 750000, 64, 5000, eps_decay=3.96*10**-6, n_policies=6, copy_policy_games=20, change_opp_policy_games=10)
        training_session.load_last_training_session()
        training_session.n_episodes = 1000
        application = DDQNTrainingSPApp(training_session)
        application.train()
