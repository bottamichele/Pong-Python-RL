from rl.common.sa.opponent_type import OpponentType

from rl.deep_q_networks.dueling_ddqn.sa.dddqn_training_sa_session import DDDQNTrainingSASession
from rl.deep_q_networks.dueling_ddqn.sa.train_dddqn_sa_app import DDDQNTrainingSAApp

from rl.deep_q_networks.dueling_ddqn.sp.dddqn_training_sp_session import DuelingDDQNTrainingSPSession
from rl.deep_q_networks.dueling_ddqn.sp.train_dddqn_sp_app import DDDQNTrainingSPApp

USE_SELF_PLAY = True

if __name__ == "__main__":
    if not USE_SELF_PLAY:
        training_session = DDDQNTrainingSASession(500, OpponentType.BASIC_BOT, 500000, 64, 5000, eps_decay=4.95*10**-6)
        #training_session.load_last_training_session()
        application = DDDQNTrainingSAApp(training_session)
        application.train()
    else:
        training_session = DuelingDDQNTrainingSPSession(500, 750000, 64, 5000, eps_decay=3.96*10**-6, n_policies=6, copy_policy_games=20, change_opp_policy_games=10)
        #training_session.load_last_training_session()
        application = DDDQNTrainingSPApp(training_session)
        application.train()
