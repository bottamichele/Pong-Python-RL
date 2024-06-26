from enum import Enum

from rl.common.sa.opponent_type import OpponentType

from rl.deep_q_networks.dueling_ddqn.sa.dddqn_training_sa_session import DDDQNTrainingSASession
from rl.deep_q_networks.dueling_ddqn.sa.train_dddqn_sa_app import DDDQNTrainingSAApp

from rl.deep_q_networks.dueling_ddqn.sa_per.dddqn_training_per_sa_session import DDDQNTraining_PER_SASession
from rl.deep_q_networks.dueling_ddqn.sa_per.train_dddqn_per_sa_app import DDDQNTraining_PER_SAApp

from rl.deep_q_networks.dueling_ddqn.sp.dddqn_training_sp_session import DuelingDDQNTrainingSPSession
from rl.deep_q_networks.dueling_ddqn.sp.train_dddqn_sp_app import DDDQNTrainingSPApp

from rl.deep_q_networks.dueling_ddqn.sp_per.dddqn_training_per_sp_session import DuelingDDQNTraining_PER_SPSession
from rl.deep_q_networks.dueling_ddqn.sp_per.train_dddqn_per_sp_app import DDDQNTraining_PER_SPApp

USE_SELF_PLAY = True
USE_PER = True

if __name__ == "__main__":
    if not USE_SELF_PLAY and not USE_PER:
        training_session = DDDQNTrainingSASession(500, OpponentType.BASIC_BOT, 500000, 64, 5000, eps_decay=4.95*10**-6)
        #training_session.load_last_training_session()
        application = DDDQNTrainingSAApp(training_session)
        application.train()
    elif not USE_SELF_PLAY and USE_SELF_PLAY:
        training_session = DDDQNTraining_PER_SASession(250, OpponentType.BASIC_BOT, 500000, 64, 5000, eps_decay=4.95*10**-6)
        #training_session.load_last_training_session()
        application = DDDQNTraining_PER_SAApp(training_session)
        application.train()
    elif USE_SELF_PLAY and not USE_PER:
        training_session = DuelingDDQNTrainingSPSession(500, 750000, 64, 5000, eps_decay=3.96*10**-6, n_policies=6, copy_policy_games=20, change_opp_policy_games=10)
        #training_session.load_last_training_session()
        application = DDDQNTrainingSPApp(training_session)
        application.train()
    elif USE_SELF_PLAY and USE_PER:
        training_session = DuelingDDQNTraining_PER_SPSession(500, 750000, 64, 5000, eps_decay=3.96*10**-6, n_policies=6, copy_policy_games=20, change_opp_policy_games=10)
        training_session.load_last_training_session()
        application = DDDQNTraining_PER_SPApp(training_session)
        application.train()