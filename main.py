from pong_app import Pong, ControllerType

if __name__ == "__main__":
    game = Pong(controller_1_type=ControllerType.DUELING_DDQN_SP_BOT, controller_2_type=ControllerType.BOT)
    game.run()