from pong_app import Pong, ControllerType

if __name__ == "__main__":
    game = Pong(controller_1_type=ControllerType.BASIC_BOT, controller_2_type=ControllerType.DUELING_DDQN_SA_BOT)
    game.run()