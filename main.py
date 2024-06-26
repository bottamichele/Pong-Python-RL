from pong_app import Pong, ControllerType

if __name__ == "__main__":
    game = Pong(controller_1_type=ControllerType.PLAYER, controller_2_type=ControllerType.DUELING_DDQN_PER_SP_BOT)
    game.run()