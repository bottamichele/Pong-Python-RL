from pong_app import Pong, ControllerType

if __name__ == "__main__":
    game = Pong(controller_1_type=ControllerType.PLAYER, controller_2_type=ControllerType.DDQN_SP_BOT)
    game.run()