from pong_app import Pong, ControllerType

if __name__ == "__main__":
    game = Pong(controller_2_type=ControllerType.BASIC_BOT)
    game.run()