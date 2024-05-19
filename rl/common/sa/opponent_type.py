from enum import Enum

class OpponentType(Enum):
    """Opponent type against to train"""
    BASIC_BOT = 0           #Train against a basic bot.
    BOT = 1                 #Train against a high skill bot.