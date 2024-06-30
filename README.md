# Pong-Python-RL
This repository contains Reinforcement Learning (RL) approaches applied on my project [Pong clone](https://github.com/bottamichele/Pong-Python).
This repository implements two type of agent:
- a single agent is trained against one of Pong bots to play against one of them
- an agent is trained against itself to play against Pong bots and human players
The relative code of RL implemented is located on ["rl"](https://github.com/bottamichele/Pong-Python-RL/tree/main/rl) folder. 
Pre-trained models for relative approaches are available and are located on ["models"](https://github.com/bottamichele/Pong-Python-RL/tree/main/rl/models) folder under "rl" folder, 
any one of them is loaded when you choose the corrispondent type of controller to use on [main.py](https://github.com/bottamichele/Pong-Python-RL/blob/main/main.py).

# Reinforcement Learning approach
RL approaches applied, for now, are:
- Double Deep Q-Networks (DDQN) ([paper](https://arxiv.org/abs/1509.06461))
- Dueling architeture ([paper](https://arxiv.org/abs/1511.06581))
- Prioritized Experience Replay (PER) ([paper](https://arxiv.org/abs/1511.05952))
- Self-Play technique: the technique implemented is based on some concepts of Unity's [ML-Agents](https://github.com/Unity-Technologies/ml-agents) self-play.

## Library used
- [PyGame](https://www.pygame.org/)
- [pybox2D](https://github.com/pybox2d/pybox2d)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)