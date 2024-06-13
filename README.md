# Pong-Python-RL
This repository does the use of Reinforcement Learning (RL) algorithms on one of my project I've developed called [Pong](https://github.com/bottamichele/Pong-Python) clone.
Here contains a several RL algorithms to beat and learn playing against itself (self-play technique) on Pong.

## Library used
- [PyGame](https://www.pygame.org/)
- [pybox2D](https://github.com/pybox2d/pybox2d)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)

# Reinforcement Learning methods
RL methods used at the moment are:
- Double Deep Q-Networks (DDQN) ([paper](https://arxiv.org/abs/1509.06461))
- Dueling architeture ([paper](https://arxiv.org/abs/1511.06581))
- Priotirized Experience Replay  ([paper](https://arxiv.org/abs/1511.05952))
- Self-play technique: the technique implemented in this repository is based on some concepts of Unity's [ML-Agents](https://github.com/Unity-Technologies/ml-agents) self-play.
