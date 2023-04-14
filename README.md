# Pacman AI - Q-Learning Agent

This project is a reinforcement learning agent that uses Q-Learning algorithm to play Pacman, based on the Pacman AI project developed by UC Berkeley. This code implements a Q-learning algorithm to train an agent to play the Pacman game. The QLearnAgent class contains hyperparameters such as the learning rate, exploration rate, and discount factor. The agent selects actions using an epsilon-greedy exploration strategy and updates its Q-values based on rewards and observed next states. The GameStateFeatures class extracts information about the current game state. 

## Prerequisites

- Python 3.7 or higher
- Pygame
- Numpy

## Usage
To run the Pacman game with the Q-Learning agent, navigate to the pacman_utils directory and run:
```
python pacman.py -p QLearnAgent -x 2000 -n 2010 -l smallGrid
```
- p QLearnAgent: specifies the agent to be used.
- x 2000: sets the number of training episodes.
- n 2010: sets the number of testing episodes.
- l smallGrid: specifies the layout of the game.

You can also adjust the hyperparameters of the Q-Learning agent by modifying the `QLearnAgent` class in the `mlLearningAgents.py` file. Following the usage instructions will train the PACMAN to consistently win games in a 5x5 grid.
