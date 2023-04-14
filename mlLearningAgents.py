# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent
from pacman_utils import util
import math

class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    # Constructor of the GameStateFeatures class, taking GameStateFeatures as argument
    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        # Initialise variables, extract useful information from GameState class
        self.pacman_position = state.getPacmanPosition()
        self.ghost_positions = state.getGhostPositions()
        self.food = state.getFood()
        self.food_number = state.getNumFood()
        self.score = state.getScore()
        self.legal = state.getLegalPacmanActions()
        # Remove the STOP action if it is in the legal actions
        if Directions.STOP in self.legal:
            self.legal.remove(Directions.STOP)
    # Returns legal actions Pacman can take
    def getLegalPacmanActions(self):
        return self.legal   
    # Returns number of food pellets
    def getNumFood(self):
        return self.food_number
    # Return hash value of state due  to the initialization of q_map and count as dict, gamestatefeatures as key
    def __hash__(self):
        return hash((self.pacman_position, tuple(self.ghost_positions), self.food))
    # Compare 2 GameStateFeatures objects, 'other' argument: GameStateFeature to compare with
    def __eq__(self, other):
        if not isinstance(other, GameStateFeatures):
            return False
        return (self.pacman_position == other.pacman_position and
                self.ghost_positions == other.ghost_positions and
                self.food == other.food)

class QLearnAgent(Agent):
    # Q-learning agent implementation
    def __init__(self,
                 alpha: float = 0.5,
                 epsilon: float = 0.05,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 20):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = float(alpha)
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # Q map with key: (state, action), value: q-value
        self.q_map = util.Counter()
        # Visitation counts map, key: (state, action), value: number of visits.
        self.state_count = util.Counter()
        # Check if it is in the first action
        self.firstAction = True
        # Add a previous state attribute
        self.prevState = None  
        self.prevAction = None
        self.prevStateFeatures = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self) -> float:
        return self.alpha

    def setAlpha(self, value: float):
        self.alpha = value

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    def getqmap(self):
        return self.q_map
    
    def getcountmap(self):
        return self.state_count

    def updateEpsilon(self):
        self.epsilon *= 0.99

    def updateAlpha(self):
        self.alpha *= 0.99
    
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        "*** YOUR CODE HERE ***"
        # Base reward is the difference of the scores
        base_reward = endState.getScore() - startState.getScore()
        # Additional reward for eating the food
        food_reward = 0
        start_food_count = startState.getNumFood()
        end_food_count = endState.getNumFood()
        if start_food_count > end_food_count:
            food_reward = 10 * (start_food_count - end_food_count)
        # Penalty for being close to a ghost
        penalty_reward = 0
        pacman_position = endState.getPacmanPosition()
        ghost_positions = endState.getGhostPositions()
        dist = [util.manhattanDistance(pacman_position, ghost_position) for ghost_position in ghost_positions]
        if min(dist) <= 2:
            penalty_reward = -20
        # Return total reward 
        return base_reward + food_reward + penalty_reward

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        "*** YOUR CODE HERE ***"
        # Check if the state, action pair is in the q-value
        if (state, action) not in self.q_map:
            # if not, add a default value of 0.01
            self.q_map[(state, action)] = 0.01
        return self.q_map[(state, action)]
    
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            q_value: the maximum estimated Q-value attainable from the state
        """
        "*** YOUR CODE HERE ***"
        # the max q-value is 0 if there no food is located
        if state.getNumFood() == 0:
            q_max = 0

        else:
            # Else, initialise max q-value to -infinity
            q_max = float('-inf')
            # Loop through all legal actions
            for action in state.getLegalPacmanActions():
                q_value = self.getQValue(state, action)
                # Update the max q-value if a higher value is found
                if q_value > q_max:
                    q_max = q_value
        # Return max q-value
        return q_max

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        "*** YOUR CODE HERE ***"
        # Get current state, action pair's q-value 
        q = self.getQValue(state, action)
        # Update q-value
        self.q_map[(state, action)] = q + self.alpha * (reward + self.gamma * self.maxQValue(nextState) - q)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        "*** YOUR CODE HERE ***"
        # Update the number of times an actions was taken in a given state
        self.state_count[(state, action)] += 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        "*** YOUR CODE HERE ***"
        # Return number of times an action was taken in a given state
        return self.state_count[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Return a value based on the counts

        HINT: Do a greed-pick or a least-pick

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        "*** YOUR CODE HERE ***"
        # If count for action is zero, return -infinity.
        # Actions has never been taken, thus encouraging exploration
        if counts == 0:
            return float('inf')
        else:
            # Return utility value + 1 divided by count, encourage agent
            # to explore actions that have been visited less frequently
            return utility + 1/counts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning

        If you wish to use epsilon-greedy exploration, implement it in this method.
        HINT: look at pacman_utils.util.flipCoin

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        # Create a GameStateFeatures object which passes all the features at each state
        StateFeatures = GameStateFeatures(state)
        # If it is not the first step of the game, implement the Q-learning algorithm
        if not self.firstAction:
           reward = self.computeReward(self.prevState, state)
           self.learn(self.prevStateFeatures, self.prevAction, reward, StateFeatures)
        else:
           self.firstAction = False
            
        # Now pick what action to take.
        # A probability of epsilon that chooses a random legal direction for exploration
        if util.flipCoin(self.epsilon):
            action = random.choice(StateFeatures.getLegalPacmanActions())
        
        # Choose direction with the balance of maximum q-value and count
        else:
            all_utility = {}
            for direction in StateFeatures.getLegalPacmanActions():
                # Get the counts and maximum q-value and choose the direction that has the maximum utility
                counts = self.getCount(StateFeatures, direction)
                q_value = self.getQValue(StateFeatures, direction)
                exploration_utility = self.explorationFn(q_value, counts)
                all_utility.update({direction:exploration_utility})
            action = max(all_utility, key=all_utility.get)

        # Update the necessary parameters for Q-learning in the next step
        self.prevState = state
        self.prevAction = action
        self.prevStateFeatures = StateFeatures
        self.updateCount(StateFeatures,action)
        
        return action
    
    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        # Perform a Q-learning update for the last action
        reward = self.computeReward(self.prevState, state)
        StateFeatures = GameStateFeatures(state)
        self.learn(self.prevStateFeatures, self.prevAction, reward, StateFeatures)
        
        # Reset or update the attributes
        self.firstAction = True
        self.updateEpsilon()
        self.updateAlpha()

        print(f"Game {self.getEpisodesSoFar()} just ended!")

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)