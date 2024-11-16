# multi_agents.py
# --------------
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


from util import manhattan_distance
from game import Directions, Actions
from pacman import GhostRules
import random, util
from game import *
from itertools import product
import copy

inf = 99999999
max_depth = 2


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        Just like in the previous project, get_action takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legal_moves = game_state.get_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (new_food) and Pacman position after moving (new_pos).
        new_scared_times holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successor_game_state = current_game_state.generate_pacman_successor(action)
        new_pos = successor_game_state.get_pacman_position()
        new_food = successor_game_state.get_food()
        score = successor_game_state.get_score()
        new_ghost_states = successor_game_state.get_ghost_states()
        new_scared_times = [ghostState.scared_timer for ghostState in new_ghost_states]
        # Step 1: We first calculate the food score, this score increases if pacman has a smaller distance to food spots
        food_pos = new_food.as_list()
        distances = [manhattan_distance(new_pos, food) for food in food_pos]
        min_distance = min(distances) if distances else 0
        # The food score is 1 / min distance if there are remaining food spots or
        # 100 if the pacman eats the last food with this move
        food_score = 1 / min_distance if min_distance > 0 else 100
        # Step 2: We calculate the ghost score, the further pacman is away from the ghosts, the better the score
        # BUT: If a ghost is scared, the distance doesn't matter, the score increases if pacman gets closer to them
        ghost_distances = [manhattan_distance(new_pos, ghost.get_position()) for ghost in new_ghost_states]
        ghost_score = 0
        for idx, dis in enumerate(ghost_distances):
            if new_scared_times[idx] > 0:
                ghost_score += 10 / dis if dis > 0 else 1
            else:  # Active ghost (avoid)
                ghost_score -= 1 / dis if dis > 0 else 10
        # Step 3: Finally we create a punishment for still existing food pieces.
        # Each remaining food influences the score negatively
        remaining_food = len(food_pos)
        food_score -= remaining_food
        # The final score consist of 1. The game score, 2. The ghost score (distance to ghost, scared / not scared)
        # 3. The food score (distance to food - punishment for remaining food)
        return score + ghost_score + food_score


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.get_score()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, eval_fn='score_evaluation_function', depth='2'):
        super().__init__()
        self.index = 0  # Pacman is always agent index 0
        self.evaluation_function = util.lookup(eval_fn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action from the current game_state using self.depth
        and self.evaluation_function.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
        Returns a list of legal actions for an agent
        agent_index=0 means Pacman, ghosts are >= 1

        game_state.generate_successor(agent_index, action):
        Returns the successor game state after an agent takes an action

        game_state.get_num_agents():
        Returns the total number of agents in the game

        game_state.is_win():
        Returns whether or not the game state is a winning state

        game_state.is_lose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        # We need to Traverse our minmax tree, we do this iteratively by setting up a stack
        initial_depth = 0
        initial_index = 0

        def minimax(agent_index, depth, state):
            # Check if terminal state. game over or max depth
            if depth == self.depth or game_state.is_lose() or game_state.is_win():
                return self.evaluation_function(state)

            # If Pacman (MAX) plays:
            if agent_index == 0:
                return max_value(depth, state)
            else:
                return min_value(agent_index, depth, state)

        def max_value(depth, state):
            # Initializes max with -inf
            value = -inf
            recommended_action = None
            actions = state.get_legal_actions(0)
            if not actions:
                return self.evaluation_function(state)

            for action in actions:
                state_new = state.generate_successor(0, action)
                # We now go one level deeper for each agent.
                # We start by agent 1 and then go through the agents in the min_value function
                new_value = minimax(1, depth, state_new)
                if new_value > value:
                    value = new_value
                    recommended_action = action
            # If were at the root, we're finished and can return the action
            if depth == 0:
                return recommended_action
            # If not we return the max_value
            return value

        def min_value(agent_index, depth, state):
            # Initializes min with inf
            value = inf

            number_of_agents = state.get_num_agents()
            actions = state.get_legal_actions(agent_index)

            if not actions:
                return self.evaluation_function(state)
            next_agent = agent_index + 1
            for action in actions:
                state_new = state.generate_successor(agent_index, action)
                if next_agent == number_of_agents:  # Were finished with all agents, now we go one level deeper
                    new_value = minimax(0, depth + 1, state_new)
                else:
                    new_value = minimax(next_agent, depth, state_new)
                if new_value < value:
                    value = new_value
            # Now we have the min value fo the ghosts
            # As this is the move of the opposite players, we cant be at the root, hence we dont have to return an action
            return value

        return minimax(initial_index, initial_depth, game_state)


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"
        initial_alpha = -inf
        initial_beta = inf
        initial_depth = 0
        initial_index = 0

        def alphabeta(agent_index, depth, state, alpha, beta):
            # Check if terminal state. game over or max depth
            if depth == self.depth or game_state.is_lose() or game_state.is_win():
                return self.evaluation_function(state)

            # If Pacman (MAX) plays:
            if agent_index == 0:
                return max_value(depth, state, alpha, beta)
            else:
                return min_value(agent_index, depth, state, alpha, beta)

        def max_value(depth, state, alpha, beta):
            # Node is -inf at first
            value = -inf
            recommended_action = None
            actions = state.get_legal_actions(0)
            if not actions:
                return self.evaluation_function(state)

            for action in actions:
                state_new = state.generate_successor(0, action)
                # We now go one level deeper for each agent.
                # We start by agent 1 and then go through the agents in the min_value function
                new_value = alphabeta(1, depth, state_new, alpha, beta)
                if new_value > value:
                    value = new_value
                    recommended_action = action
                # If the value is smaller tha alpha, we can prune the coming section of the tree
                if value > beta:
                    return value
                if value > alpha:
                    alpha = value
            # If were at the root, we're finished and can return the action
            if depth == 0:
                return recommended_action
            # If not we return the max_value
            return value

        def min_value(agent_index, depth, state, alpha, beta):
            # Initializes min with inf
            value = inf
            number_of_agents = state.get_num_agents()
            actions = state.get_legal_actions(agent_index)

            if not actions:
                return self.evaluation_function(state)

            next_agent = agent_index + 1
            for action in actions:
                state_new = state.generate_successor(agent_index, action)
                if next_agent == number_of_agents:  # Were finished with all agents, now we go one level deeper
                    new_value = alphabeta(0, depth + 1, state_new, alpha, beta)
                else:
                    new_value = alphabeta(next_agent, depth, state_new, alpha, beta)
                if new_value < value:
                    value = new_value
                # If the value is smaller tha alpha, we can prune the coming section of the tree
                if value < alpha:
                    return value
                # If not we update beta
                if value < beta:
                    beta = value

            # Now we have the min value fo the ghosts
            # As this is the move of the opposite players, we cant be at the root, hence we dont have to return an action
            return value

        return alphabeta(initial_index, initial_depth, game_state, initial_alpha, initial_beta)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluation_function

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raise_not_defined()


def better_evaluation_function(current_game_state):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raise_not_defined()


# Abbreviation
better = better_evaluation_function
