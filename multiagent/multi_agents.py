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

inf=99999999
max_depth=2
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
        chosen_index = random.choice(best_indices) # Pick randomly among the best

        "Add more of your code here if you want to"
        print(legal_moves[chosen_index])
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
        self.index = 0 # Pacman is always agent index 0
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
        eval_action = self.get_evaluation_and_action(game_state, 0)
        print(eval_action)
        return eval_action[1]
        
    def get_evaluation_and_action(self, game_state, current_depth):
        if current_depth == self.depth or game_state.is_lose() or game_state.is_win():
            return (self.evaluation_function(game_state)-current_depth+1, 'Stop') 
        else :
            # for every legal move that pacman (agent 0) has
            pacman_moves = game_state.get_legal_actions(0)
            #print(pacman_moves)
            max_evaluation = -inf
            best_action = pacman_moves[0]
            #print(pacman_moves)
            # for each legal move of pacman
            for pacman_move in pacman_moves:
                min_evaluation = inf
                all_ghost_moves = []
                # we take a look at every possible outcome
                nr_agents= game_state.get_num_agents()
                for index in range(1, nr_agents):
                    all_ghost_moves.append(game_state.get_legal_actions(index))
                    
                all_combinations_ghost_moves = list(product(*all_ghost_moves))
                # and we select the one with the lowest evaluation
                #print(all_combinations_ghost_moves)
                for combination_ghost_moves in all_combinations_ghost_moves:
                    new_game_state = copy.deepcopy(game_state) 
                    
                    if new_game_state.is_lose() == False or new_game_state.is_win() == False:
                        new_game_state = new_game_state.generate_successor(0,pacman_move)

                    # check the evaluation for each 
                    for ghost_index,ghost_move in zip(range(1,nr_agents),combination_ghost_moves):
                        if new_game_state.is_lose() == False or new_game_state.is_win() == False:
                            new_game_state = new_game_state.generate_successor(ghost_index,ghost_move)

                    #print(new_game_state.get_score())
                    #print(new_game_state)
                    new_eval_action = self.get_evaluation_and_action(new_game_state, current_depth+1)
                    current_evaluation = new_eval_action[0]
                    #print(current_evaluation)
                    #return the move that achieves the highest minimum evaluation
                    if current_evaluation < min_evaluation:
                        min_evaluation = current_evaluation
                if max_evaluation < min_evaluation : 
                    best_action = pacman_move
                    max_evaluation = min_evaluation
            eval_and_action = [max_evaluation, best_action]        
            return eval_and_action
            
    
    

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluation_function
        """
        "*** YOUR CODE HERE ***"
        
        eval_action = self.get_evaluation_and_action(game_state, alpha, beta)
        return eval_action[1]
        util.raise_not_defined()
        
    def get_evaluation_and_action(self, game_state, alpha, beta):
        if self.depth == max_depth or game_state.is_lose() or game_state.is_win():
            return (self.evaluation, 'Stop') 
        else :
            # for every legal move that pacman (agent 0) has
            pacman_moves = game_state.get_legal_actions(0)
            #####include the possibility when they are neighbors
            max_evaluation = -inf
            # for each legal move of pacman
            for pacman_move in pacman_moves:
                min_evaluation = inf
                all_ghost_moves = []
                # we take a look at every possible outcome
                for index in range(1, game_state.get_num_agents()):
                    all_ghost_moves.append(game_state.get_legal_action(index))
                all_combinations_ghost_moves = list(product(*all_ghost_moves))
                # and we select the one with the lowest evaluation
                for combination_ghost_moves in all_combinations_ghost_moves:
                    new_game_state = game_state.generate_successor(0,pacman_move)
                    index = 1
                    # check the evaluation for each 
                    for ghost_move in combination_ghost_moves:
                        new_game_state = new_game_state.generate_successor(index,ghost_move)
                        index+=1
                    new_minimax_agent = MinimaxAgent('score_evaluation_function',depth+1)
                    new_eval_action = new_minimax_agent.get_evaluation_and_action()
                    current_evaluation = new_eval_action[0]
                    print(current_evaluation)
                    #return the move that achieves the highest minimum evaluation
                    if current_evaluation < min_evaluation:
                        min_evaluation = current_evaluation
                if max_evaluation < min_evaluation : 
                    best_action = pacman_move
                    max_evaluation = min_evaluation
            eval_and_action = [max_evaluation, best_action] 
            
            return eval_and_action
    

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
