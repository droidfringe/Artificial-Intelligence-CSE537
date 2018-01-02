# multiAgents.py
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


from util import manhattanDistance
from game import Directions
import random, util
import sys
#from pacman import GameState

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """
    def ManhattanDistanceBetweenTwoPoints(self, position1, position2):
        return abs(position1[0] - position2[0]) + abs(position1[1] - position2[1])

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #print successorGameState.__class__
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood().asList()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        MAX_EVAL = 10**6
        MIN_EVAL = -1*10**6
        newNumFood = successorGameState.getNumFood()
        currentFood = currentGameState.getFood()
        # If all the food is eaten in successor state, choose this action!!
        if newNumFood == 0:
            return MAX_EVAL
        evalFunction = 0
        # Calcluate pacman's manhattan distance from all food dots
        foodDistances = [self.ManhattanDistanceBetweenTwoPoints(foodPosition, successorGameState.getPacmanPosition())
                         for foodPosition in newFood]
        # Calcluate pacman's manhattan distance from all ghosts
        ghostDistances = [
            self.ManhattanDistanceBetweenTwoPoints(ghostState.getPosition(), newPos) for
            ghostState in newGhostStates]
        closestGhostDistance = min(ghostDistances)
        # If making this move leads to collision with a ghost,
        # avoid this move at all costs
        if closestGhostDistance == 0:
            return MIN_EVAL
        # If ghost is adjacant, move away!!
        elif closestGhostDistance == 1:
            return MIN_EVAL/10
        closestFoodDistance = min(foodDistances)
        # If food is available at new position, eat it!!
        # Can freely eat food because there is no ghost in vicinity
        if currentFood[newPos[0]][newPos[1]]:
            return MAX_EVAL
        elif closestFoodDistance == 0:
            return MAX_EVAL
        # If choosing this action brings a food item adjacant, choose this action
        elif closestFoodDistance == 1:
            return MAX_EVAL/10
        # Make sure to bring food closer
        evalFunction = MAX_EVAL/10
        evalFunction -= closestFoodDistance**2
        # If bringing food closer also brings a ghost close,
        # Lower the evaluation of this move
        if closestGhostDistance < 5:
            evalFunction -= closestGhostDistance**2
        # In presence of walls in the game, pacman sometimes
        # gets stuck in a loop and does not make progress
        # Can use actual closest distance to a food to avoid this
        return evalFunction


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        # Add bestMove field to keep track of best move to make
        # after running minimax or alpha-beta pruning or expectimax
        self.bestMove = None
        #self.nodesExpanded = 0


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def MiniMax(self, gameState, agentIdx, level, maxDepth):
        # TODO - Check if we have already hit a ghost
        # Check if a node is terminal (meaning no successors)
        # or we have reached max depth of exploration
        legalActions = gameState.getLegalActions(agentIdx)
        if(level == maxDepth or not legalActions):
            return (self.evaluationFunction(gameState))
        # Use numAgents and current level to keep track of
        # which agent needs to make a move
        numAgents = gameState.getNumAgents()
        childrenScores = []
        # For every legal action for the current agent,
        # make a move and call MiniMax recursively for next agent
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIdx, action)
            childrenScores.append((self.MiniMax(successorState, (agentIdx+1)%numAgents, level+1, maxDepth), action))
        # Check if we are at a max node or min node
        isMaxNode = (agentIdx%numAgents == 0)
        if isMaxNode:
            # Choose the max value of all actions at max node
            tup = max(childrenScores, key=lambda x:x[0])
            # When recursion stack unfolds, set the best move at first stage
            # The first node is max node
            # If it were a min node, update this in else condition
            self.bestMove = tup[1]
        else:
            # Choose the min value of all actions at max node
            tup = min(childrenScores, key=lambda x:x[0])
        return tup[0]

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        maxDepth = numAgents*self.depth
        minimaxVal = self.MiniMax(gameState, 0, 0, maxDepth)
        #print minimaxVal
        #print len(GameState.explored)
        return self.bestMove

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def AlphaBetaPruning(self, gameState, agentIdx, level, maxDepth, alpha, beta):
        # Use numAgents and current level to keep track of
        # which agent needs to make a move
        legalActions = gameState.getLegalActions(agentIdx)
        numAgents = gameState.getNumAgents()
        isMaxNode = (agentIdx % numAgents == 0)
        # Check if a node is terminal (meaning no successors)
        # or we have reached max depth of exploration
        if(level == maxDepth or not legalActions):
            val = self.evaluationFunction(gameState)
            return val

        childrenScores = []
        # For every legal action for the current agent,
        # make a move and call AlphaBetaPruning recursively for next agent
        # Prune the exploration of children if alpha > beta
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIdx, action)
            val = self.AlphaBetaPruning(successorState, (agentIdx + 1) % numAgents, level + 1, maxDepth, alpha, beta)
            childrenScores.append(val)
            # If current node is max node, update alpha
            if isMaxNode:
                if(val > alpha):
                    alpha = val
                    # Update the action which lead to best score
                    self.bestMove = action
            # If current node is min node, update beta
            else:
                if(val < beta):
                    beta = min(beta, val)
            if(alpha > beta):
                break
        # If it is a max node, return max value of all children
        if isMaxNode:
            return max(childrenScores)
        # If it is a min node, return min value of all children
        else:
            return min(childrenScores)

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        maxDepth = numAgents * self.depth
        alphabetaVal = self.AlphaBetaPruning(gameState, 0, 0, maxDepth, -sys.maxint, sys.maxint)
        #print alphabetaVal
        #print len(GameState.explored)
        return self.bestMove

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def ExpectiMax(self, gameState, agentIdx, level, maxDepth):
        legalActions = gameState.getLegalActions(agentIdx)
        # Check if a node is terminal (meaning no successors)
        # or we have reached max depth of exploration
        if(level == maxDepth or not legalActions):
            return (self.evaluationFunction(gameState))
        # Use numAgents and current level to keep track of
        # which agent needs to make a move
        numAgents = gameState.getNumAgents()
        childrenScores = []
        # For every legal action for the current agent,
        # make a move and call AlphaBetaPruning recursively for next agent
        for action in legalActions:
            successorState = gameState.generateSuccessor(agentIdx, action)
            childrenScores.append((self.ExpectiMax(successorState, (agentIdx+1)%numAgents, level+1, maxDepth),action))
        isMaxNode = (agentIdx%numAgents == 0)
        # If it is a max node, choose maximum of all the successors
        if isMaxNode:
            tup = max(childrenScores, key=lambda x:x[0])
            self.bestMove = tup[1]
            return tup[0]
        # If it is a min node, choose expected value of all the successors
        else:
            expectedVal = 0.0
            for t in childrenScores:
                expectedVal += t[0]
            expectedVal /= 1.0*len(childrenScores)
            return expectedVal

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        numAgents = gameState.getNumAgents()
        maxDepth = numAgents*self.depth
        expectiVal = self.ExpectiMax(gameState, 0, 0, maxDepth)
        #print expectiVal
        return self.bestMove
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction

