# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from util import PriorityQueue
from collections import deque

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]
"""
The method depthFirstSearch takes a problem as input and returns a possible list of moves to move from initial state to goal state.
To find the path to goal state, nodes are expanded in depth first fashion.
An iterative implementation of DFS is used over a recursive approach because a recursive implementation can lead to stack-overflow error
"""
def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    "*** YOUR CODE HERE ***"
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    #print "Start's successors:", problem.getSuccessors(problem.getStartState())
    # Node expansion in the above line lead to failure of testcase!!
    print "Hello pacman"
    print problem.__class__
    # Initialise the list of moves
    dfsMovesList = []
    currentState = problem.getStartState()
    # Check if start state is the goal state.
    # If yes, return an empty list
    if (problem.isGoalState(currentState)):
        return dfsMovesList
    # Maintain a Fringe List
    # Fringe List is a Stack in DFS
    # When Goal state is found, the state, move pairs in fringeList will give the path from start state to goal state
    fringeList = []
    fringeList.append((currentState,None))
    # expandedList stores the list of successors of each state which are yet to be explored
    # If a state is not in expandedList, it means that all of its successors are yet to be explored
    # If the list corresponding to a state is empty, it means that all the successors of a state have been explored
    expandedList = {}
    isGoalFound = False
    # Search for the Goal state till goal is not reached and Fringe List has some nodes
    # If Goal is not found and Fringe List gets empty, no path exists from start state to goal state
    while ((not isGoalFound) and (len(fringeList) > 0)):
        # Get the top element of Fringe List (Stack)
        currentState, move = fringeList[-1]
        # If goal is found, get out of loop
        # The states,move tuples remaining in fringeList is the path from start state to goal state
        if problem.isGoalState(currentState):
            isGoalFound = True
            break

        # If current state is not in expanded list, add it to expanded list
        # Initialise the list of this state with all its non-expanded successors
        if currentState not in expandedList:
            expandedList[currentState] = []
            currentStateSuccessors = problem.getSuccessors(currentState)
            for successor, move, cost in currentStateSuccessors:
                if successor not in expandedList:
                    expandedList[currentState].append((successor, move))
        # If current state is in expanded list, check if any of its successors are not explored
        # If all the successors of this state have been explored, backtrack (by removing this node from fringe list)
        elif(len(expandedList[currentState]) == 0):
            fringeList.pop()
        # If current state has some un-expanded neighbors, remove the first neighbor and add it to the fringe list
        else:
            successor, move = expandedList[currentState].pop()
            fringeList.append((successor, move))

    # If goal is not found, return an empty list, signifying there is no path from start state to goal state
    if(not isGoalFound):
        return dfsMovesList
    # The (state, move) tuples remaining in Fringe List are on the path from start state to goal state
    # Pop elements from Fringe List and add the corresponding moves to dfsMovesList
    while(len(fringeList) > 1):
        currentState, move = fringeList.pop()
        dfsMovesList.insert(0, move)
    return dfsMovesList

"""
The method breadthFirstSearch takes a problem as input and returns a possible list of moves to move from initial state to goal state.
To find the path to goal state, nodes are expanded in breadth first fashion.
"""
def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    "*** YOUR CODE HERE ***"
    # Initialise the list of moves
    bfsMovesList = []
    currentState = problem.getStartState()
    # Check if start state is the goal state.
    # If yes, return an empty list
    if (problem.isGoalState(currentState)):
        return bfsMovesList
    # Maintain a Fringe List
    # Fringe List is a Queue in BFS
    fringeList = deque([])
    fringeList.append(currentState)
    # Maintain which states have already been visited to avoid exploring them multiple times
    visitedStates = {}
    # Maintain the parent (predecessor) of states
    parentOfStates = {}
    visitedStates[currentState] = True
    isGoalFound = False
    # Search for the Goal state till goal is not reached and Fringe List has some nodes
    # If Goal is not found and Fringe List gets empty, no path exists from start state to goal state
    while ((not isGoalFound) and (len(fringeList) > 0)):
        # Get the first node from Fringe List
        currentState = fringeList.popleft()
        # If goal is found, get out of loop
        # Use the parent information to track the path from start state to goal state
        if problem.isGoalState(currentState):
            isGoalFound = True
            #visitedStates[successor] = True
            #parentOfStates[successor] = (currentState, move)
            #currentState = successor
            break
        # Add the non-visited neighbors of current state to Fringe List
        currentStateSuccessors = problem.getSuccessors(currentState)
        for successor, move, cost in currentStateSuccessors:
            if successor not in visitedStates:
                # Add successor to fringe list, uodate that it is visited and store its parent information
                fringeList.append(successor)
                visitedStates[successor] = True
                parentOfStates[successor] = (currentState, move)

    # If goal is not found, return an empty list, signifying there is no path from start state to goal state
    if (not isGoalFound):
        return bfsMovesList
    # Find path from start state to goal state by looking at predecessors
    while currentState in parentOfStates:
        currentState, move = parentOfStates[currentState]
        bfsMovesList.insert(0, move)
        #print move
    return bfsMovesList

"""
The method uniformCostSearch takes a problem as input and returns a possible list of moves to move from initial state to goal state.
To find the path to goal state, nodes are explored in increasing order of cost.
"""
def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    "*** YOUR CODE HERE ***"
    print problem.__class__
    # Initialise the list of moves
    ucsMovesList = []
    currentState = problem.getStartState()
    # Check if start state is the goal state.
    # If yes, return an empty list
    if(problem.isGoalState(currentState)):
        return ucsMovesList
    # Maintain a Fringe List
    # Fringe List is a Priority Queue in UCS
    fringeList = PriorityQueue()
    fringeList.push(currentState, 0)
    # Maintain the minimum cost incurred to reach the states
    minCostToReachState = {}
    minCostToReachState[currentState] = 0
    # Maintain the parent (predecessor) of states
    parentOfStates = {}
    isGoalFound = False
    # Search for the Goal state till goal is not reached and Fringe List has some nodes
    # If Goal is not found and Fringe List gets empty, no path exists from start state to goal state
    while((not isGoalFound) and (not fringeList.isEmpty())):
        # Get the first node from Fringe List
        currentState = fringeList.pop()
        currentStateCost = minCostToReachState[currentState]
        # If goal is found, get out of loop
        # Use the parent information to track the path from start state to goal state
        if problem.isGoalState(currentState):
            print 'Goal Found'
            isGoalFound = True
            break
        currentStateSuccessors = problem.getSuccessors(currentState)
        # Explore the successors of current state
        for successor, move, cost in currentStateSuccessors:
            # If this is the first time successor is seen, initialise the least cost path to reach successor
            if successor not in minCostToReachState:
                fringeList.push(successor, currentStateCost + cost)
                minCostToReachState[successor] = currentStateCost + cost
                parentOfStates[successor] = (currentState, move)
            # If a lesser cost path is found to reach successor, update this information in Fringe List, Dictionary and parent information
            elif(currentStateCost + cost < minCostToReachState[successor]):
                fringeList.update(successor, currentStateCost + cost)
                minCostToReachState[successor] = currentStateCost + cost
                parentOfStates[successor] = (currentState, move)

    # If goal is not found, return an empty list, signifying there is no path from start state to goal state
    if (not isGoalFound):
        return ucsMovesList
    # Find path from start state to goal state by looking at predecessors
    while currentState in parentOfStates:
        currentState, move = parentOfStates[currentState]
        ucsMovesList.insert(0, move)

    return ucsMovesList
    #util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

"""
The method aStarSearch takes a problem as input and returns a possible list of moves to move from initial state to goal state.
To find the path to goal state, nodes are explored in increasing order of 
(cost incurred to reach current state + Heuristic cost to reach goal state from current state).
"""
def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    print problem.__class__
    # Initialise the list of moves
    aStarMovesList = []
    currentState = problem.getStartState()
    # Check if start state is the goal state.
    # If yes, return an empty list
    if(problem.isGoalState(currentState)):
        return aStarMovesList
    # Maintain a Fringe List
    # Fringe List is a Priority Queue in UCS
    # Priority of a state in fringe list is (cost incurred to reach this state + heuristic cost to reach goal state from this state)
    fringeList = PriorityQueue()
    currentStateActualCost = 0
    currentStateHeuristicCost = heuristic(currentState, problem)

    fringeList.push(currentState, currentStateHeuristicCost)
    # Maintain a Dictionary of actual cost incurred to reach states : G(state)
    actualCostOfStates = {}
    actualCostOfStates[currentState] = 0
    # Maintain a Dictionary of estimated cost incurred to reach states : F(state)
    # F(state) = G(state) + H(state)
    estimatedCostOfStates = {}
    estimatedCostOfStates[currentState] = currentStateHeuristicCost
    # Maintain the parent (predecessor) of states
    parentOfStates = {}
    isGoalFound = False

    # Search for the Goal state till goal is not reached and Fringe List has some nodes
    # If Goal is not found and Fringe List gets empty, no path exists from start state to goal state
    while((not isGoalFound) and (not fringeList.isEmpty())):
        # Get the first node from Fringe List
        currentState = fringeList.pop()
        currentStateActualCost = actualCostOfStates[currentState]
        # If goal is found, get out of loop
        # Use the parent information to track the path from start state to goal state
        if(problem.isGoalState(currentState)):
            isGoalFound = True
            break
        currentStateSuccessors = problem.getSuccessors(currentState)
        # Explore the successors of current state
        for successor, move, cost in currentStateSuccessors:
            # If this is the first time successor is seen, initialise its F and G values
            heuristicCostOfSuccessor = heuristic(successor, problem)
            if successor not in estimatedCostOfStates:
                actualCostOfStates[successor] = currentStateActualCost + cost
                estimatedCostOfStates[successor] = actualCostOfStates[successor] + heuristicCostOfSuccessor
                fringeList.push(successor, estimatedCostOfStates[successor])
                parentOfStates[successor] = (currentState, move)
            # If a lesser F(successor) is found, update this information in Fringe List, F, G and parent information
            elif(currentStateActualCost + cost + heuristicCostOfSuccessor < estimatedCostOfStates[successor]):
                actualCostOfStates[successor] = currentStateActualCost + cost
                estimatedCostOfStates[successor] = actualCostOfStates[successor] + heuristicCostOfSuccessor
                fringeList.update(successor, estimatedCostOfStates[successor])
                parentOfStates[successor] = (currentState, move)

    # If goal is not found, return an empty list, signifying there is no path from start state to goal state
    if (not isGoalFound):
        return aStarMovesList
    # Find path from start state to goal state by looking at predecessors
    while currentState in parentOfStates:
        currentState, move = parentOfStates[currentState]
        aStarMovesList.insert(0, move)
    return aStarMovesList
    #util.raiseNotDefined()


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
