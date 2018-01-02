#do not modify the function names
#You are given L and M as input
#Each of your functions should return the minimum possible L value alongside the marker positions
#Or return -1,[] if no solution exists for the given L
from copy import deepcopy
import time

'''
This function takes current positions of marker on ruler, a dictionary of possible pairwise distances between current positions, and a new position.
It returns whether a marker can be placed on the new position pos such that it does not violate golomb ruler constraints 
'''
def IsAssignmentPossible(positions, pairwiseDistances, pos):
    currentPointDistances = {}
    # pos should not be equidistant from two existing points in positions
    for markedPosition in positions:
        currDistance = abs(pos - markedPosition)
        if (currDistance in pairwiseDistances) or (currDistance in currentPointDistances):
            return False
        currentPointDistances[currDistance] = True
    return True
'''
This function takes current positions of marker on ruler, a dictionary of possible pairwise distances between current positions, and a new position.
It updates the pairwise distances dictionary to include distances of new position pos with current marker positions
'''
def UpdatePairwiseDistances(positions, pairwiseDistances, pos):
    for markedPosition in positions:
        pairwiseDistances[abs(pos - markedPosition)] = True
    return

'''
This function takes current positions of marker on ruler, a dictionary of possible pairwise distances between current positions, and a new position.
It removes from the pairwiseDistances dictionary the distances of pos from current markings on the ruler
'''
def RemovePairwiseDistances(positions, pairwiseDistances, pos):
    for markedPosition in positions:
        pairwiseDistances.pop(abs(pos - markedPosition))
    return

'''
This function performs recursive backtracking.
It places one marker at a position, and checks whether this assignment leads to a solution.
If not, then it places the marker at a different position and tries again
Inputs: scale : 0/1 list which specifies the occupied positions on the ruler
positions : positions on the ruler where marker has been placed
pairwiseDistances : all pairs of distances between marker positions, stored in a dictionary for fast checks
index : The index of marker to be placed
M : Total number of markers to be placed
'''
def BackTrackingRec(scale, positions, pairwiseDistances, index, M):
    # If all markers have been successfully placed, return success
    if(M == index):
        return True
    # Find the starting position of new marker
    # We place marker in increasing order of positions
    startPos = 0
    for i in range(len(scale)-1,-1,-1):
        if(scale[i] == 1):
            startPos = i+1
            break
    for pos in range(startPos, len(scale)):
        # If the position is already occupied by another marker or the place has been ruled incosistent, skip placing marker here
        if(scale[pos] == 1):
            continue
        # If placing a marker here does not violate constraints, place a marker here and recurse
        if(IsAssignmentPossible(positions, pairwiseDistances, pos)):
            positions.append(pos)
            scale[pos] = 1
            # Add this postion to list of markers and add its distances with other markers in the dictionary
            UpdatePairwiseDistances(positions, pairwiseDistances, pos)
            isFurtherAssignmentConsistent = BackTrackingRec(scale, positions, pairwiseDistances, index+1, M)
            # If we can successfully place further markers, we have a solution.
            if(isFurtherAssignmentConsistent):
                return True
            # If this assignment did not lead to a solution, undo its changes
            positions.pop()
            RemovePairwiseDistances(positions, pairwiseDistances, pos)
            scale[pos] = 0
    return False

'''
This function takes Length L and number of markers M as input.
It returns the solution found using Plain Backtracking
'''
def FindBacktrackingSolution(L, M):
    pairwiseDistances = {}
    scale = [0 for i in range(L+1)]
    # scale[i] is 0 if a marker is not placed at i, else 1
    scale[0] = 1
    # positions stores the assignments i.e. markings on the ruler
    positions = [0]
    isSolutionPossible = BackTrackingRec(scale,positions,pairwiseDistances,1,M)
    if isSolutionPossible:
        return positions[-1],positions
    return -1,[]

#Your backtracking function implementation
def BT(L, M):
    "*** YOUR CODE HERE ***"
    sol = []
    l = -1
    # Keep decreasing length of ruler until we do not find a solution
    # The last length for which solution existed is the optimal length
    for i in xrange(L,M-1,-1):
    #for i in xrange(L,L+1):
        #print i
        l1,sol1 = FindBacktrackingSolution(i,M)
        if(l1 == -1):
            break
        else:
            #print 'Solution found for %d' %(l1),sol1
            l = l1
            sol = sol1
    return l,sol

'''
This function is for Forward Checking
If we place a marker at pos, then update the positions on ruler where a marker can not be placed according to constraints
Inputs:
remVals : boolean array, which signifies if a marker can be placed at a position or not
positions : current assignment of markers
pos : new position on scale. update positions on the scale where marker can not be placed according to constraints
pairwiseDistances : all pairs of distances between marker positions, stored in a dictionary for fast checks
'''
def UpdateRemainingVals(pairwiseDistances, remVals, positions, pos):
    # We can not place a new marker which is at a distance already in pairwise distances from pos
    for dist in pairwiseDistances:
        if(pos-dist >= 0):
            remVals[pos-dist] = False
        if(pos+dist < len(remVals)):
            remVals[pos+dist] = False
    currPositionDistancs = [abs(p - pos) for p in positions]
    # We can not place a marker at a position if it is equidistant from both pos, and a place in positions
    for dist in currPositionDistancs:
        for markedPosition in positions:
            if(markedPosition-dist >= 0):
                remVals[markedPosition-dist] = False
            if(markedPosition+dist < len(remVals)):
                remVals[markedPosition+dist] = False
    
'''
This function performs recursive backtracking with forward checking.
It places one marker at a position, and checks whether this assignment leads to a solution.
If yes, it modifies the places where a new marker can be placed and recursively tries to place a new marker
If not, then it places the marker at a different position and tries again
Inputs: remVals : boolean list which specifies the occupied positions on the ruler
positions : positions on the ruler where marker has been placed
pairwiseDistances : all pairs of distances between marker positions, stored in a dictionary for fast checks
index : The index of marker to be placed
M : Total number of markers to be placed
'''
def ForwardCheckingRec(pairwiseDistances, positions, remVals, index, M):
    #print index, positions, remVals
    # If all markers have been successfully placed, return success
    if(index == M):
        return True
    # If number of markers to place is less than available positions, FAIL
    if(sum(remVals) < M - index):
        return False
    # Find the starting position of new marker
    # We place marker in increasing order of positions
    startPos = positions[-1] + 1
    for pos in range(startPos, len(remVals)):
        # If the position is already occupied by another marker or the place has been ruled incosistent, skip placing marker here
        if(not remVals[pos]):
            continue
        # If placing a marker here does not violate constraints, place a marker here and recurse
        if(IsAssignmentPossible(positions, pairwiseDistances, pos)):
            #pairwiseDistancesCopy = deepcopy(pairwiseDistances)
            # Create inputs for next recursive call
            pairwiseDistancesCopy = {}
            for k,v in pairwiseDistances.iteritems():
                pairwiseDistancesCopy[k] = v
            UpdatePairwiseDistances(positions, pairwiseDistancesCopy, pos)
            #remValsCopy = deepcopy(remVals)
            remValsCopy = [v for v in remVals]
            UpdateRemainingVals(pairwiseDistancesCopy, remValsCopy, positions, pos)
            positions.append(pos)
            remValsCopy[pos] = False
            # Try placing next markers
            isFurtherAssignmentConsistent = ForwardCheckingRec(pairwiseDistancesCopy, positions, remValsCopy, index+1, M)
            # If we can successfully place further markers, we have a solution.
            if(isFurtherAssignmentConsistent):
                return True
            positions.pop()
    return False    

'''
This function takes Length L and number of markers M as input.
It returns the solution found using Backtracking with Forward checking
'''
def FindForwardCheckingSolution(L,M):
    pairwiseDistances = {}
    # remVals = [[True for i in range(L+1)] for i in range(M)]
    # for i in range(1,M):
        # remVals[i][0] = False
    # numRemVals = [L for i in range(M)]
    # numRemVals[0] = L+1
    positions = [0]
    remVals = [True for i in range(L+1)]
    remVals[0] = False
    isSolutionPossible = ForwardCheckingRec(pairwiseDistances, positions, remVals, 1, M)
    if isSolutionPossible:
        return positions[-1],positions
    return -1,[]

#Your backtracking+Forward checking function implementation
def FC(L, M):
    "*** YOUR CODE HERE ***"
    sol = []
    l = -1
    # Keep decreasing length of ruler until we do not find a solution
    # The last length for which solution existed is the optimal length
    for i in xrange(L,M-1,-1):
    #for i in xrange(L,L+1):
        #print i
        #start = time.time()
        l1,sol1 = FindForwardCheckingSolution(i,M)
        #end = time.time()
        #print 'FC For %d,%d' %(i,M), (end-start)
        if(l1 == -1):
            break
        else:
            l = l1
            sol = sol1
    return l,sol

'''
This method performs Constraint propogation until all arcs are consistent
Inputs:
remVals : 2D Matrix of booleans, whose i,j th entry describes if a marker i can be placed at position j
positions : positions on the ruler where marker has been placed
pairwiseDistances : all pairs of distances between marker positions, stored in a dictionary for fast checks
index : The index of next marker to be placed. Last marker was placed at index-1 and we need to do constraint propogation relative to that asignment
pos : last marker was placed at this position
'''
def UpdateRemainingValsConstraintProp(pairwiseDistances, remVals, pos, index, positions):
    # No marker can be placed at pos because last marker was placed at pos
    for i in range(index, len(remVals)):
        remVals[i][pos] = False
    #print 'After step 1'
    #printList(remVals)
    # Use arcs as a Queue
    arcs = []
    M = len(remVals)
    L = len(remVals[0])
    # We have a fully connected constraint graph.
    # Perform arc-consisteny across all edges
    for i in range(index,M):
        for j in range(i+1,M):
            arcs.append((i,j))
            arcs.append((j,i))

    while(len(arcs) > 0):
        # Pop an element from queue
        # Make this arc consistent
        pos1, pos2 = arcs.pop(0)
        #print 'Making %d,%d consistent' %(pos1,pos2)
        # Keep track if pos1 loses a valie
        # If yes, add its neighbours to the queue so as to perform further constaint propagation
        loseValue = False
        # Check that for all values of pos1, there exists a legal assignment of pos2
        for i in range(len(remVals[pos1])):
            # If current is not a legal value of pos1, skip
            if(not remVals[pos1][i]):
                continue
            # Calculate and store distances of pos1 from all points where markers have been placed
            currPositionDistances = {}
            for markedPosition in positions:
                currPositionDistances[abs(markedPosition-i)] = True
            # Check whether a legal assignment of pos2 is possible if we place a marker at pos1
            isConsistentValuePresent = False
            for j in range(len(remVals[pos2])):
                # If this is not a legal value of pos2, skip
                if(not remVals[pos2][j]):
                    continue
                # Check if placing a marker at j satisfies all the constraints
                areAllDistancesUnique = True
                nextPositionDistances = {}
                for markedPosition in positions:
                    currDistance = abs(j - markedPosition)
                    if ((currDistance in pairwiseDistances) or (currDistance in currPositionDistances) or (currDistance in nextPositionDistances)):
                        areAllDistancesUnique = False
                        break
                    nextPositionDistances[currDistance] = True
                # If placing a marker at j is possible given we place a marker at i, we have found a consistent assignment for i
                # Hence we need not remove i from list of values for pos1
                if(areAllDistancesUnique):
                    isConsistentValuePresent = True
                    break
            # If no consistent value is present for pos2 given we place a marker at i, remove i from pos1
            # And add neighbours of pos1 for further arc-consistency
            if(not isConsistentValuePresent):
                #print 'No consistent value present for %d in %d' %(i,pos2)
                loseValue = True
                remVals[pos1][i] = False
        # If pos1 loses a value, add its neighbours for further constraint propagation
        if(loseValue):
            for i in range(index,M):
                arcs.append((pos1,i))
                arcs.append((i,pos1))

'''
This method prints remaing values for each variable
Wrote for debugging
'''
def printList(remValsCopy):
    # for v in remValsCopy:
        # l = []
        # for i in range(len(v)):
            # if(v[i]):
                # l.append(i)
        # print l
    x = 1
'''
This method prints remaing values for each variable
same as printList
'''
def printList2(remValsCopy):
    for v in remValsCopy:
        l = []
        for i in range(len(v)):
            if(v[i]):
                l.append(i)
        print l

'''
This function performs recursive backtracking with constraint propogation.
It places one marker at a position, and checks whether this assignment leads to a solution.
If yes, it modifies the places where a new marker can be placed for every variable and recursively tries to place a new marker
If not, then it places the marker at a different position and tries again
Inputs: remVals : boolean list which specifies the occupied positions on the ruler
positions : positions on the ruler where marker has been placed
pairwiseDistances : all pairs of distances between marker positions, stored in a dictionary for fast checks
index : The index of marker to be placed
M : Total number of markers to be placed
'''
def ConstraintPropagationSolutionRec(pairwiseDistances, positions, remVals, index, M):
    #print index, positions, remVals
    #print positions   
    # If all markers have been successfully placed, return success
    if(index == M):
        return True
    # If any variable has 0 remaining legal values, FAIL
    for i in range(index, len(remVals)):
        #print remVals[i]
        if(sum(remVals[i]) == 0):
            return False
    # Find the starting position of new marker
    # We place marker in increasing order of positions
    startPos = positions[-1] + 1
    #for pos in range(len(remVals[index])):
    for pos in range(startPos,len(remVals[index])):
        # if(index == 1):
            # print 'here'
            # #printList2(remVals)
        # If the position is already occupied by another marker or the place has been ruled incosistent, skip placing marker here
        if(not remVals[index][pos]):
            continue
        # If placing a marker here does not violate constraints, place a marker here and recurse
        if(IsAssignmentPossible(positions, pairwiseDistances, pos)):
            #pairwiseDistancesCopy = deepcopy(pairwiseDistances)
            # Create inputs for next recursive call
            pairwiseDistancesCopy = {}
            for k,v in pairwiseDistances.iteritems():
                pairwiseDistancesCopy[k] = v
            UpdatePairwiseDistances(positions, pairwiseDistancesCopy, pos)
            #remValsCopy = deepcopy(remVals)
            remValsCopy = [[True for i in range(len(remVals[0]))] for i in range(len(remVals))]
            for i in range(len(remVals)):
                for j in range(len(remVals[i])):
                    remValsCopy[i][j] = remVals[i][j]
            #remValsCopy = [v for v in remVals]
            #print pairwiseDistancesCopy
            #print 'Before constraint prop %d %d' %(index,pos)
            #print 'Current assignments: ', positions
            #printList(remValsCopy)
            # Perform constraint propagation before next recursive call
            UpdateRemainingValsConstraintProp(pairwiseDistancesCopy, remValsCopy, pos, index+1, positions)
            #print 'After constraint prop %d %d' %(index,pos)
            #printList(remValsCopy)
            positions.append(pos)
            remValsCopy[index][pos] = False
            # Try placing next markers
            isFurtherAssignmentConsistent = ConstraintPropagationSolutionRec(pairwiseDistancesCopy, positions, remValsCopy, index+1, M)
            # If we can successfully place further markers, we have a solution.
            if(isFurtherAssignmentConsistent):
                return True
            positions.pop()
    return False 

'''
This function takes Length L and number of markers M as input.
It returns the solution found using Backtracking with Constraint Propogation
'''
def FindConstraintPropagationSolution(L, M):
    pairwiseDistances = {}
    remVals = [[True for i in range(L+1)] for i in range(M)]
    #print remVals
    for i in range(1,M):
        remVals[i][0] = False
    positions = [0]
    isSolutionPossible = ConstraintPropagationSolutionRec(pairwiseDistances, positions, remVals,1,M)
    if isSolutionPossible:
        return positions[-1],positions
    return -1,[]


#Bonus: backtracking + constraint propagation
def CP(L, M):
    sol = []
    l = -1
    # Keep decreasing length of ruler until we do not find a solution
    # The last length for which solution existed is the optimal length
    for i in xrange(L,M-1,-1):
    #for i in xrange(L,L+1):
        #print i
        l1,sol1 = FindConstraintPropagationSolution(i,M)
        if(l1 == -1):
            break
        else:
            l = l1
            sol = sol1
    return l,sol

if __name__ == '__main__':
    for i in range(1,11):
        start = time.time()
        l,p = BT(60,i)
        end = time.time()
        print 'BT', l, p, (end-start)
        start = time.time()
        l,p = FC(60,i)
        end = time.time()
        print 'FC', l, p, (end-start)
    for i in range(1,9):
        start = time.time()
        l,p = CP(60,i)
        end = time.time()
        print 'CP', l, p, (end-start)


