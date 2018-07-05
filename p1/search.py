# search.py
# ---------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

"""
In search.py, you will implement generic search algorithms which are called 
by Pacman agents (in searchAgents.py).
"""

import util

class SearchProblem:
  """
  This class outlines the structure of a search problem, but doesn't implement
  any of the methods (in object-oriented terminology: an abstract class).
  
  You do not need to change anything in this class, ever.
  """
  
  def startingState(self):
    """
    Returns the start state for the search problem 
    """
    util.raiseNotDefined()

  def isGoal(self, state): #isGoal -> isGoal
    """
    state: Search state

    Returns True if and only if the state is a valid goal state
    """
    util.raiseNotDefined()

  def successorStates(self, state): #successorStates -> successorsOf
    """
    state: Search state
     For a given state, this should return a list of triples, 
     (successor, action, stepCost), where 'successor' is a 
     successor to the current state, 'action' is the action
     required to get there, and 'stepCost' is the incremental 
     cost of expanding to that successor
    """
    util.raiseNotDefined()

  def actionsCost(self, actions): #actionsCost -> actionsCost
    """
      actions: A list of actions to take
 
     This method returns the total cost of a particular sequence of actions.  The sequence must
     be composed of legal moves
    """
    util.raiseNotDefined()
           

def tinyMazeSearch(problem):
  """
  Returns a sequence of moves that solves tinyMaze.  For any other
  maze, the sequence of moves will be incorrect, so only use this for tinyMaze
  """
  from game import Directions
  s = Directions.SOUTH
  w = Directions.WEST
  return  [s,s,w,s,w,w,s,w]

def depthFirstSearch(problem):
  """
  Search the deepest nodes in the search tree first [p 85].
  
  Your search algorithm needs to return a list of actions that reaches
  the goal.  Make sure to implement a graph search algorithm [Fig. 3.7].
  
  To get started, you might want to try some of these simple commands to
  understand the search problem that is being passed in:
  
  print "Start:", problem.startingState()
  print "Is the start a goal?", problem.isGoal(problem.startingState())
  print "Start's successors:", problem.successorStates(problem.startingState())
  """

  """ function GRAPH-SEARCH(problem) returns a solution, or failure
    initialize the frontier using the initial state of problem
    initialize the explored set to be empty
    loop do
      if the frontier is empty then return failure
      choose a leaf node and remove it from the frontier
      if the node contains a goal state then return the corresponding solution
      add the node to the explored set
      expand the chosen node, adding the resulting nodes to the frontier
        only if not in the frontier or explored set
  """
  
  print "Start:", problem.startingState()
  print "Is the start a goal?", problem.isGoal(problem.startingState())
  print "Start's successors:", problem.successorStates(problem.startingState())

  # Creates a new stack (LIFO)
  frontier = util.Stack()
  explored = []
  actionslist = []
  state = problem.startingState()
  frontier.push((state, [], None))

  while True:
    if frontier.isEmpty():
      return None
    currentState = frontier.pop()
    if problem.isGoal(currentState[0]):
      print("WE HAVE FOUND DE WAY!")
      print("PATH:", currentState[1])
      return currentState[1]
    explored.append(currentState[0])
    #actionslist.append(currentState[1])
    successors = problem.successorStates(currentState[0])
    for tup in successors:
      #print("SUCCESSORS",tup)
      if tup[0] in explored or tup[0] in frontier.list:
        continue
      else:
        #print("pushed in tuple:", tup)
        list2 = currentState[1][:]
        list2.append(tup[1])
        #print("LIST2:", list2)
        frontier.push((tup[0], list2 ,tup[2]))
        #print("STACK:", frontier.list)
    #print("-------------")
  
  
  util.raiseNotDefined()


def breadthFirstSearch(problem):
  '''
  "Search the shallowest nodes in the search tree first. [p 81]"
  Implementation of BFS using a queue data structure from Util.py
  startingState: A position tuple (x,y) in PositionalSearchProblem.
      A tuple with the position and a set of visited corners ((x,y), set()) in FourCornersProblem.
  frontier: A queue that contains tuples containing (startingState, path, cost)
      path: Path taken to get to that state. ["West", "North", ..., etc]
      cost: Integer cost taken to get to that state. 
  explored: A list that contains states that have been visited. [(x,y), (x1,y1) ..., (xn,yn)]
  frontierList: A list that contains states current in the queue. [(x,y), (x1,y1) ..., (xn,yn)]
  list2: A copy of the list to append the next action. Used a copy of list because when appending
      to the current path of a given state overwrote the whole queue paths.  
  '''
  #print("Goal:", problem.goal)
  if problem.isGoal(problem.startingState()):
    return []
  frontier = util.Queue()
  frontier.push((problem.startingState(), [], 0))
  explored = []
  while True:
    if frontier.isEmpty():
      return None
    state, path, currentCost = frontier.pop()
    explored.append(state)
    successors = problem.successorStates(state)
    for child in successors:
      #print("child:", child)
      frontierList = [x[0] for x in frontier.list]
      if child[0] in frontierList:
        continue
      if child[0] in explored:
        continue
      else:
        if problem.isGoal(child[0]):
          list2 = path[:]
          list2.append(child[1])
          return list2
        list2 = path[:]
        list2.append(child[1])
        frontier.push((child[0], list2, child[2]))
  #Does not reach here
  util.raiseNotDefined()

# Original implementation before generalized to work with later problems.
'''
  if problem.isGoal(problem.startingState()):
    return []
  frontier = util.Queue()
  frontier.push((problem.startingState(), [], 0))
  explored = []
  while True:
    if frontier.isEmpty():
      return None
    node = frontier.pop()
    explored.append(node[0])
    successors = problem.successorStates(node[0])
    for child in successors:
      frontierList = [x[0] for x in frontier.list]
      if child[0] in frontierList:
        continue
      if child[0] in explored:
        continue
      else:
        if problem.isGoal(child[0]):
          list2 = node[1][:]
          list2.append(child[1])
          return list2
        list2 = node[1][:]
        list2.append(child[1])
        frontier.push((child[0], list2, child[2]))
'''
  
      
def uniformCostSearch(problem):
  '''
  "Search the node of least total cost first. "
  Implementation of UCS using a PriorityQueue data structure from Util.py
  startingState: A position tuple (x,y) in PositionalSearchProblem.
      A tuple with the position and a set of visited corners ((x,y), set()) in FourCornersProblem.
  frontier: A Priority Queue that contains tuples containing (startingState, path) sorted by lowest cost first.
      path: Path taken to get to that state. ["West", "North", ..., etc]
  explored: A list that contains states that have been visited. [(x,y), (x1,y1) ..., (xn,yn)]
  frontierList: A list that contains states current in the queue. [(x,y), (x1,y1) ..., (xn,yn)]
  cost: A dict holding the cost of states and the cost to get to that state. Key: state | Value: cost
  list2: A copy of the list to append the next action. Used a copy of list because when appending
      to the current path of a given state overwrote the whole queue paths.  
  '''

  startingNode = (problem.startingState(), [])
  frontier = util.PriorityQueue()
  frontier.push(startingNode, 0)
  explored = []
  frontierList = []
  frontierList.append(problem.startingState())
  cost = {}
  cost[problem.startingState()] = 0
  #print("test:", cost[problem.startingState()])
  while True:
    if frontier.isEmpty():
      return None
    node = frontier.pop()
    frontierList.remove(node[0])
    #print("-----------------------")
    #print("node:", node)
    #print("node[0]:", node[0])
    #print("node[1]:", node[1])
    if problem.isGoal(node[0]):
      return node[1]
    explored.append(node[0])
    successors = problem.successorStates(node[0])
    #print("Heap:", frontier.heap)
    for child in successors:
      priority = cost[node[0]]
      newPriority = priority + child[2]
      #print("child:", child[0])
      #print("Priority:", priority)
      #print("newPriority:", newPriority)
      #print("explored:", explored)
      #print("In explored:", child[0] in explored)
      #print("frontierList:", frontierList)
      #print("In frontier:", child[0] in frontierList)
      if child[0] not in frontierList:
        if child[0] not in explored:
          list2 = node[1][:]
          list2.append(child[1])
          frontier.push((child[0], list2), newPriority)
          #print("Pushed child: ", child[0], list2, newPriority)
          frontierList.append(child[0])
          cost[child[0]] = newPriority
        elif child[0] in frontierList and newPriority < cost[child[0]]:
          # Might make nodes excessive cause dont get rid of original
          list2 = node[1][:]
          list2.append(child[1])
          #print("Pushed lowcost child:", child)
          frontier.push((child[0], list2), newPriority)
          frontierList.append[child[0]]
          cost[child[0]] = newPriority
        
  util.raiseNotDefined()

def nullHeuristic(state, problem=None):
  """
  A heuristic function estimates the cost from the current state to the nearest
  goal in the provided SearchProblem.  This heuristic is trivial.
  """
  return 0

def aStarSearch(problem, heuristic=nullHeuristic):
  '''
  "Search the node that has the lowest combined cost and heuristic first."
  Implementation of A* using a PriorityQueue data structure from Util.py. Similar to UCS.
      Differences lie within the queue cost. f(n) = g(n) + h(n)
          g(n): cost to get to that state
          h(n): heuristic value | Defaulted to nullHeuristic that always returns 0.
  startingState: A position tuple (x,y) in PositionalSearchProblem.
      A tuple with the position and a set of visited corners ((x,y), set()) in FourCornersProblem.
  frontier: A Priority Queue that contains tuples containing (startingState, path) sorted by lowest cost first.
      path: Path taken to get to that state. ["West", "North", ..., etc]
  explored: A list that contains states that have been visited. [(x,y), (x1,y1) ..., (xn,yn)]
  frontierList: A list that contains states current in the queue. [(x,y), (x1,y1) ..., (xn,yn)]
  cost: A dict holding the cost of states and the cost to get to that state. Key: state | Value: cost
  list2: A copy of the list to append the next action. Used a copy of list because when appending
      to the current path of a given state overwrote the whole queue paths.  
  '''

  state = problem.startingState()
  startingNode = (state, [])
  frontier = util.PriorityQueue()
  frontier.push(startingNode, 0)
  explored = []
  frontierList = []
  frontierList.append(problem.startingState())
  cost = {}
  #print("problem.startingState():,", problem.startingState())
  cost[repr(problem.startingState())] = 0
  while True:
    if frontier.isEmpty():
      return None
    state, path = frontier.pop()
    frontierList.remove(state)
    if problem.isGoal(state):
      return path
    explored.append(state)
    successors = problem.successorStates(state)
    for child in successors:
      priority = cost[repr(state)]
      estPriority = priority + heuristic(child[0], problem)
      actualPriority = priority + child[2]
      if child[0] not in frontierList:
        if child[0] not in explored:
          list2 = path[:]
          list2.append(child[1])
          frontier.push((child[0], list2), estPriority)
          frontierList.append(child[0])
          cost[repr(child[0])] = actualPriority
        elif child[0] in frontierList and estPriority < cost[child[0]]:
          # Might make nodes excessive cause dont get rid of original
          list2 = path[:]
          list2.append(child[1])
          frontier.push((child[0], list2), estPriority)
          frontierList.append[child[0]]
          cost[repr(child[0])] = actualPriority
  
  util.raiseNotDefined()
  
# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
