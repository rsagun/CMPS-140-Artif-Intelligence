# multiAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """


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
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.

    Print out these variables to see what you're getting, then combine them
    to create a masterful evaluation function.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    oldPosition = currentGameState.getPacmanPosition()
    newPosition = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood().asList()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    successorGhostPos = successorGameState.getGhostPosition(1)
    "*** YOUR CODE HERE ***"
    totalEval = 0
    # If Pacman can get food in its next state increase priority
    for food in oldFood:
      if newPosition == food:
        totalEval = totalEval + 5
    # If pacman gets too close to the ghost deem action dangeous
    # and make sure Pacman does not take at less have to.
    # Uses manhattan distance to determin how close pacman is to the ghost.
    if manhattanDist(newPosition, successorGhostPos) < 2:
      totalEval = totalEval - 50
    else:
      totalEval = totalEval + 2
    # If pacman moves towards the closest food deem action valuable
    # This ensures he does not take random moves that prolong the game.
    # Uses manhattan distance to determine closeness
    closestFoodDist = float("inf")
    closestFood = [(0,0)]
    for food in oldFood:
      currentDist = manhattanDist(oldPosition, food)
      if currentDist < closestFoodDist:
        closestFoodDist = currentDist
        closestFood[0] = food
    #print("Mdist:" ,manhattanDist(newPosition, closestFood[0]))
    #print("CDist:" ,closestFoodDist)
    if manhattanDist(newPosition, closestFood[0]) < closestFoodDist:
      totalEval = totalEval + 5
    #print("closestFood:" ,closestFood[0])
    return totalEval

def manhattanDist(xy1, xy2):
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
  
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
    self.treeDepth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (question 2)
  """
  
  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.treeDepth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    # Populate a list with the min values corresponding to each legal action
    # Find the max of those, and remember the index of it.
    # Return the element from the list of legal actions with the same index as the max
    minValueList = []
    depth = self.treeDepth
    startGhostIndex = 1
    pacmanIndex = 0
    pacmanActions = gameState.getLegalActions(pacmanIndex)
    ghostCount = gameState.getNumAgents()-1
    for action in pacmanActions:
      minValue = self.getMin(self.Result(gameState, action, pacmanIndex), depth, startGhostIndex)
      minValueList.append(minValue)
    maxValue = max(minValueList)
    return pacmanActions[minValueList.index(maxValue)]

  def getMin(self, gameState, depth, ghostIndex):
    if self.terminalTest(gameState, depth):
      return self.evaluationFunction(gameState)
    v = float("inf")
    ghostActions = gameState.getLegalActions(ghostIndex)
    ghostAmount = gameState.getNumAgents() - 1 # Does not include Pacman
    # If there are still more ghost to simulate get the min of the next ghost.
    for action in ghostActions:
      if ghostIndex < ghostAmount:
        v = min(v, self.getMin(self.Result(gameState, action, ghostIndex), depth, ghostIndex+1))
    # If there are not anymore ghost to check. New depth has been aquired.
      else:
        v = min(v, self.getMax(self.Result(gameState, action, ghostIndex), depth-1))
    return v

  def getMax(self, gameState, depth):
    startGhostIndex = 1
    pacmanIndex = 0
    if self.terminalTest(gameState, depth):
      return self.evaluationFunction(gameState)
    v = float("-inf")
    pacmanActions = gameState.getLegalActions(pacmanIndex)
    for action in pacmanActions:
      v = max(v, self.getMin(self.Result(gameState, action, pacmanIndex), depth, startGhostIndex))
    return v

  def Result(self, gameState, action, agentIndex):
    return gameState.generateSuccessor(agentIndex, action)

  def terminalTest(self, gameState, depth):
    # Terminates if either find a state that is a win or loss, or when search his the maximum depth.
    # Depth is decremented in the previous iteration in which it is checked.
    # Terminal test then decrement and pass variable in.
    oldFood = gameState.getFood().asList()
    pacmanPosition = gameState.getPacmanPosition()
    ghostLocations = gameState.getGhostPositions()
    # Winning Condition
    if oldFood == []:
      return True
    # Losing Condition
    for ghostPosition in ghostLocations:
      if pacmanPosition == ghostPosition:
        return True
    # Depth Satisfaction
    if depth == 0:
      return True
    return False
      
class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (question 3)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.treeDepth
      and self.evaluationFunction.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      Directions.STOP:
        The stop direction, which is always legal

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game
    """
    "*** YOUR CODE HERE ***"
    # Populate a list with the min values corresponding to each legal action
    # Find the max of those, and remember the index of it.
    # Return the element from the list of legal actions with the same index as the max
    minValueList = []
    depth = self.treeDepth
    startGhostIndex = 1
    pacmanIndex = 0
    a = float("-inf")
    b = float("inf")
    pacmanActions = gameState.getLegalActions(pacmanIndex)
    ghostCount = gameState.getNumAgents()-1
    for action in pacmanActions:
      minValue = self.getMin(self.Result(gameState, action, pacmanIndex), depth, startGhostIndex, a, b)
      minValueList.append(minValue)
    maxValue = max(minValueList)
    return pacmanActions[minValueList.index(maxValue)]

  def getMin(self, gameState, depth, ghostIndex, a, b):
    if self.terminalTest(gameState, depth):
      return self.evaluationFunction(gameState)
    v = float("inf")
    ghostActions = gameState.getLegalActions(ghostIndex)
    ghostAmount = gameState.getNumAgents() - 1 # Does not include Pacman
    # If there are still more ghost to simulate get the min of the next ghost.
    for action in ghostActions:
      if ghostIndex < ghostAmount:
        v = min(v, self.getMin(self.Result(gameState, action, ghostIndex), depth, ghostIndex+1, a, b))
        if v <= a:
          return v
        b = min(b,v)
    # If there are not anymore ghost to check. New depth has been aquired.
      else:
        v = min(v, self.getMax(self.Result(gameState, action, ghostIndex), depth-1, a, b))
        if v <= a:
          return v
        b = min(b,v)
    return v

  def getMax(self, gameState, depth, a, b):
    startGhostIndex = 1
    pacmanIndex = 0
    if self.terminalTest(gameState, depth):
      return self.evaluationFunction(gameState)
    v = float("-inf")
    pacmanActions = gameState.getLegalActions(pacmanIndex)
    for action in pacmanActions:
      v = max(v, self.getMin(self.Result(gameState, action, pacmanIndex), depth, startGhostIndex, a, b))
      if v >= b:
        a = max(a,v)
    return v

  def Result(self, gameState, action, agentIndex):
    return gameState.generateSuccessor(agentIndex, action)

  def terminalTest(self, gameState, depth):
    # Terminates if either find a state that is a win or loss, or when search his the maximum depth.
    # Depth is decremented in the previous iteration in which it is checked.
    # Terminal test then decrement and pass variable in.
    oldFood = gameState.getFood().asList()
    pacmanPosition = gameState.getPacmanPosition()
    ghostLocations = gameState.getGhostPositions()
    # Winning Condition
    if oldFood == []:
      return True
    # Losing Condition
    for ghostPosition in ghostLocations:
      if pacmanPosition == ghostPosition:
        return True
    # Depth Satisfaction
    if depth == 0:
      return True
    return False

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (question 4)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.treeDepth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    "*** YOUR CODE HERE ***"

    # Populate a list with the min values corresponding to each legal action
    # Find the max of those, and remember the index of it.
    # Return the element from the list of legal actions with the same index as the max
    '''
    minValueList = []
    depth = self.treeDepth
    startGhostIndex = 1
    pacmanIndex = 0
    pacmanActions = gameState.getLegalActions(pacmanIndex)
    ghostCount = gameState.getNumAgents()-1
    for action in pacmanActions:
      minValue = self.getMin(self.Result(gameState, action, pacmanIndex), depth, startGhostIndex, a, b)
      minValueList.append(minValue)
    maxValue = max(minValueList)
    return pacmanActions[minValueList.index(maxValue)]
    #gameState.generateSuccessor(agentIndex, action):
    #Returns the successor game state after an agent takes an action
    '''
    expectedUtilities = []
    depth = self.treeDepth
    startGhostIndex = 1
    pacmanIndex = 0
    pacmanActions = gameState.getLegalActions(pacmanIndex)
    agentIndex = gameState.getNumAgents()
    for action in pacmanActions:
      successorState = gameState.generateSuccessor(pacmanIndex, action)
      expectedUtility = self.value(self.Result(gameState, action, pacmanIndex), startGhostIndex, depth)
      expectedUtilities.append(expectedUtility)
    maxUtility = max(expectedUtilities)
    return pacmanActions[expectedUtilities.index(maxUtility)]

    
  def value(self, gameState, agentIndex, depth):
    if self.terminalTest(gameState, depth):
      #print("Terminal reached")
      return self.getTerminal(gameState)
    elif agentIndex == 0:
      return self.maxValue(gameState, depth)
    elif agentIndex > 0:
      return self.expValue(gameState, agentIndex, depth)
    print("Could not find specified node.")
    exit()
    return None

  def maxValue(self, gameState, depth):
    values = []
    pacmanIndex = 0
    pacmanActions = gameState.getLegalActions(pacmanIndex)
    #print("depth:", depth)
    #print("pacman actions:", pacmanActions)
    for action in pacmanActions:
      successorState = gameState.generateSuccessor(pacmanIndex, action)
      v = self.value(successorState, pacmanIndex+1, depth)
      values.append(v)
    return max(values)
  
  def expValue(self, gameState, ghostIndex, depth):
    values = []
    weights = []
    ghostActions = gameState.getLegalActions(ghostIndex)
    ghostAmount = gameState.getNumAgents() - 1 # Does not include Pacman
    ghostAmount = gameState.getNumAgents() - 1 # Does not include Pacman
    # If there are still more ghost to simulate get the min of the next ghost.
    for action in ghostActions:
      if ghostIndex < ghostAmount:
        v = self.value(self.Result(gameState, action, ghostIndex), ghostIndex+1, depth)
        values.append(v)
    # If there are not anymore ghost to check. New depth has been aquired.
      else:
        v = self.value(self.Result(gameState, action, ghostIndex), 0, depth-1)
        values.append(v)      
    return self.expectation(values, weights)

  def expectation(self, values, weights):
    #if len(values) != len(weights):
      #print("There is not a 1 to 1 mapping of values and probabilities")
      #exit()
    expectedValue = 0.0
    nodeCount = len(values)
    for i in range(0, nodeCount):
      expectedValue = expectedValue + values[i]
    return expectedValue/nodeCount       

  def getTerminal(self, gameState):
    return self.evaluationFunction(gameState)

  def terminalTest(self, gameState, depth):
    # Terminates if either find a state that is a win or loss, or when search his the maximum depth.
    # Depth is decremented in the previous iteration in which it is checked.
    # Terminal test then decrement and pass variable in.
    oldFood = gameState.getFood().asList()
    pacmanPosition = gameState.getPacmanPosition()
    ghostLocations = gameState.getGhostPositions()
    # Winning Condition
    if oldFood == []:
      return True
    # Losing Condition
    for ghostPosition in ghostLocations:
      if pacmanPosition == ghostPosition:
        return True
    # Depth Satisfaction
    if depth == 0:
      return True
    return False

  def Result(self, gameState, action, agentIndex):
    return gameState.generateSuccessor(agentIndex, action)

def betterEvaluationFunction(currentGameState):
  """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
  """
  "*** YOUR CODE HERE ***"
  currentPosition = currentGameState.getPacmanPosition()
  currentFood = currentGameState.getFood()
  currentFoodList = currentGameState.getFood().asList()
  ghostIndexes = currentGameState.getNumAgents()
  pacmanActions = currentGameState.getLegalPacmanActions()
  numFood = currentGameState.getNumFood()
  ghostPositions = currentGameState.getGhostPositions()

  '''
  Food Distances
  GhostDistances
  Number of Foods
  Number of Ghost
  Ghost Distances
  '''

  total = 0
  closestGhostDist = float("inf")
  for ghostPos in ghostPositions:
    dist = manhattanDist(currentPosition, ghostPos)
    if dist < closestGhostDist:
      closestGhostDist = dist
  print(closestGhostDist)
  total = total + closestGhostDist

  closestFoodDist = float("inf")
  for food in currentFoodList:
    foodDist = manhattanDist(currentPosition, food)
    if foodDist < closestFoodDist:
      closestFoodDist = foodDist
  total = total + closestFoodDist

  

  # If gameState is a win or lose deem really good or really bad.
  if currentGameState.isLose():
    total = float("-inf")
  if currentGameState.isWin():
    total = float("inf")
  ''' 
  successorList = []
  for action in pacmanActions:
    successor = currentGameState.generatePacmanSuccessor(action)
    successorList.append(successor)
  for successor in successorList:
  ''' 
  
  '''
  totalEval = currentGameState.getScore()
  succStates = []
  succPacmanPos = []
  pacmanActions = currentGameState.getLegalPacmanActions()
  for action in pacmanActions:
    successor = currentGameState.generatePacmanSuccessor(action)
    succStates.append(successor)
  for succ in succStates:
    newPacmanPos = succ.getPacmanPosition()
    succPacmanPos.append(newPacmanPos)
  for newPos in succPacmanPos:
    if currentFood[newPos[0]][newPos[1]]:
      totalEval = totalEval + 10

  furthestFoodDist = float("-inf")
  for food in currentFoodList:
    foodDist = manhattanDist(currentPosition, food)
    if foodDist > furthestFoodDist:
      furthestFoodDist = foodDist
  print("TEBefore foodDist:", totalEval)
  totalEval = totalEval - furthestFoodDist
  print("TE FoodDist:", totalEval)

  # Finds the closest ghost and subtracts based on manhattan distance away    
  closestGhostDist = float("inf")
  for i in range(1, ghostIndexes):
    ghostDist = manhattanDist(currentPosition, currentGameState.getGhostPosition(i))
    if ghostDist < closestGhostDist:
      closestGhostDist = ghostDist
  print("TEBefore ghostDist:", totalEval)
  totalEval = totalEval - closestGhostDist
  print("TE ghostDist:", totalEval)

  # If gameState is a win or lose deem really good or really bad.
  if currentGameState.isLose():
    totalEval = float("-inf")
  if currentGameState.isWin():
    totalEval = float("inf")
  #((1 - (abs(.5 - temp)) - .5) / .5)
  print(totalEval)
  '''
  '''
  # If pacman gets too close to the ghost deem action dangeous
  # and make sure Pacman does not take at less have to.
  # Uses manhattan distance to determin how close pacman is to the ghost.
  for i in range(1, ghostIndexes):
    if manhattanDist(currentPosition, currentGameState.getGhostPosition(i)) < 2:
      totalEval = totalEval - 10
    else:
      totalEval = totalEval + 1
  # If gameState is a win or lose deem really good or really bad.
  if currentGameState.isLose():
    totalEval = float("-inf")
  if currentGameState.isWin():
    totalEval = float("inf")

  # Less Food is closer to an end game.
  totalEval = totalEval - numFood  
  
  # If pacman moves towards the closest food deem action valuable
  # Uses manhattan distance to determine closeness
  closestFoodDist = float("inf")
  closestFood = [(0,0)]
  for food in currentFood:
    currentDist = manhattanDist(currentPosition, food)
    if currentDist < closestFoodDist:
      closestFoodDist = currentDist
  totalEval = totalEval - closestFoodDist #Two closest food?
  
  # Pacman favors states that approach 0 food.
  foodLeft = len(currentFood)
  totalEval = totalEval - foodLeft
  '''
  
  return total

# Abbreviation
better = betterEvaluationFunction

class ContestAgent(MultiAgentSearchAgent):
  """
    Your agent for the mini-contest
  """

  def getAction(self, gameState):
    """
      Returns an action.  You can use any method you want and search to any depth you want.
      Just remember that the mini-contest is timed, so you have to trade off speed and computation.

      Ghosts don't behave randomly anymore, but they aren't perfect either -- they'll usually
      just make a beeline straight towards Pacman (or away from him if they're scared!)
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

