# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
  """
      * Please read learningAgents.py before reading this.*

      A ValueIterationAgent takes a Markov decision process
      (see mdp.py) on initialization and runs value iteration
      for a given number of iterations using the supplied
      discount factor.
  """
  def __init__(self, mdp, discountRate = 0.9, iters = 100):
    """
      Your value iteration agent should take an mdp on
      construction, run the indicated number of iterations
      and then act according to the resulting policy.

      Some useful mdp methods you will use:
          mdp.getStates()
          mdp.getPossibleActions(state)
          mdp.getTransitionStatesAndProbs(state, action)
          mdp.getReward(state, action, nextState)
    """
    self.mdp = mdp
    self.discountRate = discountRate
    self.iters = iters
    self.values = util.Counter() # A Counter is a dict with default 0

    """Description:
    At each iteration update values for each state. Keeps track of old values (k-1)
    and new values (k) in two separate counters. Runs for certain amount of iterations.
    """
    """ YOUR CODE HERE """
    # Where does the new values go? Either U or UPrime.
    uValues = self.values.copy()
    for i in range(0,iters+1):
      self.values = uValues.copy()
      for state in mdp.getStates():
        #print("State:", state)
        #print("Actions Possible:", mdp.getPossibleActions(state))
        qValueList = []
        if mdp.getPossibleActions(state) == ():
          # What do I return here? Return Reward?
          qValueList.append(0)
        else:
          for action in mdp.getPossibleActions(state):
            qValue = self.getQValue(state,action)
            #print("qValue:", qValue)
            qValueList.append(qValue)
        #print("List QVal:", qValueList)
        uValues[state] = max(qValueList)    
    """ END CODE """

  def getValue(self, state):
    """
      Return the value of the state (computed in __init__).
    """
    return self.values[state]

    """Description:
    Returns the value of the state stored in the values counter. Did not change anything.
    """
    """ YOUR CODE HERE """
    # Getting the value is returning the max value of a certain state
    # Produce actions here for each state and send those to the qValue
    # Store q values into a list to get max?
    # return the max value back to the initial call
    '''
    qValueList = []
    legalActions = self.mdp.getPossibleActions(state)
    if len(legalActions) == 0:
      return 0
    for action in legalActions:
      qValueList.append(self.getQValue(state,action))
    print("qValList:", qValueList)
    print("max value:", max(qValueList))
    return max(qValueList)
    '''
    """ END CODE """

  def getQValue(self, state, action):
    """
      The q-value of the state action pair
      (after the indicated number of value iteration
      passes).  Note that value iteration does not
      necessarily create this quantity and you may have
      to derive it on the fly.
    """
    """Description:
    For each nextState and probability calculate the qValue. for the given state action pair.
    """
    """ YOUR CODE HERE """
    qVal = 0
    for nextState,prob in self.mdp.getTransitionStatesAndProbs(state,action):
      qVal += prob*(self.mdp.getReward(state,action,nextState)+(self.discountRate*self.values[nextState]))
    return qVal
    
    # Using the current state and action, find the q value for that pair
    # q value will be sent back up to getValue which sticks it into a list
    # this function will get called as many actions as there are found in getValue
    # return q value computed from given state and action
    '''
    qVal = 0
    for nextState,prob in self.mdp.getTransitionStatesAndProbs(state, action):
      #print("state:", state)
      #print("action:", action)
      #print("valueNext:", self.values[nextState])
      #print("reward:", self.mdp.getReward(state,action,nextState))
      qVal = qVal + (prob * (self.mdp.getReward(state,action,nextState) + (self.discountRate * self.values[nextState])))
      #print("qVal:", qVal)
    return qVal
    '''
    """ END CODE """

  def getPolicy(self, state):
    """
      The policy is the best action in the given state
      according to the values computed by value iteration.
      You may break ties any way you see fit.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return None.
    """

    """Description:
    Finds the optimal action to take in a given state.
    """
    """ YOUR CODE HERE """
    # Uses the current state and returns the optimal policy.
    # Use the value stored in values for the given state to find the max
    # get the action that the value derives from and return that action
    #print("State:", state)
    qValueList = []
    if self.mdp.getPossibleActions(state) == ():
      return None
    for action in self.mdp.getPossibleActions(state):
      qValue = self.getQValue(state,action)
      #print("qValue:", qValue)
      qValueList.append(qValue)
    maxVal = max(qValueList)
    maxIndex = qValueList.index(maxVal)
    return self.mdp.getPossibleActions(state)[maxIndex]

    
    '''
    actionValues = []
    legalActions = self.mdp.getPossibleActions(state)
    if len(legalActions) == 0:
      return None
    else:
      currentValue = float("-inf")
      for action in legalActions:
        actionValues.append(self.getQValue(state,action))
      listMax = max(actionValues)
      listIndex = actionValues.index(listMax)
      return legalActions[listIndex]
    '''
    
    
    """ END CODE """

  def getAction(self, state):
    "Returns the policy at the state (no exploration)."
    return self.getPolicy(state)
