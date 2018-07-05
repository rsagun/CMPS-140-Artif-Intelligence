# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import *
from learningAgents import ReinforcementAgent
from featureExtractors import *

import random,util,math

class QLearningAgent(ReinforcementAgent):
  """
    Q-Learning Agent

    Functions you should fill in:
      - getQValue
      - getAction
      - getValue
      - getPolicy
      - update

    Instance variables you have access to
      - self.epsilon (exploration prob)
      - self.alpha (learning rate)
      - self.discountRate (discount rate)

    Functions you should use
      - self.getLegalActions(state)
        which returns legal actions
        for a state
  """
  def __init__(self, **args):
    "You can initialize Q-values here..."
    ReinforcementAgent.__init__(self, **args)
    self.qValues = util.Counter()
    # Sample of V(s) = R(s,Policy,sP)+ discount*V(sP)
    # Update of V(s) = (1-a)V(s) + (alpha)*sample
    # Same Update = V(s) = V(s) + alpha(sample-V(s))
    


  def getQValue(self, state, action):
    """
      Returns Q(state,action)
      Should return 0.0 if we never seen
      a state or (state,action) tuple
    """
    """Description:
    Returns the qValue of the state action pair.
    """
    """ YOUR CODE HERE """
    return self.qValues[(state,action)]
    """ END CODE """



  def getValue(self, state):
    """
      Returns max_action Q(state,action)
      where the max is over legal actions.  Note that if
      there are no legal actions, which is the case at the
      terminal state, you should return a value of 0.0.
    """
    """Description:
    If the state is the terminal state return 0. Else get all legal actions
    and for each action calculate a qValue for that action paired with the state
    and store all of the values in a list. Returns the max of the qValues.
    """
    """ YOUR CODE HERE """
    # Call getQValue here
    qValueList = []
    if len(self.getLegalActions(state)) == 0:
      return 0.0
    for action in self.getLegalActions(state):
      qValue = self.getQValue(state,action)
      qValueList.append(qValue)
    return max(qValueList)
    """ END CODE """

  def getPolicy(self, state):
    """
      Compute the best action to take in a state.  Note that if there
      are no legal actions, which is the case at the terminal state,
      you should return None.
    """
    """Description:
    For a given state, get legal actions. For each action add the qValue to a list and find the max.
    Use the max to find the index in which the best action lies in the list. Return that action.
    """
    """ YOUR CODE HERE """
    # Call getQValue here
    qValueList = []
    if len(self.getLegalActions(state)) == 0:
      return None
    for action in self.getLegalActions(state):
      qValue = self.getQValue(state,action)
      qValueList.append(qValue)
    maxVal = max(qValueList)
    maxIndex = qValueList.index(maxVal)
    return self.getLegalActions(state)[maxIndex]
    """ END CODE """

  def getAction(self, state):
    """
      Compute the action to take in the current state.  With
      probability self.epsilon, we should take a random action and
      take the best policy action otherwise.  Note that if there are
      no legal actions, which is the case at the terminal state, you
      should choose None as the action.

      HINT: You might want to use util.flipCoin(prob)
      HINT: To pick randomly from a list, use random.choice(list)
    """
    # Pick Action
    legalActions = self.getLegalActions(state)
    action = None

    """Description:
    If terminal state return no action. Agent chooses the best action according to the policy.
    There is a random chance (epsilon) the agent will take a random action.
    """
    """ YOUR CODE HERE """
    if len(legalActions) == 0:
      return None
    if util.flipCoin(self.epsilon):
      action = random.choice(legalActions)
    else:
      action = self.getPolicy(state)
    """ END CODE """

    return action

  def update(self, state, action, nextState, reward):
    """
      The parent class calls this to observe a
      state = action => nextState and reward transition.
      You should do your Q-Value update here

      NOTE: You should never call this function,
      it will be called on your behalf
    """
    """Description:
    Updates qValues for a state action pair. Uses the formula from the class slides.
    """
    """ YOUR CODE HERE """
    sample = reward + (self.discountRate*self.getValue(nextState))
    # Q(s,a) = (1-alpha)*self.qValues[state,action) + alpha*sample
    self.qValues[(state,action)] = self.qValues[(state,action)] + self.alpha*(sample - self.qValues[(state,action)])
    """ END CODE """

class PacmanQAgent(QLearningAgent):
  "Exactly the same as QLearningAgent, but with different default parameters"

  def __init__(self, epsilon=0.05,gamma=0.8,alpha=0.2, numTraining=0, **args):
    """
    These default parameters can be changed from the pacman.py command line.
    For example, to change the exploration rate, try:
        python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

    alpha    - learning rate
    epsilon  - exploration rate
    gamma    - discount factor
    numTraining - number of training episodes, i.e. no learning after these many episodes
    """
    args['epsilon'] = epsilon
    args['gamma'] = gamma
    args['alpha'] = alpha
    args['numTraining'] = numTraining
    self.index = 0  # This is always Pacman
    QLearningAgent.__init__(self, **args)

  def getAction(self, state):
    """
    Simply calls the getAction method of QLearningAgent and then
    informs parent of action for Pacman.  Do not change or remove this
    method.
    """
    action = QLearningAgent.getAction(self,state)
    self.doAction(state,action)
    return action


class ApproximateQAgent(PacmanQAgent):
  """
     ApproximateQLearningAgent

     You should only have to overwrite getQValue
     and update.  All other QLearningAgent functions
     should work as is.
  """
  def __init__(self, extractor='IdentityExtractor', **args):
    self.featExtractor = util.lookup(extractor, globals())()
    PacmanQAgent.__init__(self, **args)

    # You might want to initialize weights here.
    self.weights = util.Counter()

  def getQValue(self, state, action):
    """
      Should return Q(state,action) = w * featureVector
      where * is the dotProduct operator
    """
    """Description:
    [Enter a description of what you did here.]
    """
    """ YOUR CODE HERE """
    '''
    features = self.featExtractor.getFeatures(state,action)
    qVal = 0
    qVal += self.weights[features] * features[(state,action)]
    return qVal
    '''
    util.raiseNotDefined()
    """ END CODE """

  def update(self, state, action, nextState, reward):
    """
       Should update your weights based on transition
    """
    """Description:
    [Enter a description of what you did here.]
    """
    """ YOUR CODE HERE """
    '''
    features = self.featExtractor.getFeatures(state,action)
    correction = (reward + (self.discountRate * self.weights[nextState])) - self.getQValue(state,action)
    self.weights[feature] = self.weights[feature] + (self.alpha * correction * features[(state,action)])
    '''
    util.raiseNotDefined()
    """ END CODE """

  def final(self, state):
    "Called at the end of each game."
    # call the super-class final method
    PacmanQAgent.final(self, state)

    # did we finish training?
    if self.episodesSoFar == self.numTraining:
      # you might want to print your weights here for debugging
      util.raiseNotDefined()
